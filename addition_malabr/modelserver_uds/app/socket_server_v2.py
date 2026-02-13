import socket
import struct
import os
import threading
import time
import mmap
from concurrent.futures import ThreadPoolExecutor

from types_defs import Payload
from response import respond
from logger import get_logger
from router_v2 import Router

logger = get_logger(__name__)

MAX_HEADER = 8 * 1024

# Shared memory configuration (must match C++ exactly)
NUM_CHUNKS = 128
CHUNK_SIZE = 64 * 1024  # 64KB

# Ring layout matching C++ struct Ring:
# uint32_t head (4 bytes)
# uint32_t tail (4 bytes)  
# bool done (1 byte + 3 bytes padding)
# char chunks[NUM_CHUNKS][CHUNK_SIZE]

RING_STRUCT_FORMAT = "II?7x"  # head, tail, done, padding
RING_HEADER_SIZE = 16
RING_SIZE = RING_HEADER_SIZE + NUM_CHUNKS * CHUNK_SIZE

class Server:
    def __init__(self, sock_path: str, router: Router, max_workers=10):
        self.sock_path = sock_path
        self.router = router
        self.max_workers = max_workers
        self.server_socket = None
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

    def start(self):
        self._setup_socket()
        print(f"[SERVER] Listening on {self.sock_path} with {self.max_workers} workers.")
        try:
            while True:
                conn, _ = self.server_socket.accept()
                self.thread_pool.submit(self._handle_request, conn)
        except KeyboardInterrupt:
            print("\n[SERVER] Shutdown signal received.")
        finally:
            self._shutdown()

    def _setup_socket(self):
        if os.path.exists(self.sock_path):
            os.remove(self.sock_path)

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.sock_path)
        self.server_socket.listen(50)

    def _handle_request(self, conn: socket.socket):
        try:
            raw_len = self._recv_exact(conn, 4)
            header_len = struct.unpack("!I", raw_len)[0]

            header_bytes = self._recv_exact(conn, header_len)
            header_str = header_bytes.decode("utf-8")
            fb_id, label, payload_size_str, use_shm_str = header_str.split(",")
            use_shm = (use_shm_str == "1")
            payload_size = int(payload_size_str)

            if use_shm:
                payload_bytes = self._recv_via_shared_memory(conn, payload_size)
            else:
                payload_bytes = self._recv_exact(conn, payload_size)
                
            payload: Payload = {
                "label": label,
                "payload_bytes": payload_bytes,
                "fb_id": fb_id,
                "payload_size": payload_size,
            }

            handler = self.router.get_handler(label)
            if not handler:
                logger.error(f"[WORKER:{threading.get_ident()}] No handler for label {label}")
                respond(conn, "error", f"Unknown label: {label}")
                return

            handler(conn, payload)

        except Exception as e:
            logger.error(f"[Worker:{threading.get_ident()}] Error: {e}")
            try:
                respond(conn, "error", str(e))
            except Exception:
                pass
        finally:
            conn.close()

    def _recv_via_shared_memory(self, conn: socket.socket, expected_size: int) -> bytes:
        """
        Receive payload from shared memory ring buffer.
        
        Process:
        1. Receive shared memory FD from client via SCM_RIGHTS
        2. mmap the shared memory
        3. Read chunks from ring buffer as client writes them
        4. Return complete payload
        """
        logger.info(
            f"[WORKER:{threading.get_ident()}] "
            f"Receiving {expected_size} bytes via shared memory"
        )
        
        # Receive file descriptor
        shm_fd = self._recv_fd(conn)
        logger.info(f"[WORKER:{threading.get_ident()}] Received shm_fd={shm_fd}")
        
        # Map shared memory
        shm = mmap.mmap(shm_fd, RING_SIZE, mmap.MAP_SHARED, 
                        mmap.PROT_READ | mmap.PROT_WRITE)
        
        # Read payload from ring buffer
        payload_bytes = bytearray()
        bytes_read = 0
        chunks_read = 0
        
        while bytes_read < expected_size:
            # Read ring header (head, tail, done)
            shm.seek(0)
            head, tail, done = struct.unpack(
                RING_STRUCT_FORMAT,
                shm.read(RING_HEADER_SIZE)
            )
            
            # Check if data available
            if tail == head:
                if done and bytes_read >= expected_size:
                    break
                # No data yet, wait a bit
                time.sleep(0.001)  # 1ms
                continue
            
            # Read chunk at tail position
            chunk_offset = RING_HEADER_SIZE + tail * CHUNK_SIZE
            shm.seek(chunk_offset)
            chunk_data = shm.read(CHUNK_SIZE)
            
            # Calculate how much of this chunk is actual data
            remaining = expected_size - bytes_read
            chunk_len = min(CHUNK_SIZE, remaining)
            
            # Append to payload
            payload_bytes.extend(chunk_data[:chunk_len])
            bytes_read += chunk_len
            chunks_read += 1
            
            # Advance tail
            new_tail = (tail + 1) % NUM_CHUNKS
            shm.seek(struct.calcsize("I"))  # offset of tail field
            shm.write(struct.pack("I", new_tail))
            
            if chunks_read % 10 == 0:
                logger.debug(
                    f"[WORKER:{threading.get_ident()}] "
                    f"Read {bytes_read}/{expected_size} bytes "
                    f"({chunks_read} chunks)"
                )
        
        logger.info(
            f"[WORKER:{threading.get_ident()}] "
            f"Completed reading {bytes_read} bytes from shared memory "
            f"({chunks_read} chunks)"
        )
        
        # Cleanup
        shm.close()
        os.close(shm_fd)
        
        return bytes(payload_bytes)

    def _recv_fd(self, conn: socket.socket) -> int:
        """
        Receive a file descriptor via SCM_RIGHTS.
        
        Returns:
            File descriptor integer
        """
        msg, ancdata, flags, addr = conn.recvmsg(
            1,  # Dummy data size
            socket.CMSG_LEN(struct.calcsize("i"))
        )
        
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                fd = struct.unpack("i", cmsg_data[:4])[0]
                return fd
        
        raise RuntimeError("Did not receive file descriptor via SCM_RIGHTS")


    def _recv_exact(self, conn: socket.socket, n: int) -> bytes:
        """Read exactly n bytes from the socket into a preallocated buffer."""
        buf = bytearray(n)
        view = memoryview(buf)
        read = 0
        while read < n:
            chunk = conn.recv_into(view[read:], n - read)
            if chunk == 0:
                raise ConnectionError(
                    f"Socket closed unexpectedly, got {read}/{n} bytes"
                )
            read += chunk
        return buf

    def _shutdown(self):
        os.remove(self.sock_path)
        self.thread_pool.shutdown(wait=True)
        self.server_socket.close()
        print("[SERVER] Shutdown complete.")
