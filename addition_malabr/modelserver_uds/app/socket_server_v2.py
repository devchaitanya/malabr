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

# Configuration - Double Ring
NUM_CHUNKS = 128
CHUNK_SIZE = 64 * 1024  # 64KB
MAX_SLOTS = 64

# Single ring layout (part of double ring)
RING_STRUCT_FORMAT = "II?7x"  # head, tail, done, 7 bytes padding
RING_HEADER_SIZE = 16
SINGLE_RING_SIZE = RING_HEADER_SIZE + NUM_CHUNKS * CHUNK_SIZE

# Double ring layout: Ring A + Ring B + Control
DOUBLE_RING_SIZE = (SINGLE_RING_SIZE * 2) + 16  # 16 bytes for control atomics

# Pool layout
CONTROL_SIZE = 4096  # First 4KB for allocation bitmap
POOL_SIZE = CONTROL_SIZE + (MAX_SLOTS * DOUBLE_RING_SIZE)


class Server:
    def __init__(self, sock_path: str, router: Router, max_workers=10):
        self.sock_path = sock_path
        self.router = router
        self.max_workers = max_workers
        self.server_socket = None
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Create persistent pool at startup
        self.pool_name = f"/shm_pool_{os.getpid()}"
        self.pool_fd = None
        self.pool_mmap = None
        self._create_pool()

    def _create_pool(self):
        """Create persistent shared memory pool."""
        import posix_ipc
        
        try:
            # Remove old pool if exists
            try:
                posix_ipc.unlink_shared_memory(self.pool_name)
            except:
                pass
            
            # Create new pool
            shm = posix_ipc.SharedMemory(
                self.pool_name,
                flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,
                mode=0o666,
                size=POOL_SIZE
            )
            
            self.pool_fd = shm.fd
            self.pool_mmap = mmap.mmap(
                self.pool_fd,
                POOL_SIZE,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE
            )
            
            # Initialize allocation bitmap (all slots free)
            self.pool_mmap.seek(0)
            self.pool_mmap.write(struct.pack("Q", 0))  # 64-bit bitmap
            
            logger.info(f"[SERVER] Pool created: {self.pool_name}")
            logger.info(f"[SERVER] Pool size: {POOL_SIZE / (1024*1024):.1f} MB")
            logger.info(f"[SERVER] Max slots: {MAX_SLOTS}")
            logger.info(f"[SERVER] Double ring size: {DOUBLE_RING_SIZE / (1024*1024):.1f} MB")
            
        except Exception as e:
            logger.error(f"[SERVER] Pool creation failed: {e}")
            raise

    def allocate_slot(self) -> int:
        """Allocate a double-ring slot from pool."""
        self.pool_mmap.seek(0)
        bitmap = struct.unpack("Q", self.pool_mmap.read(8))[0]
        
        for slot_id in range(MAX_SLOTS):
            if not (bitmap & (1 << slot_id)):
                # Mark as allocated
                new_bitmap = bitmap | (1 << slot_id)
                self.pool_mmap.seek(0)
                self.pool_mmap.write(struct.pack("Q", new_bitmap))
                
                # Initialize double ring at this slot
                self._init_double_ring(slot_id)
                
                logger.info(f"[SERVER] Allocated slot {slot_id}")
                return slot_id
        
        raise RuntimeError("Pool exhausted - all slots in use")

    def free_slot(self, slot_id: int):
        """Free a slot back to pool."""
        self.pool_mmap.seek(0)
        bitmap = struct.unpack("Q", self.pool_mmap.read(8))[0]
        new_bitmap = bitmap & ~(1 << slot_id)
        self.pool_mmap.seek(0)
        self.pool_mmap.write(struct.pack("Q", new_bitmap))
        logger.info(f"[SERVER] Freed slot {slot_id}")

    def _init_double_ring(self, slot_id: int):
        """Initialize double ring at slot."""
        offset = CONTROL_SIZE + (slot_id * DOUBLE_RING_SIZE)
        
        # Initialize Ring A
        self.pool_mmap.seek(offset)
        self.pool_mmap.write(struct.pack("II?7x", 0, 0, False))  # head, tail, done
        
        # Initialize Ring B
        self.pool_mmap.seek(offset + SINGLE_RING_SIZE)
        self.pool_mmap.write(struct.pack("II?7x", 0, 0, False))
        
        # Initialize control atomics
        self.pool_mmap.seek(offset + (SINGLE_RING_SIZE * 2))
        self.pool_mmap.write(struct.pack("BB??4x", 0, 0, False, False))
        # active_write_ring, active_read_ring, swap_requested, swap_acknowledged

    def start(self):
        self._setup_socket()
        print(f"[SERVER] Listening on {self.sock_path} with {self.max_workers} workers.")
        print(f"[SERVER] Pool ready: {self.pool_name}")
        
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
        slot_id = None
        
        try:
            # Parse header
            raw_len = self._recv_exact(conn, 4)
            header_len = struct.unpack("!I", raw_len)[0]

            header_bytes = self._recv_exact(conn, header_len)
            header_str = header_bytes.decode("utf-8")
            fb_id, label, payload_size_str, use_shm_str = header_str.split(",")
            use_shm = (use_shm_str == "1")
            payload_size = int(payload_size_str)

            logger.info(
                f"[WORKER:{threading.get_ident()}] "
                f"Request: size={payload_size}, use_shm={use_shm}"
            )

            if use_shm:
                # Allocate slot from pool
                slot_id = self.allocate_slot()
                
                # Send pool name + slot_id to client
                pool_info = f"{self.pool_name},{slot_id}".encode()
                conn.sendall(struct.pack("!I", len(pool_info)))
                conn.sendall(pool_info)
                
                logger.info(f"[WORKER:{threading.get_ident()}] Assigned slot {slot_id}")
                
                # Receive via double ring
                payload_bytes = self._recv_via_shared_memory(slot_id, payload_size)
            else:
                payload_bytes = self._recv_exact(conn, payload_size)
                
            # Create payload and route to handler
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
            import traceback
            logger.error(traceback.format_exc())
            try:
                respond(conn, "error", str(e))
            except Exception:
                pass
        finally:
            # Free slot if allocated
            if slot_id is not None:
                self.free_slot(slot_id)
            conn.close()

    def _recv_via_shared_memory(self, slot_id: int, expected_size: int) -> bytes:
        """
        Receive payload from double-buffered ring.
        
        Uses slot from the pool instead of receiving FD.
        """
        logger.info(
            f"[WORKER:{threading.get_ident()}] "
            f"Receiving {expected_size} bytes via double ring (slot {slot_id})"
        )
        
        # Calculate offsets in pool
        double_ring_offset = CONTROL_SIZE + (slot_id * DOUBLE_RING_SIZE)
        ring_a_offset = double_ring_offset
        ring_b_offset = double_ring_offset + SINGLE_RING_SIZE
        control_offset = double_ring_offset + (SINGLE_RING_SIZE * 2)
        
        payload_bytes = bytearray()
        bytes_read = 0
        transfer_count = 0
        
        while bytes_read < expected_size:
            # Read which ring server should consume from
            self.pool_mmap.seek(control_offset + 1)  # active_read_ring offset
            active_read_idx = struct.unpack("B", self.pool_mmap.read(1))[0]
            
            ring_offset = ring_a_offset if active_read_idx == 0 else ring_b_offset
            ring_name = 'A' if active_read_idx == 0 else 'B'
            
            logger.info(
                f"[WORKER:{threading.get_ident()}] "
                f"Transfer {transfer_count}: Reading from Ring {ring_name}"
            )
            
            # Read from current ring
            ring_bytes = self._read_single_ring(ring_offset, expected_size - bytes_read)
            payload_bytes.extend(ring_bytes)
            bytes_read += len(ring_bytes)
            transfer_count += 1
            
            logger.info(
                f"[WORKER:{threading.get_ident()}] "
                f"Read {len(ring_bytes)} bytes, total: {bytes_read}/{expected_size}"
            )
            
            # Check if client requested swap
            self.pool_mmap.seek(control_offset + 2)  # swap_requested offset
            swap_requested = struct.unpack("?", self.pool_mmap.read(1))[0]
            
            if swap_requested and bytes_read < expected_size:
                logger.info(f"[WORKER:{threading.get_ident()}] Client requested swap")
                
                # Switch to the other ring
                new_read_idx = 1 - active_read_idx
                self.pool_mmap.seek(control_offset + 1)
                self.pool_mmap.write(struct.pack("B", new_read_idx))
                
                # Acknowledge swap
                self.pool_mmap.seek(control_offset + 3)  # swap_acknowledged offset
                self.pool_mmap.write(struct.pack("?", True))
                
                logger.info(
                    f"[WORKER:{threading.get_ident()}] "
                    f"Swapped to Ring {'A' if new_read_idx == 0 else 'B'}"
                )
        
        logger.info(
            f"[WORKER:{threading.get_ident()}] "
            f"Complete: {bytes_read} bytes in {transfer_count} transfers"
        )
        
        return bytes(payload_bytes)

    def _read_single_ring(self, ring_offset: int, max_bytes: int) -> bytes:
        """Read from a single ring within the double ring."""
        payload_bytes = bytearray()
        bytes_read = 0
        chunks_read = 0
        
        while bytes_read < max_bytes:
            # Read ring header (head, tail, done)
            self.pool_mmap.seek(ring_offset)
            head, tail, done = struct.unpack(
                RING_STRUCT_FORMAT,
                self.pool_mmap.read(RING_HEADER_SIZE)
            )
            
            # Check if data available
            if tail == head:
                if done and bytes_read >= max_bytes:
                    break
                # No data yet, wait
                time.sleep(0.001)  # 1ms
                continue
            
            # Read chunk at tail position
            chunk_offset = ring_offset + RING_HEADER_SIZE + tail * CHUNK_SIZE
            self.pool_mmap.seek(chunk_offset)
            chunk_data = self.pool_mmap.read(CHUNK_SIZE)
            
            # Calculate how much of this chunk is actual data
            remaining = max_bytes - bytes_read
            chunk_len = min(CHUNK_SIZE, remaining)
            
            # Append to payload
            payload_bytes.extend(chunk_data[:chunk_len])
            bytes_read += chunk_len
            chunks_read += 1
            
            # Advance tail
            new_tail = (tail + 1) % NUM_CHUNKS
            self.pool_mmap.seek(ring_offset + 4)  # offset of tail field
            self.pool_mmap.write(struct.pack("I", new_tail))
            
            if chunks_read % 10 == 0:
                logger.debug(
                    f"[WORKER:{threading.get_ident()}] "
                    f"Read {bytes_read}/{max_bytes} bytes ({chunks_read} chunks)"
                )
        
        return bytes(payload_bytes)

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
        """Cleanup and shutdown."""
        logger.info("[SERVER] Shutting down")
        
        # Close pool
        if self.pool_mmap:
            self.pool_mmap.close()
        if self.pool_fd:
            os.close(self.pool_fd)
        
        # Unlink pool
        try:
            import posix_ipc
            posix_ipc.unlink_shared_memory(self.pool_name)
            logger.info(f"[SERVER] Unlinked pool: {self.pool_name}")
        except:
            pass
        
        # Close socket
        if os.path.exists(self.sock_path):
            os.remove(self.sock_path)
        
        self.thread_pool.shutdown(wait=True)
        
        if self.server_socket:
            self.server_socket.close()
        
        print("[SERVER] Shutdown complete.")