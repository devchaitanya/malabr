#include "extensions/browser/api/read_server_uds/ml_server_uds_v2.h"

#include <arpa/inet.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <sstream>

#include <string>

#include "extensions/browser/api/read_server_uds/socket_uds.h"
#include "net/base/net_errors.h"

namespace extensions {

// Shared memory configuration
// constexpr size_t SHM_THRESHOLD = 5 * 1024 * 1024;  // 5MB

// ── CHANGED: was NUM_CHUNKS=128 / CHUNK_SIZE=64KB for a flat ring.
//   Now: a pool of NUM_POOL_SLOTS independent buffers + a small index ring.
//   Invariant: NUM_POOL_SLOTS >= NUM_RING_SLOTS (pool can never starve ring).
constexpr uint32_t NUM_POOL_SLOTS = 32;          // independent payload buffers
constexpr uint32_t SLOT_SIZE      = 64 * 1024;   // 64KB — same as old CHUNK_SIZE
constexpr uint32_t NUM_RING_SLOTS = 16;          // index-ring capacity
static_assert(NUM_POOL_SLOTS >= NUM_RING_SLOTS,
              "Pool must be >= ring capacity to prevent deadlock");

// ── CHANGED: was one Ring struct with embedded char chunks[][].
//   Now PoolSlot holds the data; Ring holds only uint32_t slot indices.
//
// PoolSlot in_use: 0=free  1=producer-writing  2=ready-for-consumer
// Layout: in_use(1) _pad(7) data[SLOT_SIZE]   ← must match Python exactly
struct PoolSlot {
    std::atomic<uint8_t> in_use;
    char                 _pad[7];
    char                 data[SLOT_SIZE];
};
static_assert(sizeof(PoolSlot) == 8 + SLOT_SIZE, "PoolSlot size mismatch");

struct Pool {
    PoolSlot slots[NUM_POOL_SLOTS];
};

// Ring layout: head(4) tail(4) done(1) padding(7) indices[NUM_RING_SLOTS]
//              ← must match Python; indices[] must land at offset 16
struct Ring {
    std::atomic<uint32_t> head;  // producer (client) writes
    std::atomic<uint32_t> tail;  // consumer (server) writes
    std::atomic<bool>     done;  // producer sets when finished
    char                  padding[7];
    uint32_t              indices[NUM_RING_SLOTS];
};
static_assert(offsetof(Ring, indices) == 16, "Ring::indices must be at offset 16");


MLServerUDSV2::MLServerUDSV2(const std::string& socket_path,
                             const std::string& label)
    : socket_path_(socket_path), label_(label), weak_ptr_factory_(this) {}

MLServerUDSV2::~MLServerUDSV2() {
  LOG(INFO) << "MLServerUDSV2 destroyed";
}

int MLServerUDSV2::Send(const char* payload,
                        const size_t payload_size,
                        const std::string fb_file_identifier,
                        std::string& response,
                        std::string& error_msg) {
  base::FilePath path(socket_path_);
  SocketUDS socket(path.value());

  // ---- 1. Connect ----
  int result = socket.Connect();
  if (result != net::OK) {
    error_msg = "Connect failed: " + std::to_string(result);
    LOG(ERROR) << error_msg;
    return result;
  }
  LOG(INFO) << "Socket Connected";

  bool use_shm = true;

  // ---- 2. Send header ----
  std::string header_payload =
      GetHeaderPayload(payload_size, fb_file_identifier, use_shm);
  if (!WriteExact(socket, header_payload.data(), header_payload.size(),
                  error_msg)) {
    return -1;
  }
  LOG(INFO) << "Header write done! " << header_payload.size() 
            << " (use_shm=" << use_shm << ")";

  // ---- 3. Send payload ----
  if(use_shm) {
    if (!SendViaSharedMemory(socket, payload, payload_size, error_msg)) {
      return -1;
    }
  } else{
    if (!WriteExact(socket, payload, payload_size, error_msg)) {
      return -1;
    }
  }
  LOG(INFO) << "Payload write done! " << payload_size;

  // ---- 4. Read response length prefix (4 bytes) ----
  uint32_t response_len = 0;
  if (!ReadExact(socket, reinterpret_cast<char*>(&response_len),
                 sizeof(response_len), error_msg)) {
    return -1;
  }
  response_len = ntohl(response_len);
  LOG(INFO) << "Expecting response of length " << response_len;

  // ---- 5. Read response body ----
  std::string response_accum(response_len, '\0');
  if (!ReadExact(socket, response_accum.data(), response_len, error_msg)) {
    return -1;
  }

  response = std::move(response_accum);
  LOG(INFO) << "Response Read done! " << response;

  return static_cast<int>(response.size());
}

// ── ADDED: scan all pool slots; CAS the first free one (0->1) and return its index.
uint32_t AcquirePoolSlot(Pool* pool) {
  while (true) {
    for (uint32_t i = 0; i < NUM_POOL_SLOTS; ++i) {
      uint8_t expected = 0;
      if (pool->slots[i].in_use.compare_exchange_weak(
              expected, 1,
              std::memory_order_acquire,
              std::memory_order_relaxed)) {
        return i;
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

bool MLServerUDSV2::SendViaSharedMemory(SocketUDS& socket,
                                        const char* payload,
                                        size_t payload_size,
                                        std::string& error_msg) {
  // Generate unique shared memory name
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  std::ostringstream oss;
  oss << "/shm_ring_" << getpid() << "_" << pthread_self() << "_" << millis;
  std::string shm_name = oss.str();

  // ── CHANGED: SHM_SIZE = Pool + Ring (was just sizeof(Ring)).
  const size_t SHM_SIZE = sizeof(Pool) + sizeof(Ring);

  // Create shared memory
  int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
  if (shm_fd == -1) {
    error_msg = "shm_open failed: " + std::string(strerror(errno));
    LOG(ERROR) << error_msg;
    return false;
  }

  // Set size
  if (ftruncate(shm_fd, SHM_SIZE) == -1) {
    error_msg = "ftruncate failed: " + std::string(strerror(errno));
    LOG(ERROR) << error_msg;
    close(shm_fd);
    shm_unlink(shm_name.c_str());
    return false;
  }

  // Map shared memory
  void* map = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, 
                   MAP_SHARED, shm_fd, 0);
  if (map == MAP_FAILED) {
    error_msg = "mmap failed: " + std::string(strerror(errno));
    LOG(ERROR) << error_msg;
    close(shm_fd);
    shm_unlink(shm_name.c_str());
    return false;
  }

  // ── CHANGED: placement-new Pool at offset 0, Ring immediately after.
  Pool* pool = new (map) Pool();
  Ring* ring = new (static_cast<char*>(map) + sizeof(Pool)) Ring();

  for (uint32_t i = 0; i < NUM_POOL_SLOTS; ++i)
    pool->slots[i].in_use.store(0, std::memory_order_relaxed);
  ring->head.store(0, std::memory_order_relaxed);
  ring->tail.store(0, std::memory_order_relaxed);
  ring->done.store(false, std::memory_order_relaxed);

  LOG(INFO) << "Shared memory created: " << shm_name;

  // Send shared memory FD to server via SCM_RIGHTS
  if (!SendFileDescriptor(socket, shm_fd, error_msg)) {
    munmap(map, SHM_SIZE);
    close(shm_fd);
    shm_unlink(shm_name.c_str());
    return false;
  }

  LOG(INFO) << "Shared memory FD sent to server";

  // ── CHANGED: was memcpy directly into ring->chunks[head].
  //   Now: acquire pool slot -> write -> mark ready (1->2) -> push index to ring.
  size_t offset = 0;
  while (offset < payload_size) {
    // Step 1: acquire a free pool slot (0 -> 1).
    uint32_t slot_idx = AcquirePoolSlot(pool);

    // Step 2: copy payload chunk into the slot.
    size_t remaining = payload_size - offset;
    size_t chunk_len = (remaining > SLOT_SIZE) ? SLOT_SIZE : remaining;
    memcpy(pool->slots[slot_idx].data, payload + offset, chunk_len);
    if (chunk_len < SLOT_SIZE)
      memset(pool->slots[slot_idx].data + chunk_len, 0, SLOT_SIZE - chunk_len);
    offset += chunk_len;

    // Step 3: mark slot ready for consumer (1 -> 2).
    pool->slots[slot_idx].in_use.store(2, std::memory_order_release);

    // Step 4: push slot index into ring; spin if ring full.
    while (true) {
      uint32_t head = ring->head.load(std::memory_order_relaxed);
      uint32_t tail = ring->tail.load(std::memory_order_acquire);
      uint32_t next = (head + 1) % NUM_RING_SLOTS;
      if (next == tail) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      ring->indices[head] = slot_idx;
      ring->head.store(next, std::memory_order_release);
      break;
    }
  }

  // Signal completion
  ring->done.store(true, std::memory_order_release);
  
  LOG(INFO) << "Payload written to shared memory (" << payload_size 
            << " bytes in " << ((payload_size + SLOT_SIZE - 1) / SLOT_SIZE) 
            << " chunks)";

  // ── CHANGED: wait for ring drained (head==tail) AND all pool slots free.
  //   The extra pool check prevents cleanup while consumer still holds a slot.
  auto start = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(30);
  
  while (true) {
    uint32_t head = ring->head.load(std::memory_order_acquire);
    uint32_t tail = ring->tail.load(std::memory_order_acquire);
    if (head == tail) {
      bool pool_clear = true;
      for (uint32_t i = 0; i < NUM_POOL_SLOTS; ++i) {
        if (pool->slots[i].in_use.load(std::memory_order_acquire) != 0) {
          pool_clear = false;
          break;
        }
      }
      if (pool_clear) break;  // Server consumed everything
    }
    
    if (std::chrono::steady_clock::now() - start > timeout) {
      error_msg = "Timeout waiting for server to consume shared memory";
      LOG(ERROR) << error_msg;
      munmap(map, SHM_SIZE);
      close(shm_fd);
      shm_unlink(shm_name.c_str());
      return false;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  LOG(INFO) << "Server consumed all data from shared memory";

  // Cleanup
  munmap(map, SHM_SIZE);
  close(shm_fd);
  shm_unlink(shm_name.c_str());

  return true;
}

bool MLServerUDSV2::SendFileDescriptor(SocketUDS& socket,
                                       int fd,
                                       std::string& error_msg) {
  struct msghdr msg = {};
  struct iovec iov = {};
  char dummy = 0;
  
  iov.iov_base = &dummy;
  iov.iov_len = 1;
  
  char ctrl_buf[CMSG_SPACE(sizeof(int))];
  memset(ctrl_buf, 0, sizeof(ctrl_buf));
  
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl_buf;
  msg.msg_controllen = sizeof(ctrl_buf);
  
  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));
  
  int sock_fd = socket.GetRawFd();  // YOU NEED TO IMPLEMENT THIS
  
  if (sendmsg(sock_fd, &msg, 0) == -1) {
    error_msg = "sendmsg failed: " + std::string(strerror(errno));
    LOG(ERROR) << error_msg;
    return false;
  }
  
  return true;
}

bool MLServerUDSV2::WriteExact(SocketUDS& socket,
                               const char* data,
                               size_t size,
                               std::string& error_msg) {
  size_t total_written = 0;
  while (total_written < size) {
    int written = socket.Write(data + total_written, size - total_written);
    if (written <= 0) {
      error_msg = "Write failed at offset " + std::to_string(total_written) +
                  " with result " + std::to_string(written);
      LOG(ERROR) << error_msg;
      return false;
    }
    total_written += written;
  }
  return true;
}

bool MLServerUDSV2::ReadExact(SocketUDS& socket,
                              char* buffer,
                              size_t size,
                              std::string& error_msg) {
  size_t total_read = 0;
  while (total_read < size) {
    int r = socket.Read(buffer + total_read, size - total_read);
    if (r <= 0) {
      error_msg = "Read failed at offset " + std::to_string(total_read) +
                  " with result " + std::to_string(r);
      LOG(ERROR) << error_msg;
      return false;
    }
    total_read += r;
  }
  return true;
}

std::string MLServerUDSV2::GetHeaderPayload(size_t payload_size,
                                            std::string fb_file_identifier,
                                            bool use_shm) {
  // Modified header format: "fb_id,label,payload_size,use_shm"
  std::string header =
      fb_file_identifier + "," + label_ + "," + std::to_string(payload_size) +
      "," + (use_shm ? "1" : "0");

  uint32_t header_len = static_cast<uint32_t>(header.size());
  uint32_t header_len_net = htonl(header_len);

  std::string out;
  out.reserve(sizeof(header_len_net) + header.size());
  out.append(reinterpret_cast<const char*>(&header_len_net),
             sizeof(header_len_net));
  out.append(header);

  return out;
}

}  // namespace extensions