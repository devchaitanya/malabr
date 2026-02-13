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
constexpr uint32_t NUM_CHUNKS = 128;
constexpr uint32_t CHUNK_SIZE = 64 * 1024;  // 64KB

// Ring buffer structure (must match Python exactly)
struct Ring {
    std::atomic<uint32_t> head;  // producer (client) writes
    std::atomic<uint32_t> tail;  // consumer (server) writes
    std::atomic<bool> done;      // producer sets when finished
    char padding[7];  // padding for alignment
    char chunks[NUM_CHUNKS][CHUNK_SIZE];
};


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

  const size_t SHM_SIZE = sizeof(Ring);

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

  // Initialize ring buffer with placement new
  Ring* ring = new (map) Ring();
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

  // Write payload to ring buffer in chunks
  size_t offset = 0;
  while (offset < payload_size) {
    uint32_t head = ring->head.load(std::memory_order_relaxed);
    uint32_t tail = ring->tail.load(std::memory_order_acquire);
    uint32_t next = (head + 1) % NUM_CHUNKS;

    // Check if ring is full
    if (next == tail) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // Calculate chunk size to write
    size_t remaining = payload_size - offset;
    size_t chunk_len = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;

    // Copy data to chunk
    memcpy(ring->chunks[head], payload + offset, chunk_len);
    
    // Zero-fill remaining space in chunk if needed
    if (chunk_len < CHUNK_SIZE) {
      memset(ring->chunks[head] + chunk_len, 0, CHUNK_SIZE - chunk_len);
    }

    offset += chunk_len;

    // Advance head
    ring->head.store(next, std::memory_order_release);
  }

  // Signal completion
  ring->done.store(true, std::memory_order_release);
  
  LOG(INFO) << "Payload written to shared memory (" << payload_size 
            << " bytes in " << ((payload_size + CHUNK_SIZE - 1) / CHUNK_SIZE) 
            << " chunks)";

  // Wait for server to consume all data (tail catches up to head)
  // This ensures server has read everything before we cleanup
  auto start = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(30);
  
  while (true) {
    uint32_t head = ring->head.load(std::memory_order_acquire);
    uint32_t tail = ring->tail.load(std::memory_order_acquire);
    
    if (head == tail) {
      break;  // Server consumed everything
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
