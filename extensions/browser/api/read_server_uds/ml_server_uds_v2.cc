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

// Configuration
constexpr uint32_t NUM_CHUNKS = 128;
constexpr uint32_t CHUNK_SIZE = 64 * 1024;  // 64KB

// Double ring buffer structure - ONLY structure we use
struct DoubleRing {
    // Ring A
    alignas(64) std::atomic<uint32_t> head_a;
    std::atomic<uint32_t> tail_a;
    std::atomic<bool> done_a;
    char padding_a[7];
    char chunks_a[NUM_CHUNKS][CHUNK_SIZE];
    
    // Ring B
    alignas(64) std::atomic<uint32_t> head_b;
    std::atomic<uint32_t> tail_b;
    std::atomic<bool> done_b;
    char padding_b[7];
    char chunks_b[NUM_CHUNKS][CHUNK_SIZE];
    
    // Control atomics
    std::atomic<uint8_t> active_write_ring;  // 0=A, 1=B
    std::atomic<uint8_t> active_read_ring;
    std::atomic<bool> swap_requested;
    std::atomic<bool> swap_acknowledged;
    char control_padding[4];
};

MLServerUDSV2::MLServerUDSV2(const std::string& socket_path,
                             const std::string& label)
    : socket_path_(socket_path), label_(label), weak_ptr_factory_(this) {}

MLServerUDSV2::~MLServerUDSV2() {
  LOG(INFO) << "MLServerUDSV2 destroyed";
}

void MLServerUDSV2::Clear() {
  // Cleanup if needed
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
    error_msg = "Connect failed: " + (path.value());
    LOG(ERROR) << error_msg;
    return result;
  }
  LOG(INFO) << "Socket Connected";

  bool use_shm = true;  // Always use shared memory (pool + double ring)

  // ---- 2. Send header ----
  std::string header_payload =
      GetHeaderPayload(payload_size, fb_file_identifier, use_shm);
  if (!WriteExact(socket, header_payload.data(), header_payload.size(),
                  error_msg)) {
    return -1;
  }
  LOG(INFO) << "Header write done! " << header_payload.size() 
            << " (use_shm=" << use_shm << ")";

  // ---- 3. Send payload via pool + double ring ----
  if (!SendViaSharedMemory(socket, payload, payload_size, error_msg)) {
    return -1;
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
  // Receive pool info from server
  uint32_t info_len;
  if (!ReadExact(socket, reinterpret_cast<char*>(&info_len), 
                 sizeof(info_len), error_msg)) {
    return false;
  }
  info_len = ntohl(info_len);
  
  std::string pool_info(info_len, '\0');
  if (!ReadExact(socket, pool_info.data(), info_len, error_msg)) {
    return false;
  }
  
  // Parse: "/shm_pool_12345,7"
  size_t comma_pos = pool_info.find(',');
  std::string pool_name = pool_info.substr(0, comma_pos);
  int slot_id = std::stoi(pool_info.substr(comma_pos + 1));
  
  LOG(INFO) << "Pool: " << pool_name << ", Slot: " << slot_id;
  
  // Open existing pool (created by server)
  int pool_fd = shm_open(pool_name.c_str(), O_RDWR, 0666);
  if (pool_fd == -1) {
    error_msg = "shm_open failed: " + std::string(strerror(errno));
    LOG(ERROR) << error_msg;
    return false;
  }
  
  // Calculate sizes
  const size_t DOUBLE_RING_SIZE = sizeof(DoubleRing);
  const size_t MAX_SLOTS = 64;
  const size_t CONTROL_SIZE = 4096;
  const size_t POOL_SIZE = CONTROL_SIZE + (MAX_SLOTS * DOUBLE_RING_SIZE);
  
  // Map pool
  void* pool_map = mmap(nullptr, POOL_SIZE, PROT_READ | PROT_WRITE,
                        MAP_SHARED, pool_fd, 0);
  if (pool_map == MAP_FAILED) {
    error_msg = "mmap failed: " + std::string(strerror(errno));
    LOG(ERROR) << error_msg;
    close(pool_fd);
    return false;
  }
  
  // Get our double ring from the pool
  size_t double_ring_offset = CONTROL_SIZE + (slot_id * DOUBLE_RING_SIZE);
  DoubleRing* dr = reinterpret_cast<DoubleRing*>(
      static_cast<char*>(pool_map) + double_ring_offset
  );
  
  // Initialize both rings
  dr->head_a.store(0, std::memory_order_relaxed);
  dr->tail_a.store(0, std::memory_order_relaxed);
  dr->done_a.store(false, std::memory_order_relaxed);
  
  dr->head_b.store(0, std::memory_order_relaxed);
  dr->tail_b.store(0, std::memory_order_relaxed);
  dr->done_b.store(false, std::memory_order_relaxed);
  
  dr->active_write_ring.store(0, std::memory_order_relaxed);
  dr->active_read_ring.store(0, std::memory_order_relaxed);
  dr->swap_requested.store(false, std::memory_order_relaxed);
  dr->swap_acknowledged.store(false, std::memory_order_relaxed);
  
  LOG(INFO) << "Double ring initialized";
  
  // Write payload using double buffering
  size_t offset = 0;
  int transfer_count = 0;
  const size_t RING_CAPACITY = NUM_CHUNKS * CHUNK_SIZE;
  
  while (offset < payload_size) {
    // Get current write ring index
    uint8_t write_idx = dr->active_write_ring.load(std::memory_order_acquire);
    
    // Get pointers to active ring's fields
    std::atomic<uint32_t>* head;
    std::atomic<uint32_t>* tail;
    std::atomic<bool>* done;
    char (*chunks)[CHUNK_SIZE];
    
    if (write_idx == 0) {
      // Use Ring A
      head = &dr->head_a;
      tail = &dr->tail_a;
      done = &dr->done_a;
      chunks = dr->chunks_a;
    } else {
      // Use Ring B
      head = &dr->head_b;
      tail = &dr->tail_b;
      done = &dr->done_b;
      chunks = dr->chunks_b;
    }
    
    // Calculate how much to write in this ring
    size_t remaining = payload_size - offset;
    size_t to_write = std::min(RING_CAPACITY, remaining);
    
    LOG(INFO) << "Transfer " << transfer_count << ": " << to_write 
              << " bytes to Ring " << (char)('A' + write_idx);
    
    // Write to current ring
    size_t ring_offset = 0;
    while (ring_offset < to_write) {
      uint32_t h = head->load(std::memory_order_relaxed);
      uint32_t t = tail->load(std::memory_order_acquire);
      uint32_t next = (h + 1) % NUM_CHUNKS;
      
      // Check if ring is full
      if (next == t) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        continue;
      }
      
      // Calculate chunk size to write
      size_t chunk_remaining = to_write - ring_offset;
      size_t chunk_len = std::min((size_t)CHUNK_SIZE, chunk_remaining);
      
      // Copy data to chunk
      memcpy(chunks[h], payload + offset + ring_offset, chunk_len);
      
      // Zero-fill remaining space in chunk if needed
      if (chunk_len < CHUNK_SIZE) {
        memset(chunks[h] + chunk_len, 0, CHUNK_SIZE - chunk_len);
      }
      
      ring_offset += chunk_len;
      
      // Advance head
      head->store(next, std::memory_order_release);
    }
    
    offset += to_write;
    transfer_count++;
    
    // Signal completion for this ring
    done->store(true, std::memory_order_release);
    
    // If more data remains, swap rings
    if (offset < payload_size) {
      LOG(INFO) << "Requesting ring swap";
      
      // Request swap
      dr->swap_requested.store(true, std::memory_order_release);
      
      // Wait for server acknowledgment
      auto start = std::chrono::steady_clock::now();
      const auto timeout = std::chrono::seconds(10);
      
      while (!dr->swap_acknowledged.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() - start > timeout) {
          error_msg = "Swap timeout";
          LOG(ERROR) << error_msg;
          munmap(pool_map, POOL_SIZE);
          close(pool_fd);
          return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
      
      // Perform swap
      uint8_t new_write_idx = 1 - write_idx;
      dr->active_write_ring.store(new_write_idx, std::memory_order_release);
      
      // Reset the ring we just finished writing
      head->store(0, std::memory_order_relaxed);
      tail->store(0, std::memory_order_relaxed);
      done->store(false, std::memory_order_relaxed);
      
      // Clear swap flags
      dr->swap_requested.store(false, std::memory_order_release);
      dr->swap_acknowledged.store(false, std::memory_order_release);
      
      LOG(INFO) << "Swapped to Ring " << (char)('A' + new_write_idx);
    }
  }
  
  LOG(INFO) << "Transfer complete: " << transfer_count << " ring fills";
  
  // Wait for server to consume all data from final ring
  uint8_t final_idx = dr->active_write_ring.load(std::memory_order_acquire);
  
  std::atomic<uint32_t>* final_head;
  std::atomic<uint32_t>* final_tail;
  
  if (final_idx == 0) {
    final_head = &dr->head_a;
    final_tail = &dr->tail_a;
  } else {
    final_head = &dr->head_b;
    final_tail = &dr->tail_b;
  }
  
  auto start = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(30);
  
  while (true) {
    uint32_t h = final_head->load(std::memory_order_acquire);
    uint32_t t = final_tail->load(std::memory_order_acquire);
    
    if (h == t) {
      break;  // Server consumed everything
    }
    
    if (std::chrono::steady_clock::now() - start > timeout) {
      error_msg = "Timeout waiting for server to consume shared memory";
      LOG(ERROR) << error_msg;
      munmap(pool_map, POOL_SIZE);
      close(pool_fd);
      return false;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  LOG(INFO) << "Server consumed all data from shared memory";
  
  // Cleanup (pool persists on server)
  munmap(pool_map, POOL_SIZE);
  close(pool_fd);
  
  return true;
}

bool MLServerUDSV2::SendFileDescriptor(SocketUDS& socket,
                                       int fd,
                                       std::string& error_msg) {
  // NOT USED - Pool already exists, no FD passing needed
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