#include "extensions/browser/api/malabr/msocket_uds.h"

#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <memory>

#include "base/environment.h"
#include "base/logging.h"
#include "net/base/net_errors.h"

namespace extensions {

std::string GetMalabrSocketPath() {
  std::unique_ptr<base::Environment> env(base::Environment::Create());
  std::string path;
  if (env->GetVar("MALABR_SOCKET_PATH", &path) && !path.empty()) {
    return path;
  }
  return "/tmp/malabr_v3.sck";  // must match config.py's default
}

MSocketUDS::MSocketUDS(const std::string& path) : sockfd_(-1), path_(path) {}

MSocketUDS::~MSocketUDS() {
  if (sockfd_ != -1) {
    close(sockfd_);
  }
}

int MSocketUDS::Connect() {
  sockfd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sockfd_ == -1) {
    return MapErrno(errno);
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;

  // sun_path is a fixed 108 bytes on Linux and strncpy TRUNCATES silently.
  // Harmless while the path was a short hardcoded constant, but it is now
  // configurable via MALABR_SOCKET_PATH -- a long value would quietly connect
  // to a DIFFERENT, truncated path and fail with nothing pointing at the
  // cause. Reject loudly instead (audit hole 11).
  if (path_.size() >= sizeof(addr.sun_path)) {
    LOG(ERROR) << "MALABR: socket path too long (" << path_.size()
               << " bytes, max " << sizeof(addr.sun_path) - 1
               << "): " << path_;
    close(sockfd_);
    sockfd_ = -1;
    return net::ERR_FILE_PATH_TOO_LONG;
  }
  strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);

  if (connect(sockfd_, reinterpret_cast<struct sockaddr*>(&addr),
              sizeof(addr)) == -1) {
    int err = MapErrno(errno);
    close(sockfd_);
    sockfd_ = -1;
    return err;
  }

  return net::OK;
}

int MSocketUDS::SetReadTimeout(int seconds) {
  if (sockfd_ == -1) {
    return net::ERR_CONNECTION_CLOSED;
  }

  struct timeval tv;
  tv.tv_sec = seconds;
  tv.tv_usec = 0;

  if (setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == -1) {
    return MapErrno(errno);
  }
  return net::OK;
}

void MSocketUDS::Disconnect() {
  if (sockfd_ != -1) {
    close(sockfd_);
    sockfd_ = -1;
  }
}

bool MSocketUDS::IsConnected() const {
  return sockfd_ != -1;
}

int MSocketUDS::Read(char* buf, int buf_len) {
  if (sockfd_ == -1) {
    return net::ERR_CONNECTION_CLOSED;
  }

  int ret = recv(sockfd_, buf, buf_len, 0);
  if (ret >= 0) {
    return ret;
  }
  return MapErrno(errno);
}

int MSocketUDS::Write(const char* buf, int buf_len) {
  if (sockfd_ == -1) {
    return net::ERR_CONNECTION_CLOSED;
  }

  int ret = send(sockfd_, buf, buf_len, 0);
  if (ret >= 0) {
    return ret;
  }
  return MapErrno(errno);
}

int MSocketUDS::MapErrno(int e) {
  switch (e) {
    // With SO_RCVTIMEO set, a recv() that times out fails with EAGAIN
    // (== EWOULDBLOCK on Linux). This is NOT a connection reset: the peer is
    // very likely alive and merely slow. Previously both mapped to
    // ERR_CONNECTION_RESET, which made a wedged-but-alive server
    // indistinguishable from one that had genuinely gone away -- and those
    // want different handling upstream. See phase1_design.md section 10.
    case EAGAIN:
#if EWOULDBLOCK != EAGAIN
    case EWOULDBLOCK:
#endif
      return net::ERR_TIMED_OUT;

    case ECONNRESET:
      return net::ERR_CONNECTION_RESET;
    case EPIPE:
      return net::ERR_CONNECTION_CLOSED;
    case ENOTCONN:
      return net::ERR_CONNECTION_CLOSED;
    case ENOENT:
      // Socket file does not exist -- the Python server has not created it
      // yet. Distinct from a refused connection; the caller's retry-with-
      // backoff (section 10) is the right response.
      return net::ERR_FILE_NOT_FOUND;
    case ECONNREFUSED:
      return net::ERR_CONNECTION_REFUSED;
    default:
      return net::ERR_FAILED;
  }
}
}  // namespace extensions
