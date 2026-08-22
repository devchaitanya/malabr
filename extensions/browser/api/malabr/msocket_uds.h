#ifndef MSOCKET_UDS_H_
#define MSOCKET_UDS_H_

#include <string>

#include "net/base/io_buffer.h"

namespace extensions {

// The one place the UDS path is decided, for every caller in the browser.
//
// It was previously hardcoded in TWO C++ files while the Python server read
// MALABR_SOCKET_PATH from the environment (config.py). Setting that variable
// therefore made the server bind one path while the browser dialled another
// -- a silent, total failure with no error pointing at the cause. Same
// variable, same default, resolved once (audit hole 10).
std::string GetMalabrSocketPath();

class MSocketUDS {
 public:
  explicit MSocketUDS(const std::string& path);
  ~MSocketUDS();

  // Establish connection to UDS server.
  int Connect();

  // Disconnect and close fd.
  void Disconnect();

  // Whether fd is valid.
  bool IsConnected() const;

  // Bound how long a single Read() may block (SO_RCVTIMEO).
  //
  // Without this, a wedged server means recv() never returns and the calling
  // browser thread pool thread is leaked permanently -- it can never be
  // returned to the pool. Applies per-recv(), not to a whole response, so a
  // long generation is fine as long as SOME frame arrives within the window.
  // See phase1_design.md section 10.
  int SetReadTimeout(int seconds);

  // Synchronous read. Returns bytes read (0 = peer closed), or a negative
  // net:: error. ERR_TIMED_OUT specifically means "nothing arrived in time",
  // which is NOT the same as the peer going away.
  int Read(char* buf, int buf_len);

  // Synchronous write.
  int Write(const char* buf, int buf_len);

 private:
  int sockfd_;
  std::string path_;
  // maps raw OS errors into Chromium's error codes
  int MapErrno(int e);
};
}  // namespace extensions

#endif  // MSOCKET_UDS_H_
