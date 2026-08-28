#ifndef EXTENSIONS_BROWSER_API_READ_SERVER_UDS_MSERVER_UDS_H_
#define EXTENSIONS_BROWSER_API_READ_SERVER_UDS_MSERVER_UDS_H_

#include <memory>
#include <string>

#include "base/functional/callback.h"
#include "extensions/browser/api/malabr/msocket_uds.h"

namespace extensions {

class MServerUDS {
 public:
  // Invoked once per streamed token chunk, on the calling (blocking) thread.
  // The caller is responsible for hopping to the UI thread.
  using TokenCallback =
      base::RepeatingCallback<void(const std::string& text)>;

  // Polled once per frame. Returning true means the destination document is
  // gone (navigated away / closed) and there is no longer anyone to receive
  // these tokens, so the stream should be abandoned rather than read to
  // completion. Checked on the blocking thread, so it must be thread-safe --
  // in practice it reads a shared atomic set from the UI thread.
  using AbandonPredicate = base::RepeatingCallback<bool()>;

  MServerUDS(const std::string& socket_path,
             const std::string& route,
             const std::string& extension_id);
  ~MServerUDS();

  // Streamed request/response for malabr.generate().
  //
  // Blocks until a terminal frame arrives, invoking `on_token` for each token
  // chunk along the way. Returns true on clean completion; on failure returns
  // false and fills `error_msg`. A server-sent error frame (type 2) is also
  // reported through `error_msg`, so "finished" and "failed partway" stay
  // distinguishable -- see phase1_design.md section 6.
  bool SendStreaming(const std::string& prompt,
                     int tab_id,
                     const std::string& origin,
                     bool foreground,
                     const TokenCallback& on_token,
                     const AbandonPredicate& is_abandoned,
                     std::string& error_msg);

  // A header-only request: no payload, exactly one terminal frame back.
  //
  // Used by malabr.stop(). It carries the same browser-derived identity as
  // generate() because that identity IS the addressing -- the server needs no
  // request id to know which session to end.
  bool SendControlRequest(int tab_id,
                          const std::string& origin,
                          bool foreground,
                          std::string& error_msg);

 private:
  // "route,extension_id,tab_id,origin,visibility,payload_size",
  // length-prefixed. tab_id, origin and visibility are all attached HERE, by
  // the browser process -- the content script never sends them, and could not
  // be trusted to (phase1_design.md sections 5, 5g).
  std::string GetHeaderPayload(int tab_id,
                               const std::string& origin,
                               bool foreground,
                               size_t payload_size);

  // Reads exactly one [1B type][4B length][payload] frame.
  bool ReadFrame(MSocketUDS& socket,
                 uint8_t& type,
                 std::string& payload,
                 std::string& error_msg);

  bool ReadExact(MSocketUDS& socket,
                 char* buffer,
                 size_t size,
                 std::string& error_msg);
  bool WriteExact(MSocketUDS& socket,
                  const char* data,
                  size_t size,
                  std::string& error_msg);

  std::string socket_path_;
  std::string route_;
  std::string extension_id_;
};

}  // namespace extensions

#endif  // EXTENSIONS_BROWSER_API_READ_SERVER_UDS_MSERVER_UDS_H_
