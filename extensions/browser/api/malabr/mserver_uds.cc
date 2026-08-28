#include "extensions/browser/api/malabr/mserver_uds.h"

#include <arpa/inet.h>

#include <algorithm>
#include <string>

#include "base/logging.h"
#include "base/rand_util.h"
#include "base/threading/platform_thread.h"
#include "base/time/time.h"
#include "extensions/browser/api/malabr/msocket_uds.h"
#include "extensions/common/extension_features.h"
#include "net/base/net_errors.h"

namespace extensions {

namespace {

// Frame types on the wire (phase1_design.md section 6).
constexpr uint8_t kFrameToken = 0;
constexpr uint8_t kFrameComplete = 1;
constexpr uint8_t kFrameError = 2;

// Upper bound on a single frame's payload.
//
// A frame carries one token's text, so this is enormously generous already.
// The bound exists because the length field is read straight off the wire:
// without it, `std::string(len, '\0')` allocates whatever the wire claims.
// A co-resident process with direct UDS access could otherwise ask us to
// allocate arbitrary memory (section 10's response_len bound).
constexpr uint32_t kMaxFramePayload = 1024 * 1024;  // 1 MiB

// Connect retry budget. Sized from a MEASURED cold start, not an estimate:
// with the shipped defaults and no cached calibration the server takes ~130s
// to bind its socket. A budget shorter than that turns "the server is still
// starting" into a user-visible failure on the first message after launch.
constexpr int kConnectRetryBudgetSeconds = 180;
constexpr int kConnectRetryInitialMs = 100;
constexpr int kConnectRetryMaxDelaySeconds = 3;
constexpr int kConnectRetryJitterMs = 250;

// How long to wait for ONE frame before giving up (SO_RCVTIMEO).
//
// Runtime-tunable via extensions_features::kMalabrFrameReadTimeoutSeconds
// (default 60s) rather than a constant here, because this value is an
// acknowledged placeholder until prefill latency is measured, and changing a
// constant in this tree costs a multi-hour rebuild. See the rationale on the
// FeatureParam declaration in extensions/common/extension_features.h.
//
// Per-frame, not per-response: a long generation is fine so long as tokens
// keep arriving. The longest legitimate gap is the wait for the FIRST token,
// which includes prefill -- unchunked and blocking per section 7, worst case
// still unmeasured per section 14.

// Outbound sanity bound on a prompt, in BYTES.
//
// The authoritative limit is the server's token-based MAX_INPUT_TOKENS check
// (section 8) -- this cannot replace it, because bytes are not tokens. It
// exists only so a careless or hostile extension cannot make the browser
// stream unbounded data into the socket before the server gets a chance to
// reject it. Deliberately far above any real prompt (audit hole 12).
constexpr size_t kMaxPromptBytes = 1024 * 1024;  // 1 MiB

}  // namespace

MServerUDS::MServerUDS(const std::string& socket_path,
                       const std::string& route,
                       const std::string& extension_id)
    : socket_path_(socket_path),
      route_(route),
      extension_id_(extension_id) {}

MServerUDS::~MServerUDS() = default;

bool MServerUDS::SendStreaming(const std::string& prompt,
                               int tab_id,
                               const std::string& origin,
                               bool foreground,
                               const TokenCallback& on_token,
                               const AbandonPredicate& is_abandoned,
                               std::string& error_msg) {
  if (prompt.size() > kMaxPromptBytes) {
    error_msg = "prompt too large: " + std::to_string(prompt.size()) +
                " bytes (max " + std::to_string(kMaxPromptBytes) + ")";
    return false;
  }

  base::FilePath path(socket_path_);
  MSocketUDS socket(path.value());

  // Retry with jittered backoff. Section 10 requires this and it was missing:
  // Connect() was attempted exactly once, so every generate() issued before
  // the server finished starting failed outright.
  //
  // The window matters more than section 10 assumed. It estimated a ~13s cold
  // start (model load plus calibration). Measured with the shipped defaults
  // (n_ctx=16384, n_seq_max=8, no cached calibration) the server needs ~130s
  // before it binds the socket -- roughly 10x that. Calibration dominates;
  // once its result is cached on disk, later starts are quick.
  //
  // Jitter, not a fixed interval: several tabs sending in the first seconds of
  // a cold start would otherwise retry in lockstep and synchronise into a
  // burst the moment the server becomes ready (section 10).
  int result = net::ERR_FAILED;
  base::TimeDelta delay = base::Milliseconds(kConnectRetryInitialMs);
  const base::TimeTicks deadline =
      base::TimeTicks::Now() + base::Seconds(kConnectRetryBudgetSeconds);
  while (true) {
    result = socket.Connect();
    if (result == net::OK) {
      break;
    }
    if (base::TimeTicks::Now() >= deadline) {
      error_msg = "connect failed after retrying for " +
                  std::to_string(kConnectRetryBudgetSeconds) +
                  "s: " + std::to_string(result);
      return false;
    }
    const base::TimeDelta jitter =
        base::Milliseconds(base::RandInt(0, kConnectRetryJitterMs));
    base::PlatformThread::Sleep(delay + jitter);
    delay = std::min(delay * 2, base::Seconds(kConnectRetryMaxDelaySeconds));
  }

  // Bound every recv() on this connection. Without it, a wedged server means
  // this thread pool thread never comes back (section 10).
  socket.SetReadTimeout(
      extensions_features::kMalabrFrameReadTimeoutSeconds.Get());

  // ---- header, then prompt ----
  std::string header =
      GetHeaderPayload(tab_id, origin, foreground, prompt.size());
  if (!WriteExact(socket, header.data(), header.size(), error_msg)) {
    return false;
  }
  if (!WriteExact(socket, prompt.data(), prompt.size(), error_msg)) {
    return false;
  }

  // ---- read frames until a terminal one ----
  //
  // Deliberately NOT a single read: this is the streaming half of the design.
  // Time-to-first-token is the metric that shows scheduling responsiveness at
  // all, so tokens must surface as they are produced, not batched at the end.
  while (true) {
    // The destination document may have navigated away or closed since the
    // last frame. Continuing would burn an engine slot producing tokens that
    // are dropped on arrival, AND would let the full response enter the KV
    // cache even though the user never saw it -- leaving the model's context
    // out of sync with the displayed conversation. Bail so the server sees
    // the socket drop and rolls the partial turn back (sections 6a, 6b).
    if (is_abandoned && is_abandoned.Run()) {
      error_msg = "abandoned: destination document is gone";
      return false;
    }

    uint8_t type = 0;
    std::string payload;
    if (!ReadFrame(socket, type, payload, error_msg)) {
      return false;
    }

    switch (type) {
      case kFrameToken:
        on_token.Run(payload);
        break;

      case kFrameComplete:
        return true;

      case kFrameError:
        // Server-side failure (budget hit, cancelled, superseded). Reported
        // as an error string rather than silently ending the stream, so the
        // client can tell it apart from a clean finish (sections 6, 6b).
        error_msg = payload;
        return false;

      default:
        error_msg = "unknown frame type " + std::to_string(type);
        return false;
    }
  }
}

bool MServerUDS::ReadFrame(MSocketUDS& socket,
                           uint8_t& type,
                           std::string& payload,
                           std::string& error_msg) {
  if (!ReadExact(socket, reinterpret_cast<char*>(&type), sizeof(type),
                 error_msg)) {
    return false;
  }

  uint32_t length = 0;
  if (!ReadExact(socket, reinterpret_cast<char*>(&length), sizeof(length),
                 error_msg)) {
    return false;
  }
  length = ntohl(length);

  // Bound BEFORE allocating -- see kMaxFramePayload.
  if (length > kMaxFramePayload) {
    error_msg = "frame payload too large: " + std::to_string(length);
    return false;
  }

  payload.assign(length, '\0');
  if (length > 0 &&
      !ReadExact(socket, payload.data(), length, error_msg)) {
    return false;
  }
  return true;
}

bool MServerUDS::WriteExact(MSocketUDS& socket,
                            const char* data,
                            size_t size,
                            std::string& error_msg) {
  size_t total_written = 0;
  while (total_written < size) {
    int written = socket.Write(data + total_written, size - total_written);
    if (written <= 0) {
      error_msg = "write failed at offset " + std::to_string(total_written) +
                  " with result " + std::to_string(written);
      return false;
    }
    total_written += written;
  }
  return true;
}

bool MServerUDS::ReadExact(MSocketUDS& socket,
                           char* buffer,
                           size_t size,
                           std::string& error_msg) {
  size_t total_read = 0;
  while (total_read < size) {
    int r = socket.Read(buffer + total_read, size - total_read);
    if (r <= 0) {
      error_msg = "read failed at offset " + std::to_string(total_read) +
                  " with result " + std::to_string(r);
      return false;
    }
    total_read += r;
  }
  return true;
}

bool MServerUDS::SendControlRequest(int tab_id,
                                    const std::string& origin,
                                    bool foreground,
                                    std::string& error_msg) {
  base::FilePath path(socket_path_);
  MSocketUDS socket(path.value());

  // No retry budget here, unlike Send(). A stop is only meaningful while a
  // response is already streaming, which means the server is up and a socket
  // was opened seconds ago. Retrying for three minutes would leave the UI
  // waiting long after the thing it wanted to stop had finished on its own.
  int result = socket.Connect();
  if (result != net::OK) {
    error_msg = "connect failed: " + std::to_string(result);
    return false;
  }
  socket.SetReadTimeout(
      extensions_features::kMalabrFrameReadTimeoutSeconds.Get());

  const std::string header = GetHeaderPayload(tab_id, origin, foreground, 0);
  if (!WriteExact(socket, header.data(), header.size(), error_msg)) {
    return false;
  }

  uint8_t type = 0;
  std::string payload;
  if (!ReadFrame(socket, type, payload, error_msg)) {
    return false;
  }
  if (type == kFrameError) {
    error_msg = payload;
    return false;
  }
  return true;
}

std::string MServerUDS::GetHeaderPayload(int tab_id,
                                         const std::string& origin,
                                         bool foreground,
                                         size_t payload_size) {
  // 6 fields now (was 3). tab_id, origin and visibility are browser-derived.
  //
  // origin is safe to put in a comma-separated header: a serialized origin is
  // scheme://host:port (or the literal "null" when opaque), none of which can
  // contain a comma.
  std::string header = route_ + "," + extension_id_ + "," +
                       std::to_string(tab_id) + "," + origin + "," +
                       (foreground ? "foreground" : "background") + "," +
                       std::to_string(payload_size);

  uint32_t header_len_net = htonl(static_cast<uint32_t>(header.size()));

  std::string out;
  out.reserve(sizeof(header_len_net) + header.size());
  out.append(reinterpret_cast<const char*>(&header_len_net),
             sizeof(header_len_net));
  out.append(header);
  return out;
}

}  // namespace extensions
