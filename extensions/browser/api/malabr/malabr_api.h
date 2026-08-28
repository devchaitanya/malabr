#ifndef EXTENSIONS_BROWSER_API_MALABR_API_H_
#define EXTENSIONS_BROWSER_API_MALABR_API_H_

#include <atomic>
#include <memory>
#include <string>

#include "base/memory/weak_ptr.h"
#include "content/public/browser/weak_document_ptr.h"
#include "extensions/browser/api/malabr/mserver_uds.h"
#include "extensions/browser/extension_function.h"

namespace extensions {

// malabr.generate() -- MALABR Phase 1 chat inference.
//
// Unlike the v3.0 fit/score/predict functions this replaces, generate() is
// STREAMED: Run() responds immediately with a request id, then tokens arrive
// as onToken events until exactly one terminal onComplete. See
// phase1_design.md section 6.
//
// The whole point of doing this in the browser process is identity: ext_id,
// tab_id and visibility are all DERIVED here, never accepted from the caller.
// A content script cannot claim to be a different tab, or claim to be
// foreground to steal scheduler priority (sections 5, 5a).
// End the response currently streaming for this tab, keeping the session.
//
// Deliberately takes no arguments. The session is addressed by the same
// browser-derived (extension, tab, origin) identity the header already
// carries, so a page cannot stop a session that is not its own -- a
// page-supplied request id would be exactly the sort of claim section 5
// refuses to trust.
class MalabrStopFunction : public ExtensionFunction {
 public:
  DECLARE_EXTENSION_FUNCTION("malabr.stop", MALABR_STOP)
  MalabrStopFunction();

 protected:
  ~MalabrStopFunction() override;

 private:
  ResponseAction Run() override;

  // Runs on a MayBlock() thread pool thread.
  void DispatchStop(std::string extension_id, int tab_id, std::string origin);
  void OnStopped(bool ok, std::string error);
};

class MalabrGenerateFunction : public ExtensionFunction {
 public:
  DECLARE_EXTENSION_FUNCTION("malabr.generate", MALABR_GENERATE)
  MalabrGenerateFunction();

 protected:
  ~MalabrGenerateFunction() override;

 private:
  ResponseAction Run() override;

  // Runs on a MayBlock() thread pool thread. Opens the UDS, writes the
  // request header + prompt, then loops reading streamed frames until a
  // terminal one arrives. Blocking on purpose -- that is what MayBlock is for.
  void DispatchRequest(std::string prompt,
                       std::string extension_id,
                       int tab_id,
                       std::string origin,
                       bool foreground);

  // Per-frame callbacks, hopped back to the UI thread. Events must be
  // dispatched from the UI thread, and the streaming thread must never touch
  // a RenderFrameHost directly.
  void OnToken(std::string text);
  void OnComplete(std::string error);

  // Identity captured once, at request time.
  //
  // Deliberately NOT a raw pointer: a tab can close, or the page can navigate
  // to a different document, while tokens are still in flight toward it --
  // destroying the frame underneath us, i.e. a use-after-free. Same pattern
  // Chrome's own AIManager uses (chrome/browser/ai/ai_manager.h). See
  // phase1_design.md 6a, Problem A.
  content::WeakDocumentPtr render_frame_host_;
  // NOTE: the RenderProcessHost is deliberately NOT cached alongside this.
  // §6a suggested a base::SafeRef, but SafeRef CHECK-fails when its target is
  // gone, so a cached one could turn a recoverable "destination went away"
  // into a browser crash. Deriving the process from the ALREADY-VALIDATED
  // RenderFrameHost at dispatch time cannot be stale by construction, and
  // needs no second lifetime guard.

  // Minted here, never accepted from the frontend. An UnguessableToken rather
  // than a counter so two requests live at the moment of a server restart
  // cannot collide on a counter that resets to 0 (phase1_design.md 6).
  std::string request_id_;

  // Set on the UI thread the moment we discover the destination document is
  // gone; read on the blocking thread once per frame.
  //
  // A shared_ptr because the two threads outlive each other unpredictably:
  // this ExtensionFunction can be destroyed while the pool thread is still
  // inside SendStreaming, so the flag must not live in the object. Atomic
  // because it is genuinely written and read across threads.
  //
  // Why it exists: WeakDocumentPtr nulls when the frame navigates to a NEW
  // DOCUMENT, not only when the tab closes (verified in
  // content/public/browser/weak_document_ptr.h). Without this, clicking a
  // link mid-generation would leave the server generating into a void -- and
  // worse, the full response would enter the KV cache despite the user never
  // seeing it, desynchronising the model's context from the displayed
  // conversation (sections 2, 6b).
  std::shared_ptr<std::atomic<bool>> abandoned_ =
      std::make_shared<std::atomic<bool>>(false);

  base::WeakPtrFactory<MalabrGenerateFunction> weak_ptr_factory_{this};
};

}  // namespace extensions

#endif  // EXTENSIONS_BROWSER_API_MALABR_API_H_
