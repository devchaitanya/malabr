#include "extensions/browser/api/malabr/malabr_api.h"

#include <string>
#include <utility>

#include "base/functional/bind.h"
#include "base/task/thread_pool.h"
#include "base/unguessable_token.h"
#include "base/values.h"
#include "components/sessions/content/session_tab_helper.h"
#include "content/public/browser/browser_task_traits.h"
#include "content/public/browser/browser_thread.h"
#include "content/public/browser/render_frame_host.h"
#include "content/public/browser/render_process_host.h"
#include "content/public/browser/visibility.h"
#include "content/public/browser/web_contents.h"
#include "url/origin.h"
#include "extensions/browser/event_router.h"
#include "extensions/common/api/malabr.h"
#include "third_party/blink/public/common/service_worker/service_worker_status_code.h"

namespace extensions {

namespace {

constexpr char kMalabrGenerateRoute[] = "ROUTE_MALABR_GENERATE_API";
constexpr char kMalabrStopRoute[] = "ROUTE_MALABR_STOP";

constexpr char kOnTokenEvent[] = "malabr.onToken";
constexpr char kOnCompleteEvent[] = "malabr.onComplete";

}  // namespace

MalabrGenerateFunction::MalabrGenerateFunction() = default;

MalabrGenerateFunction::~MalabrGenerateFunction() = default;

ExtensionFunction::ResponseAction MalabrGenerateFunction::Run() {
  EXTENSION_FUNCTION_VALIDATE(has_args());
  namespace generate_api = extensions::api::malabr::Generate;
  auto params = generate_api::Params::Create(args());
  EXTENSION_FUNCTION_VALIDATE(params);

  // ---- Identity: ALL derived here, nothing from the payload (section 5) ----
  std::string ext_id = extension_id();

  content::RenderFrameHost* rfh = render_frame_host();
  if (!rfh) {
    return RespondNow(Error("no sender frame"));
  }
  content::WebContents* web_contents = GetSenderWebContents();
  if (!web_contents) {
    return RespondNow(Error("no sender WebContents"));
  }

  // Content scripts always run inside a real tab's frame, so this yields a
  // real id. -1 is legitimate ("not tab-scoped", e.g. a future service-worker
  // caller) and is treated as always-background, not as an error.
  int tab_id = sessions::SessionTabHelper::IdForTab(web_contents).id();

  // ORIGIN -- part of the session key, and a privacy boundary, not a label.
  //
  // Without it, a session keyed only by tab survives a cross-origin
  // navigation: chat privately on a bank site, navigate that SAME tab to any
  // other site, and the new page's content script inherits a session whose KV
  // cache still holds the bank conversation. The model would answer questions
  // about it. Browser-derived via GetLastCommittedOrigin() so a content script
  // cannot claim an origin it does not have. See section 5g.
  const url::Origin& frame_origin = rfh->GetLastCommittedOrigin();

  // Refuse opaque origins outright. Serialize() renders EVERY opaque origin
  // as the literal "null" (RFC 6454), so two unrelated opaque-origin
  // documents -- sandboxed pages, data: URLs -- would collide on the key
  // (ext, tab, "null") and SHARE a session. That is the very cross-origin
  // leak section 5g exists to prevent, reintroduced through a serialization
  // collision. The distinguishing nonce is not exposed, so there is no safe
  // key to build; chatting on a data: URL is not a real use case, so reject
  // rather than invent one (audit hole 13).
  if (frame_origin.opaque()) {
    return RespondNow(Error("malabr is unavailable on opaque origins"));
  }
  std::string origin = frame_origin.Serialize();

  // SEED value only. This is visibility alone, which is NOT the same as
  // foreground: two windows on two monitors can both report VISIBLE. The
  // authoritative "visible AND its window is the active one" test needs
  // BrowserList::GetLastActive(), which lives in chrome/browser/ui and is
  // NOT reachable from extensions/browser. MalabrManager owns that check and
  // pushes corrections over the control connection; per section 6's ordering
  // rule this header value only seeds a session at creation and is ignored
  // for a session that already exists. See sections 5a, 5e, 6.
  // WebContents::GetVisibility(), not RenderFrameHost::GetVisibilityState().
  // The latter returns blink::mojom::PageVisibilityState, whose full
  // definition is NOT available here -- render_frame_host.h includes only the
  // -forward.h declaration -- so using it needs a blink mojom dependency this
  // directory should not take on. WebContents::GetVisibility() returns
  // content::Visibility, is already reachable, and is exactly the signal
  // section 5a's resolved rule names.
  //
  // Reuses the web_contents already fetched and null-checked above for the
  // tab id; re-fetching it here would shadow that one for no gain.
  bool foreground =
      web_contents->GetVisibility() == content::Visibility::VISIBLE;

  // Weak/safe handles, not raw pointers: the tab may die mid-stream (6a).
  render_frame_host_ = rfh->GetWeakDocumentPtr();

  request_id_ = base::UnguessableToken::Create().ToString();

  // Keep this object alive across the streaming phase. Run() responds
  // immediately (below), which would otherwise allow destruction while frames
  // are still arriving. Released in OnComplete().
  AddRef();
  base::ThreadPool::PostTask(
      FROM_HERE, {base::MayBlock()},
      base::BindOnce(&MalabrGenerateFunction::DispatchRequest,
                     base::Unretained(this), std::move(params->request.prompt),
                     std::move(ext_id), tab_id, std::move(origin),
                     foreground));

  // Resolve NOW with the request id -- not when generation finishes. Tokens
  // follow as events (section 6).
  return RespondNow(WithArguments(request_id_));
}

void MalabrGenerateFunction::DispatchRequest(std::string prompt,
                                             std::string extension_id,
                                             int tab_id,
                                             std::string origin,
                                             bool foreground) {
  auto ml_server = std::make_unique<extensions::MServerUDS>(
      GetMalabrSocketPath(), kMalabrGenerateRoute, extension_id);

  // Each streamed frame hops to the UI thread: events must be dispatched
  // there, and this thread must never touch a RenderFrameHost.
  auto on_token = base::BindRepeating(
      [](base::WeakPtr<MalabrGenerateFunction> self, const std::string& text) {
        content::GetUIThreadTaskRunner({})->PostTask(
            FROM_HERE,
            base::BindOnce(
                &MalabrGenerateFunction::OnToken,
                self,
                text));
      },
      weak_ptr_factory_.GetWeakPtr()
    );

  // Polled by SendStreaming once per frame, on this thread.
  auto is_abandoned = base::BindRepeating(
      [](std::shared_ptr<std::atomic<bool>> flag) { return flag->load(); },
      abandoned_);

  std::string error_msg;
  ml_server->SendStreaming(prompt, tab_id, origin, foreground, on_token,
                           is_abandoned, error_msg);

  content::GetUIThreadTaskRunner({})->PostTask(
      FROM_HERE, base::BindOnce(&MalabrGenerateFunction::OnComplete,
                                weak_ptr_factory_.GetWeakPtr(),
                                std::move(error_msg)));
}

void MalabrGenerateFunction::OnToken(std::string text) {
  DCHECK_CURRENTLY_ON(content::BrowserThread::UI);

  // The destination document is gone -- tab closed, OR the page navigated to
  // a different document (WeakDocumentPtr nulls for both). Drop this token:
  // no UAF, no crash.
  //
  // Also raise the abandon flag so the blocking thread stops reading instead
  // of draining the whole response into nowhere. Dropping FUTURE tokens one
  // by one would be safe but wasteful, and would still let the unseen
  // response land in the KV cache (sections 2, 6a, 6b).
  content::RenderFrameHost* rfh = render_frame_host_.AsRenderFrameHostIfValid();
  if (!rfh) {
    abandoned_->store(true);
    return;
  }

  // The profile can shut down mid-generation, at which point ExtensionFunction
  // nulls browser_context_ (see its OnBrowserContextShutdown). EventRouter::Get
  // would then be handed nullptr and crash, so check before dispatching.
  content::BrowserContext* context = browser_context();
  if (!context) {
    abandoned_->store(true);
    return;
  }

  base::Value::List args;
  args.Append(request_id_);
  args.Append(std::move(text));

  // Routed to the RenderProcessHost captured at request time -- NOT to a
  // renderer-supplied listener filter, which event_router.cc stores verbatim
  // and unvalidated and so must never gate delivery (section 6).
  EventRouter::Get(context)
      ->DispatchEventToSender(
          rfh->GetProcess(), context,
          mojom::HostID(mojom::HostID::HostType::kExtensions, extension_id()),
          events::MALABR_ON_TOKEN, kOnTokenEvent, kMainThreadId,
          blink::mojom::kInvalidServiceWorkerVersionId, std::move(args),
          mojom::EventFilteringInfo::New());
}

void MalabrGenerateFunction::OnComplete(std::string error) {
  DCHECK_CURRENTLY_ON(content::BrowserThread::UI);

  content::RenderFrameHost* rfh = render_frame_host_.AsRenderFrameHostIfValid();
  content::BrowserContext* context = browser_context();
  if (rfh && context) {
    base::Value::List args;
    args.Append(request_id_);
    args.Append(std::move(error));

    EventRouter::Get(context)
        ->DispatchEventToSender(
            rfh->GetProcess(), context,
            mojom::HostID(mojom::HostID::HostType::kExtensions,
                          extension_id()),
            events::MALABR_ON_COMPLETE, kOnCompleteEvent, kMainThreadId,
            blink::mojom::kInvalidServiceWorkerVersionId, std::move(args),
            mojom::EventFilteringInfo::New());
  }

  // Balances the AddRef() in Run(). Exactly one terminal frame per request,
  // so this runs exactly once.
  Release();
}


// ---------------------------------------------------------------------------
// malabr.stop()
// ---------------------------------------------------------------------------

MalabrStopFunction::MalabrStopFunction() = default;
MalabrStopFunction::~MalabrStopFunction() = default;

ExtensionFunction::ResponseAction MalabrStopFunction::Run() {
  // Same identity derivation as generate(), for the same reason: the session
  // key must be browser-derived end to end, or a page could stop someone
  // else's conversation.
  content::WebContents* web_contents = GetSenderWebContents();
  if (!web_contents) {
    return RespondNow(Error("no sender WebContents"));
  }
  content::RenderFrameHost* rfh = render_frame_host();
  if (!rfh) {
    return RespondNow(Error("no sender frame"));
  }
  const url::Origin& frame_origin = rfh->GetLastCommittedOrigin();
  if (frame_origin.opaque()) {
    return RespondNow(Error("malabr is unavailable on opaque origins"));
  }

  const int tab_id = sessions::SessionTabHelper::IdForTab(web_contents).id();

  AddRef();
  base::ThreadPool::PostTask(
      FROM_HERE, {base::MayBlock()},
      base::BindOnce(&MalabrStopFunction::DispatchStop, base::Unretained(this),
                     extension_id(), tab_id, frame_origin.Serialize()));
  return RespondLater();
}

void MalabrStopFunction::DispatchStop(std::string extension_id,
                                      int tab_id,
                                      std::string origin) {
  auto server = std::make_unique<extensions::MServerUDS>(
      GetMalabrSocketPath(), kMalabrStopRoute, extension_id);

  std::string error;
  // foreground=true is a placeholder here: the header field is a SEED for a
  // NEW session only (section 6's ordering rule) and a stop never creates one,
  // so the value cannot affect scheduling.
  const bool ok = server->SendControlRequest(tab_id, origin, true, error);

  content::GetUIThreadTaskRunner({})->PostTask(
      FROM_HERE, base::BindOnce(&MalabrStopFunction::OnStopped,
                                base::Unretained(this), ok, std::move(error)));
}

void MalabrStopFunction::OnStopped(bool ok, std::string error) {
  // A stop that found nothing to stop is NOT an error: the response may have
  // finished between the click and this call, which is an ordinary race, not
  // a failure the page should have to handle.
  Respond(ok ? NoArguments() : Error(error));
  Release();
}

}  // namespace extensions
