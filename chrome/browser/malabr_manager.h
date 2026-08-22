#ifndef CHROME_BROWSER_MALABR_MANAGER_H_
#define CHROME_BROWSER_MALABR_MANAGER_H_

#include <memory>
#include <string>
#include <vector>

#include "base/feature_list.h"
#include "base/memory/ptr_util.h"
#include "base/no_destructor.h"
#include "base/time/time.h"
#include "base/process/process.h"
#include "base/threading/sequence_bound.h"
#include "base/scoped_multi_source_observation.h"
#include "chrome/browser/profiles/profile_manager_observer.h"
#include "chrome/browser/ui/browser_list_observer.h"
#include "chrome/browser/ui/tabs/tab_strip_model_observer.h"
#include "extensions/browser/extension_registry.h"
#include "extensions/browser/extension_registry_observer.h"
#include "content/public/common/content_features.h"
#include "extensions/browser/api/malabr/msocket_uds.h"

class Browser;

namespace malabr {

// Owns the ONE persistent control connection to the Python server.
//
// Lives entirely on a blocking SequencedTaskRunner, never on the UI thread:
// socket writes block, and the UI thread must never block. The sequence also
// guarantees message ORDER, which matters -- a foreground change followed by
// a tab close must not arrive reversed.
//
// Separate from the per-request sockets in mserver_uds.cc, which open and
// close per generate() call. An IDLE session has no such socket, so without
// this connection a tab closing while idle would have no delivery path at
// all. See phase1_design.md sections 5e, 5f.
class ControlConnection {
 public:
  ControlConnection();
  ~ControlConnection();

  ControlConnection(const ControlConnection&) = delete;
  ControlConnection& operator=(const ControlConnection&) = delete;

  // Connects if not already connected, sending `resync` on a fresh
  // connection. Returns whether we are connected afterwards, so the owner can
  // schedule a retry -- the server is NOT up when the browser first tries
  // (it is launched moments later and takes ~3s to load the model), so
  // without a retry the control channel could stay dead for the whole
  // session (section 5f).
  bool Connect(std::vector<std::string> resync);

  // Sends one control message, connecting first if needed. On a FRESH
  // connection every message in `resync` is sent first, because a reconnected
  // server may hold state that diverged while we were disconnected, and any
  // message written into the dead socket was lost (section 5f).
  void Send(std::string message, std::vector<std::string> resync);

 private:
  bool EnsureConnected(const std::vector<std::string>& resync);
  bool WriteFrame(const std::string& message);

  // Set by EnsureConnected when it has just handshaked and replayed `resync`.
  // Lets Send() skip a message the resync already carried, instead of pushing
  // an identical FOREGROUND twice on every reconnect.
  std::vector<std::string> last_resync_sent_;

  std::unique_ptr<extensions::MSocketUDS> socket_;
};

}  // namespace malabr

// Owns the Python server process, and the browser-side observers that keep
// it informed about tab lifecycle and which tab the user is actually looking
// at.
//
// Why the observers live HERE and not in extensions/browser/api/malabr:
// section 5a's foreground test needs BrowserList::GetLastActive(), and
// BrowserList lives in chrome/browser/ui -- which extensions/browser is not
// permitted to depend on. Same layering rule that ruled out ExtensionTabUtil
// in section 5. So the extension function derives identity, and this class
// owns window/tab observation.
class MalabrManager : public BrowserListObserver,
                      public TabStripModelObserver,
                      public ProfileManagerObserver,
                      public extensions::ExtensionRegistryObserver {
 public:
  static MalabrManager& GetInstance();

  void StartMLServerIfEnabled();
  void StopMLServer();

  // BrowserListObserver -- WINDOW activation.
  // Visibility alone is not foreground: two windows on two monitors both
  // report VISIBLE for their active tab (section 5a).
  void OnBrowserAdded(Browser* browser) override;
  void OnBrowserRemoved(Browser* browser) override;
  void OnBrowserSetLastActive(Browser* browser) override;
  void OnBrowserNoLongerActive(Browser* browser) override;

  // TabStripModelObserver -- tab switches and tab closes.
  void OnTabStripModelChanged(
      TabStripModel* tab_strip_model,
      const TabStripModelChange& change,
      const TabStripSelectionChange& selection) override;

  // ProfileManagerObserver -- ExtensionRegistry is PER PROFILE, so there is
  // no single registry to observe; each profile brings its own.
  void OnProfileAdded(Profile* profile) override;

  // ExtensionRegistryObserver -- tears down EVERY session belonging to an
  // extension in one sweep. Tab close (§6a) frees one session at a time, so
  // without this an uninstalled extension's sessions would sit holding slots
  // until each of their tabs happened to close (§10).
  void OnExtensionUnloaded(content::BrowserContext* browser_context,
                           const extensions::Extension* extension,
                           extensions::UnloadedExtensionReason reason) override;
  void OnExtensionUninstalled(content::BrowserContext* browser_context,
                              const extensions::Extension* extension,
                              extensions::UninstallReason reason) override;

  // A profile going away destroys its ExtensionRegistry. Without dropping the
  // observation here, ScopedMultiSourceObservation would keep a stale pointer
  // and dangle on the next profile teardown (audit hole 8).
  void OnShutdown(extensions::ExtensionRegistry* registry) override;

 private:
  MalabrManager();
  ~MalabrManager() override;

  // Never destroyed: a function-local static would run its destructor at
  // exit, in an order undefined relative to BrowserList -- and this
  // destructor touches BrowserList. chrome/browser builds with
  // wexit_time_destructors for exactly this reason. StopMLServer() is called
  // explicitly during shutdown (chrome_browser_main.cc), so nothing is lost
  // by never destructing (audit hole 2).
  friend class base::NoDestructor<MalabrManager>;

  // Recomputes "visible AND in the active window" and pushes it if it moved.
  // One message per switch, carrying the WHOLE answer -- see the tab_id note
  // in the .cc for why this is a tab id and not a session key.
  void UpdateForegroundTab();

  void SendControlMessage(std::string message);

  // Opens the control connection, retrying with jittered backoff until it
  // succeeds. Must be kicked off explicitly: the connection is otherwise only
  // attempted when a control EVENT occurs, so a user who never switches or
  // closes a tab would leave it permanently unconnected (audit hole 1).
  void EnsureControlConnected();
  void OnControlConnectResult(bool connected);

  // State-based resync payload, rebuilt on every send so a reconnect always
  // carries current truth:
  //   FOREGROUND,<tab_id>   -- which tab is foreground right now
  //   LIVE_TABS,<id>,<id>.. -- every tab that currently exists
  //
  // LIVE_TABS exists because TAB_CLOSED is an EVENT, and an event written
  // into a dead socket is lost forever -- after reconnect the browser has no
  // memory that the tab ever existed, so there is nothing to replay. Sending
  // the set of live tabs instead lets the server RECONCILE: any session whose
  // tab is absent from that set is dead and its slot must be freed. Same
  // principle as foreground_tab_id -- push state, not events, because state
  // is idempotent and self-correcting while events are neither.
  std::vector<std::string> BuildResyncMessages() const;

  base::Process mserver_uds_process_;

  // Last value pushed, so we do not spam identical messages.
  int foreground_tab_id_ = -1;

  base::SequenceBound<malabr::ControlConnection> control_;

  // Backoff state for the connect retry. Jittered per section 10 so several
  // clients cannot synchronise into a burst the moment the server is ready.
  base::TimeDelta control_retry_delay_ = base::Milliseconds(500);
  bool control_connect_pending_ = false;

  // LIVE_TABS changes only when a tab is opened or closed -- NOT when the
  // user merely switches tabs, which is the frequent event. Cached and
  // rebuilt on demand so the common path does not walk every tab in every
  // window on every foreground change (audit hole 5).
  mutable std::string cached_live_tabs_;
  mutable bool live_tabs_dirty_ = true;

  base::ScopedMultiSourceObservation<extensions::ExtensionRegistry,
                                     extensions::ExtensionRegistryObserver>
      registry_observations_{this};

  MalabrManager(const MalabrManager&) = delete;
  MalabrManager& operator=(const MalabrManager&) = delete;
};

#endif  // CHROME_BROWSER_MALABR_MANAGER_H_
