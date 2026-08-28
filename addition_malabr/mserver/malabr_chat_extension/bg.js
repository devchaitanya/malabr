// Minimal service worker. Its ONLY job: open chrome.storage.session to content
// scripts.
//
// chrome.storage.session defaults to accessLevel TRUSTED_CONTEXTS, which
// EXCLUDES content scripts -- so the panel's save()/restore() (which run in a
// content script) silently read and write nothing, and reload-persistence never
// works. setAccessLevel must be called from a privileged context; this is it.
//
// The access level is session-scoped state that outlives this ephemeral worker,
// but calling it on every worker start (top level) plus the lifecycle events is
// belt-and-braces against a cold start racing the first content script.

function openSessionStorage() {
  chrome.storage.session
    .setAccessLevel({ accessLevel: "TRUSTED_AND_UNTRUSTED_CONTEXTS" })
    .catch((e) => console.warn("MALABR: setAccessLevel failed", e));
}

openSessionStorage();
chrome.runtime.onInstalled.addListener(openSessionStorage);
chrome.runtime.onStartup.addListener(openSessionStorage);
