// MALABR Phase 1 smoke test.
//
// A CONTENT SCRIPT deliberately, not a popup or side panel: those have no real
// tab_id and no meaningful visibility, so the browser could not derive the
// identity the whole design depends on (phase1_design.md sections 2 and 5).
//
// all_frames:false so one page yields one session, not one per iframe (test 36).

(function () {
  if (window.top !== window) return;
  if (document.getElementById("malabr-root")) return;

  const host = document.createElement("div");
  host.id = "malabr-root";
  // Shadow DOM so page CSS cannot restyle the panel and page script cannot
  // read the conversation out of the DOM (section 2).
  const shadow = host.attachShadow({ mode: "closed" });
  shadow.innerHTML = `
    <style>
      :host { all: initial; }
      .wrap { position: fixed; right: 16px; bottom: 16px; z-index: 2147483647;
              font: 13px/1.45 system-ui, -apple-system, sans-serif; }

      /* Collapsed: a round launcher. */
      .pill { width: 48px; height: 48px; border-radius: 50%; border: 0;
              background: #1a73e8; color: #fff; font-size: 20px; cursor: pointer;
              box-shadow: 0 4px 14px rgba(0,0,0,.25); display: none; }
      .wrap.min .pill { display: block; }
      .wrap.min .box  { display: none; }

      /* Expanded: fixed width, grows with content to a ceiling, then scrolls. */
      .box { width: 360px; height: 460px; max-height: 70vh; display: flex;
             flex-direction: column; background: #fff; color: #111;
             border: 1px solid #c7c7c7; border-radius: 10px;
             box-shadow: 0 6px 24px rgba(0,0,0,.18); overflow: hidden;
             resize: both; min-width: 280px; min-height: 220px; }

      .hdr { padding: 8px 10px 8px 12px; font-weight: 600; display: flex;
             align-items: center; gap: 8px; border-bottom: 1px solid #eee;
             background: #fafafa; }
      .hdr .origin { font-weight: 400; font-size: 11px; color: #666;
                     margin-left: auto; overflow: hidden; text-overflow: ellipsis;
                     white-space: nowrap; max-width: 170px; }
      .icon { border: 0; background: transparent; cursor: pointer; font-size: 16px;
              line-height: 1; padding: 2px 6px; color: #555; border-radius: 4px; }
      .icon:hover { background: #ececec; }

      .log { flex: 1; overflow-y: auto; padding: 10px 12px; }
      .row { margin-bottom: 12px; }
      .who { font-size: 11px; color: #777; margin-bottom: 2px; }
      .txt { white-space: pre-wrap; word-break: break-word; }
      .err { color: #b00020; font-size: 12px; margin-top: 3px; }
      .note { color: #8a6d00; font-size: 12px; margin-top: 3px; }
      .cursor::after { content: "▍"; animation: blink 1s steps(2) infinite; }
      @keyframes blink { 50% { opacity: 0; } }

      .bar { display: flex; gap: 6px; padding: 8px; border-top: 1px solid #eee; }
      input { flex: 1; padding: 7px 9px; border: 1px solid #ccc; border-radius: 6px;
              font: inherit; min-width: 0; }
      .send { padding: 7px 12px; border: 0; border-radius: 6px; background: #1a73e8;
              color: #fff; font: inherit; cursor: pointer; white-space: nowrap; }
      .send.busy { background: #b45309; }
    </style>
    <div class="wrap">
      <button class="pill" title="Open MALABR">&#9679;</button>
      <div class="box">
        <div class="hdr">
          MALABR
          <span class="origin"></span>
          <button class="icon min" title="Minimise">&#8211;</button>
        </div>
        <div class="log"></div>
        <div class="bar">
          <input type="text" placeholder="Ask something..." />
          <button class="send">Send</button>
        </div>
      </div>
    </div>`;
  document.documentElement.appendChild(host);

  const wrap  = shadow.querySelector(".wrap");
  const log   = shadow.querySelector(".log");
  const input = shadow.querySelector("input");
  const send  = shadow.querySelector(".send");
  shadow.querySelector(".origin").textContent = location.origin;

  shadow.querySelector(".min").addEventListener("click", () => wrap.classList.add("min"));
  shadow.querySelector(".pill").addEventListener("click", () => {
    wrap.classList.remove("min");
    input.focus();
  });

  // requestId -> the element its tokens append to. A map, not a single
  // variable: a superseded turn and its replacement are briefly both live, and
  // their tokens must not land in the same bubble (section 6).
  const streams = new Map();
  let inFlight = null;

  function row(who) {
    const d = document.createElement("div");
    d.className = "row";
    d.innerHTML = `<div class="who"></div><div class="txt"></div>`;
    d.querySelector(".who").textContent = who;
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
    return d;
  }
  function note(rowEl, text, cls) {
    const n = document.createElement("div");
    n.className = cls;
    n.textContent = text;
    rowEl.appendChild(n);
    log.scrollTop = log.scrollHeight;
  }

  if (!chrome.malabr) {
    note(row("error"), "chrome.malabr is undefined -- the API is not exposed here.", "err");
    return;
  }

  function setBusy(busy) {
    // The button reflects reality instead of contradicting it. An earlier
    // version disabled it while leaving the Enter key wired to the same
    // handler, so Enter silently did what the greyed-out button said was
    // impossible.
    send.textContent = busy ? "Replace" : "Send";
    send.classList.toggle("busy", busy);
    send.title = busy
      ? "Sending now supersedes the response in progress (section 6b)"
      : "";
  }

  chrome.malabr.onToken.addListener((requestId, text) => {
    const s = streams.get(requestId);
    if (!s) return;
    s.txt.textContent += text;
    s.txt.classList.add("cursor");
    log.scrollTop = log.scrollHeight;
  });

  // Exactly ONE terminal event per request. A non-empty `error` means the turn
  // did NOT finish cleanly -- output cap, cancelled, or superseded -- and must
  // stay distinguishable from a clean finish (sections 6, 6b).
  chrome.malabr.onComplete.addListener((requestId, error) => {
    const s = streams.get(requestId);
    if (s) {
      s.txt.classList.remove("cursor");
      if (error) {
        const superseded = /supersed/i.test(error);
        note(s.row, superseded ? "[superseded by your next message]" : `[ended: ${error}]`,
             superseded ? "note" : "err");
      }
      streams.delete(requestId);
    }
    if (inFlight === requestId) {
      inFlight = null;
      setBusy(false);
    }
  });

  function ask() {
    const prompt = input.value.trim();
    if (!prompt) return;
    input.value = "";
    row("you").querySelector(".txt").textContent = prompt;
    const r = row("malabr");
    setBusy(true);

    // The caller supplies ONLY the prompt. tab_id, origin and visibility are
    // derived in the browser process precisely so a page cannot claim them.
    chrome.malabr.generate({ prompt }, (requestId) => {
      if (chrome.runtime.lastError) {
        note(r, chrome.runtime.lastError.message, "err");
        inFlight = null;
        setBusy(false);
        return;
      }
      streams.set(requestId, { row: r, txt: r.querySelector(".txt") });
      inFlight = requestId;
    });
  }

  send.addEventListener("click", ask);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });
  input.focus();
})();
