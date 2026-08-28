// MALABR Phase 1 chat panel.
//
// A CONTENT SCRIPT deliberately, not a popup or side panel: those have no real
// tab_id and no meaningful visibility, so the browser could not derive the
// identity the design depends on (phase1_design.md sections 2 and 5).
// all_frames:false so one page is one session, not one per iframe (test 36).

(function () {
  if (window.top !== window) return;
  if (document.getElementById("malabr-root")) return;

  // Section 8's input cap is budget - RESERVED_FOR_RESPONSE, i.e. ~1792 tokens
  // for a 2048-token session share. Page text is truncated well inside that:
  // an oversized prompt is REJECTED outright rather than silently trimmed
  // server-side, and compaction cannot help a single prompt that exceeds the
  // whole budget.
  const MAX_PAGE_CHARS = 6000;

  const host = document.createElement("div");
  host.id = "malabr-root";
  const shadow = host.attachShadow({ mode: "closed" });
  shadow.innerHTML = `
    <style>
      :host { all: initial; }
      * { box-sizing: border-box; }
      .wrap { position: fixed; right: 20px; bottom: 20px; z-index: 2147483647;
              font: 14px/1.6 ui-sans-serif, -apple-system, "Segoe UI", sans-serif; }

      .pill { width: 46px; height: 46px; border-radius: 50%; border: 0; display: none;
              background: #c96442; color: #fff; font-size: 19px; cursor: pointer;
              box-shadow: 0 6px 20px rgba(0,0,0,.45); }
      .wrap.min .pill { display: block; }
      .wrap.min .box  { display: none; }

      .box { width: 400px; height: 560px; max-height: 78vh; display: flex;
             flex-direction: column; overflow: hidden; resize: both;
             min-width: 320px; min-height: 260px;
             background: #262624; color: #f5f4ef;
             border: 1px solid #3d3d3a; border-radius: 14px;
             box-shadow: 0 16px 48px rgba(0,0,0,.5); }

      .hdr { display: flex; align-items: center; gap: 8px; padding: 11px 12px;
             border-bottom: 1px solid #3d3d3a; background: #1f1e1d; }
      .dot { width: 8px; height: 8px; border-radius: 50%; background: #c96442; }
      .title { font-weight: 600; font-size: 13px; letter-spacing: .2px; }
      .origin { margin-left: auto; font-size: 11px; color: #8a8880;
                max-width: 165px; overflow: hidden; text-overflow: ellipsis;
                white-space: nowrap; }
      .icon { border: 0; background: transparent; color: #a3a096; cursor: pointer;
              font-size: 15px; line-height: 1; padding: 4px 7px; border-radius: 6px; }
      .icon:hover { background: #34332f; color: #f5f4ef; }

      .log { flex: 1; overflow-y: auto; padding: 16px 14px; scroll-behavior: smooth; }
      .log::-webkit-scrollbar { width: 9px; }
      .log::-webkit-scrollbar-thumb { background: #46453f; border-radius: 5px; }

      .turn { margin-bottom: 18px; }
      .turn.you { display: flex; justify-content: flex-end; }
      .bubble { max-width: 86%; padding: 9px 13px; border-radius: 14px;
                background: #37362f; white-space: pre-wrap; word-wrap: break-word; }
      .turn.ai .body { white-space: pre-wrap; word-wrap: break-word; }
      .who { font-size: 11px; color: #8a8880; margin-bottom: 5px;
             text-transform: uppercase; letter-spacing: .6px; }

      /* Reasoning is collapsed by default. Suppressed at the prompt where the
         model's template allows it, so this is a fallback for models that
         cannot be told to stop thinking. */
      details.think { margin: 0 0 8px; border-left: 2px solid #4a4942;
                      padding-left: 9px; }
      details.think > summary { cursor: pointer; color: #8a8880; font-size: 12px;
                                list-style: none; user-select: none; }
      details.think > summary::-webkit-details-marker { display: none; }
      details.think > summary::before { content: "▸ "; }
      details.think[open] > summary::before { content: "▾ "; }
      details.think .inner { color: #a3a096; font-size: 12.5px; padding-top: 6px;
                             white-space: pre-wrap; }

      .meta { font-size: 11.5px; margin-top: 5px; }
      .meta.err  { color: #e0806a; }
      .meta.note { color: #8a8880; }
      .chip { display: inline-block; font-size: 10.5px; color: #a3a096;
              background: #34332f; border-radius: 5px; padding: 1px 6px;
              margin-bottom: 5px; }
      .caret::after { content: "▍"; animation: blink 1.1s steps(2) infinite; }
      @keyframes blink { 50% { opacity: 0 } }

      .composer { border-top: 1px solid #3d3d3a; padding: 9px; background: #1f1e1d; }
      .tools { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
      .toggle { font-size: 11.5px; color: #a3a096; cursor: pointer; user-select: none;
                display: flex; align-items: center; gap: 5px; }
      .toggle input { accent-color: #c96442; margin: 0; }
      .row { display: flex; gap: 7px; align-items: flex-end; }
      textarea { flex: 1; resize: none; min-height: 40px; max-height: 130px;
                 padding: 9px 11px; border-radius: 10px; font: inherit;
                 background: #262624; color: #f5f4ef; border: 1px solid #46453f; }
      textarea:focus { outline: none; border-color: #c96442; }
      .act { width: 38px; height: 38px; flex: none; border: 0; border-radius: 9px;
             background: #c96442; color: #fff; cursor: pointer; font-size: 15px;
             display: flex; align-items: center; justify-content: center; }
      .act.stop { background: #6b6a63; }
      .act:disabled { opacity: .45; cursor: default; }
    </style>
    <div class="wrap">
      <button class="pill" title="Open MALABR">&#9679;</button>
      <div class="box">
        <div class="hdr">
          <span class="dot"></span><span class="title">MALABR</span>
          <span class="origin"></span>
          <button class="icon clear" title="New chat">&#10227;</button>
          <button class="icon min" title="Minimise">&#8722;</button>
        </div>
        <div class="log"></div>
        <div class="composer">
          <div class="tools">
            <label class="toggle"><input type="checkbox" class="usepage" checked> include page text</label>
            <label class="toggle"><input type="checkbox" class="showthink"> show reasoning</label>
          </div>
          <div class="row">
            <textarea rows="1" placeholder="Ask about this page..."></textarea>
            <button class="act" title="Send">&#10148;</button>
          </div>
        </div>
      </div>
    </div>`;
  document.documentElement.appendChild(host);

  const $ = (s) => shadow.querySelector(s);
  const wrap = $(".wrap"), log = $(".log"), ta = $("textarea"), act = $(".act");
  const usePage = $(".usepage"), showThink = $(".showthink");
  $(".origin").textContent = location.host;

  $(".min").addEventListener("click", () => wrap.classList.add("min"));
  $(".pill").addEventListener("click", () => { wrap.classList.remove("min"); ta.focus(); });

  // ---- transcript persistence (section 2) --------------------------------
  // chrome.storage.session, NOT storage.local: session storage is memory-backed
  // and dies with the browser, which is what section 4's "nothing durable"
  // rule requires. storage.local would write conversations to disk.
  const KEY = "malabr:" + location.origin;
  let transcript = [];

  function save() {
    try { chrome.storage?.session?.set({ [KEY]: transcript.slice(-40) }); } catch (e) {}
  }
  function restore() {
    try {
      chrome.storage?.session?.get(KEY, (o) => {
        const saved = o && o[KEY];
        if (!Array.isArray(saved) || !saved.length) return;
        // The SERVER still holds this conversation after a reload: the session
        // key is (extension, tab, origin) and none of those change. Without
        // restoring the display the model would remember what the user cannot
        // see -- the desync section 2 exists to prevent.
        saved.forEach((m) => render(m.who, m.text, m.think, m.meta));
        transcript = saved;
        note(log.lastElementChild || log, "restored after reload", "note");
      });
    } catch (e) {}
  }

  // ---- rendering ----------------------------------------------------------
  function render(who, text, think, meta) {
    const t = document.createElement("div");
    t.className = "turn " + (who === "you" ? "you" : "ai");
    if (who === "you") {
      const b = document.createElement("div");
      b.className = "bubble"; b.textContent = text;
      t.appendChild(b);
    } else {
      const w = document.createElement("div");
      w.className = "who"; w.textContent = "malabr";
      t.appendChild(w);
      if (think) t.appendChild(thinkBlock(think));
      const body = document.createElement("div");
      body.className = "body"; body.textContent = text || "";
      t.appendChild(body);
    }
    if (meta) {
      const m = document.createElement("div");
      m.className = "meta " + (meta.cls || "note"); m.textContent = meta.text;
      t.appendChild(m);
    }
    log.appendChild(t);
    log.scrollTop = log.scrollHeight;
    return t;
  }
  function thinkBlock(text) {
    const d = document.createElement("details");
    d.className = "think";
    d.open = showThink.checked;
    const s = document.createElement("summary");
    s.textContent = "reasoning";
    const i = document.createElement("div");
    i.className = "inner"; i.textContent = text;
    d.appendChild(s); d.appendChild(i);
    return d;
  }
  function note(el, text, cls) {
    const m = document.createElement("div");
    m.className = "meta " + cls; m.textContent = text;
    (el.classList?.contains("turn") ? el : log).appendChild(m);
    log.scrollTop = log.scrollHeight;
  }

  if (!chrome.malabr) {
    render("malabr", "", null, { text: "chrome.malabr is undefined here.", cls: "err" });
    return;
  }

  // ---- page text ----------------------------------------------------------
  function pageText() {
    const drop = "script,style,noscript,svg,canvas,iframe,nav,footer,header,form";
    const root = document.querySelector("main,article,[role=main]") || document.body;
    const c = root.cloneNode(true);
    c.querySelectorAll(drop).forEach((n) => n.remove());
    const t = (c.innerText || "").replace(/\n{3,}/g, "\n\n").replace(/[ \t]{2,}/g, " ").trim();
    return t.length > MAX_PAGE_CHARS ? t.slice(0, MAX_PAGE_CHARS) + "\n[...truncated]" : t;
  }

  // ---- streaming ----------------------------------------------------------
  const live = new Map();          // requestId -> render state
  let inFlight = null;

  function splitThink(raw) {
    // Fallback only. Reasoning is suppressed at the prompt where the model's
    // template supports it, so this handles models that cannot be told to stop.
    let think = "", body = raw;
    const open = raw.indexOf("<think>");
    if (open !== -1) {
      const close = raw.indexOf("</think>");
      if (close !== -1) {
        think = raw.slice(open + 7, close).trim();
        body = (raw.slice(0, open) + raw.slice(close + 8)).trim();
      } else {                      // still inside the block
        think = raw.slice(open + 7).trim();
        body = raw.slice(0, open).trim();
      }
    }
    return { think, body };
  }

  chrome.malabr.onToken.addListener((id, text) => {
    const s = live.get(id);
    if (!s) return;
    s.raw += text;
    const { think, body } = splitThink(s.raw);
    if (think) {
      if (!s.thinkEl) { s.thinkEl = thinkBlock(""); s.turn.insertBefore(s.thinkEl, s.bodyEl); }
      s.thinkEl.querySelector(".inner").textContent = think;
    }
    s.bodyEl.textContent = body;
    s.bodyEl.classList.add("caret");
    log.scrollTop = log.scrollHeight;
  });

  chrome.malabr.onComplete.addListener((id, error) => {
    const s = live.get(id);
    if (s) {
      s.bodyEl.classList.remove("caret");
      const { think, body } = splitThink(s.raw);
      if (error) {
        const stopped = /stop/i.test(error), sup = /supersed/i.test(error);
        note(s.turn, stopped ? "stopped" : sup ? "replaced by your next message" : error,
             stopped || sup ? "note" : "err");
      }
      if (!body && !error) note(s.turn, "the model produced only reasoning", "note");
      transcript.push({ who: "malabr", text: body, think, meta: null });
      save();
      live.delete(id);
    }
    if (inFlight === id) { inFlight = null; setBusy(false); }
  });

  function setBusy(busy) {
    // A real stop needs malabr.stop(), which exists only once the browser is
    // rebuilt with it. Feature-detected so the button never claims a capability
    // the binary does not have.
    const canStop = typeof chrome.malabr.stop === "function";
    act.classList.toggle("stop", busy && canStop);
    act.innerHTML = busy && canStop ? "&#9632;" : "&#10148;";
    act.title = busy ? (canStop ? "Stop" : "Sending replaces the current answer") : "Send";
  }

  function ask() {
    const typed = ta.value.trim();
    if (!typed) return;
    if (inFlight && typeof chrome.malabr.stop === "function") { chrome.malabr.stop(); return; }
    ta.value = ""; ta.style.height = "auto";

    let prompt = typed;
    let chip = null;
    if (usePage.checked) {
      const body = pageText();
      if (body) {
        prompt = `Here is the text of the page the user is viewing (${location.href}):\n\n`
               + body + `\n\n---\nUser question: ${typed}`;
        chip = `page attached (${body.length} chars)`;
      }
    }

    const you = render("you", typed);
    if (chip) {
      const c = document.createElement("div");
      c.className = "chip"; c.textContent = chip;
      you.insertBefore(c, you.firstChild);
    }
    transcript.push({ who: "you", text: typed }); save();

    const turn = render("malabr", "");
    setBusy(true);
    chrome.malabr.generate({ prompt }, (id) => {
      if (chrome.runtime.lastError) {
        note(turn, chrome.runtime.lastError.message, "err");
        inFlight = null; setBusy(false); return;
      }
      live.set(id, { turn, bodyEl: turn.querySelector(".body"), thinkEl: null, raw: "" });
      inFlight = id;
    });
  }

  act.addEventListener("click", () => {
    if (inFlight && typeof chrome.malabr.stop === "function") { chrome.malabr.stop(); return; }
    ask();
  });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });
  ta.addEventListener("input", () => {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 130) + "px";
  });
  showThink.addEventListener("change", () => {
    shadow.querySelectorAll("details.think").forEach((d) => (d.open = showThink.checked));
  });
  $(".clear").addEventListener("click", () => {
    log.innerHTML = ""; transcript = []; save();
    note(log, "display cleared -- the model still holds this conversation", "note");
  });

  restore();
  ta.focus();
})();
