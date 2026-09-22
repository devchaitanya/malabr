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
  // for a 2048-token session share. An oversized prompt is REJECTED outright
  // (compaction cannot help a single prompt that exceeds the whole budget), so
  // the page text plus the wrapper plus the user's question plus the running
  // conversation must all fit. At a worst-case ~3 chars/token that budget is
  // ~5300 chars for the WHOLE prompt; 4000 for the page body leaves room for
  // the wrapper, the question, and a few turns of history before compaction.
  const MAX_PAGE_CHARS = 4000;

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

      .hdr { display: flex; align-items: center; gap: 8px; padding: 10px 12px;
             border-bottom: 1px solid #3d3d3a; background: #1f1e1d; }
      .dot { width: 8px; height: 8px; border-radius: 50%; background: #c96442;
             flex: none; }
      .title { font-weight: 600; font-size: 13px; letter-spacing: .2px; flex: none; }
      .origin { margin-left: auto; font-size: 11px; color: #8a8880;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                min-width: 0; }
      /* Model picker on its own row -- in the header it crowded the origin and
         the icons on a 400px panel. */
      .modelbar { display: flex; align-items: center; gap: 7px; padding: 6px 12px;
                  border-bottom: 1px solid #3d3d3a; background: #1f1e1d; }
      .mlabel { font-size: 10px; text-transform: uppercase; letter-spacing: .5px;
                color: #8a8880; flex: none; }
      .model { flex: 1; min-width: 0; font-size: 11.5px; background: #262624;
               color: #cfcec7; border: 1px solid #46453f; border-radius: 6px;
               padding: 3px 6px; }
      .model:disabled { opacity: .5; cursor: progress; }
      .icon { border: 0; background: transparent; color: #a3a096; cursor: pointer;
              font-size: 15px; line-height: 1; padding: 4px 7px; border-radius: 6px; }
      .icon:hover { background: #34332f; color: #f5f4ef; }

      .log { flex: 1; overflow-y: auto; padding: 16px 14px; scroll-behavior: smooth; }
      .log::-webkit-scrollbar { width: 9px; }
      .log::-webkit-scrollbar-thumb { background: #46453f; border-radius: 5px; }

      .turn { margin-bottom: 18px; }
      /* Column so the "page attached" chip sits ABOVE the bubble, not squished
         beside it -- both right-aligned. */
      .turn.you { display: flex; flex-direction: column; align-items: flex-end; }
      .bubble { max-width: 86%; padding: 9px 13px; border-radius: 14px;
                background: #37362f; white-space: pre-wrap; word-wrap: break-word; }
      .turn.ai .body { word-wrap: break-word; }
      .body p { margin: 0 0 8px; }
      .body p:last-child { margin-bottom: 0; }
      .body strong { font-weight: 650; color: #fdfcf8; }
      .body em { font-style: italic; }
      .body code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                   font-size: 12px; background: #1c1b19; border: 1px solid #3d3d3a;
                   border-radius: 4px; padding: 0 4px; }
      .body pre { background: #1c1b19; border: 1px solid #3d3d3a; border-radius: 8px;
                  padding: 9px 11px; overflow-x: auto; margin: 0 0 8px; }
      .body pre code { background: none; border: 0; padding: 0; font-size: 12px;
                       white-space: pre; }
      .body h4 { font-size: 13px; font-weight: 650; color: #fdfcf8; margin: 10px 0 6px; }
      .body ul, .body ol { margin: 0 0 8px; padding-left: 20px; }
      .body li { margin: 2px 0; }
      .body > *:first-child { margin-top: 0; }
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
      .chip { align-self: flex-end; max-width: 100%; font-size: 10.5px;
              color: #a3a096; background: #34332f; border-radius: 5px;
              padding: 2px 7px; margin-bottom: 5px; white-space: nowrap;
              overflow: hidden; text-overflow: ellipsis; }
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
        <div class="modelbar">
          <span class="mlabel">model</span>
          <select class="model" title="Switching clears every conversation"></select>
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
  const usePage = $(".usepage"), showThink = $(".showthink"), modelSel = $(".model");
  $(".origin").textContent = location.host.replace(/^www\./, "");

  $(".min").addEventListener("click", () => wrap.classList.add("min"));
  $(".pill").addEventListener("click", () => { wrap.classList.remove("min"); ta.focus(); });

  // ---- transcript persistence (section 2) --------------------------------
  // chrome.storage.session, NOT storage.local: session storage is memory-backed
  // and dies with the browser, which is what section 4's "nothing durable"
  // rule requires. storage.local would write conversations to disk.
  const KEY = "malabr:" + location.origin;
  let transcript = [];

  // chrome.storage.session is gated to TRUSTED_CONTEXTS by default, i.e. NOT
  // content scripts. bg.js opens it with setAccessLevel; until that lands the
  // namespace is simply absent here, so a content script's first restore() can
  // race the service worker's cold start. sessReady() waits for the namespace
  // to appear (bounded), then read/write work normally.
  function sessReady(cb, tries) {
    tries = tries || 0;
    if (chrome.storage && chrome.storage.session) return cb(chrome.storage.session);
    if (tries < 15) return setTimeout(() => sessReady(cb, tries + 1), 200);
    console.warn("MALABR: chrome.storage.session never became available "
                 + "(bg.js / setAccessLevel not applied?)");
  }

  function save() {
    sessReady((s) => {
      // Last 40 turns only: chrome.storage.session has a ~10MB quota shared by
      // the whole extension, and a turn can carry a page-sized reply. Older
      // turns stop persisting across reloads; the server's KV still has them
      // until compaction, so nothing the model knows is lost, only the display.
      try { s.set({ [KEY]: transcript.slice(-40) }); } catch (e) {}
    });
  }
  function restore() {
    sessReady((s) => {
      try {
        s.get(KEY, (o) => {
          if (chrome.runtime.lastError) return;
          const saved = o && o[KEY];
          if (!Array.isArray(saved) || !saved.length) return;
          // The SERVER still holds this conversation after a reload: the
          // session key is (extension, tab, origin) and none of those change.
          // Without restoring the display the model would remember what the
          // user cannot see -- the desync section 2 exists to prevent.
          saved.forEach((m) => render(m.who, m.text, m.think, m.meta));
          transcript = saved;
          note(log.lastElementChild || log, "restored after reload", "note");
        });
      } catch (e) {}
    });
  }

  // ---- markdown (small, injection-safe) --------------------------------
  // The model emits markdown heavily. This renders the common subset and
  // NOTHING else: every character is HTML-escaped first, then a fixed set of
  // patterns is turned into a fixed set of safe tags. No raw HTML from the
  // model can survive -- it matters because page text (which the model may be
  // repeating) is attacker-controlled when "include page text" is on.
  function esc(s) {
    return s.replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  function mdInline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,!?]|$)/g, "$1<em>$2</em>")
      .replace(/(^|[\s(])_([^_\s][^_]*?)_(?=[\s).,!?]|$)/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");   // links -> just their text
  }
  function renderMarkdown(src) {
    const lines = (src || "").split("\n");
    let html = "", i = 0, para = [];
    const flushPara = () => {
      if (para.length) { html += "<p>" + para.map(mdInline).join("<br>") + "</p>"; para = []; }
    };
    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^\s*```/);
      if (fence) {                                   // fenced code block
        flushPara();
        const body = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
        i++;                                         // skip closing fence
        html += "<pre><code>" + esc(body.join("\n")) + "</code></pre>";
        continue;
      }
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { flushPara(); html += "<h4>" + mdInline(h[2]) + "</h4>"; i++; continue; }
      const li = line.match(/^\s*([*+-]|\d+[.)])\s+(.*)$/);
      if (li) {
        flushPara();
        const ordered = /\d/.test(li[1]);
        const items = [];
        while (i < lines.length) {
          const m = lines[i].match(/^\s*([*+-]|\d+[.)])\s+(.*)$/);
          if (!m) break;
          items.push("<li>" + mdInline(m[2]) + "</li>");
          i++;
        }
        html += (ordered ? "<ol>" : "<ul>") + items.join("") + (ordered ? "</ol>" : "</ul>");
        continue;
      }
      if (line.trim() === "") { flushPara(); i++; continue; }
      para.push(line);
      i++;
    }
    flushPara();
    return html || "";
  }
  function setBody(el, text) { el.innerHTML = renderMarkdown(text); }

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
      body.className = "body"; setBody(body, text || "");
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
  // Which page text (if any) the server already has for this session. Reset on
  // "New chat" and on toggling the checkbox, so a deliberate re-check re-sends.
  let pageHash = null;
  function hashStr(s) {
    let h = 5381;
    for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
    return h;
  }

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
  let inFlight = null;             // requestId once generate()'s callback returns
  let sending = false;             // set SYNCHRONOUSLY the instant a send starts.
                                   // inFlight is only assigned in generate()'s
                                   // async callback, so without this a second
                                   // Enter/paste in the same frame slips past the
                                   // guard and opens a parallel generation --
                                   // section 6b allows exactly one per session.

  // How long to wait for generate()'s callback (which only hands back the
  // request id -- tokens arrive separately on onToken) before assuming the
  // extension message was dropped. Must exceed any legitimate round-trip: on a
  // CPU-saturated box under a large prompt the old 10s could expire while the
  // call was still healthy, dropping `sending` and letting a second Enter open
  // the parallel generation the latch exists to stop. Shared with meta().
  const GENERATE_ACK_TIMEOUT_MS = 15000;

  // ---- model switcher ---------------------------------------------------
  // The page-facing API is only generate()/stop(), so a control command rides
  // IN a generate() call: the prompt is the sentinel below, and the server
  // answers with a JSON blob as ordinary token frames. No new IDL/C++ route.
  const metaReqs = new Map();       // requestId -> { buf, resolve, reject }
  let switching = false;

  function meta(cmd) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const t = setTimeout(() => {
        if (!settled) { settled = true; reject(new Error("meta timeout")); }
      }, GENERATE_ACK_TIMEOUT_MS);
      const fin = (fn) => (v) => {
        if (settled) return;
        settled = true; clearTimeout(t); fn(v);
      };
      chrome.malabr.generate({ prompt: "\u0000MALABR::" + cmd }, (id) => {
        if (chrome.runtime.lastError) {
          fin(reject)(new Error(chrome.runtime.lastError.message));
          return;
        }
        metaReqs.set(id, { buf: "", resolve: fin(resolve), reject: fin(reject) });
      });
    });
  }

  async function refreshModels(want) {
    let info;
    try { info = await meta("list"); } catch (e) { return null; }
    modelSel.innerHTML = "";
    const list = info.available && info.available.length
      ? info.available : [info.current].filter(Boolean);
    list.forEach((m) => {
      const o = document.createElement("option");
      o.value = m; o.textContent = m;
      if (m === (want || info.current)) o.selected = true;
      modelSel.appendChild(o);
    });
    return info;
  }

  async function switchModel(name) {
    if (switching) return;
    if (!confirm(`Switch to "${name}"?\n\n`
      + `This ends every MALABR conversation in every tab. The first switch to `
      + `a model takes ~2 minutes while it calibrates.`)) {
      refreshModels();                       // snap the <select> back
      return;
    }
    switching = true;
    modelSel.disabled = true;
    // The server re-execs on a switch: its sessions and KV caches go with it,
    // so this transcript is stale everywhere. Clear it to match.
    log.innerHTML = ""; transcript = []; save();
    note(log, `switching to ${name} — loading the model, up to ~2 min on first use`, "note");
    try { await meta("switch " + name); } catch (e) { /* the connection drops as it re-execs; expected */ }
    const deadline = Date.now() + 240000;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 3000));
      const info = await refreshModels(name);
      if (info && info.current === name) {
        note(log, `now running ${name}`, "note");
        switching = false; modelSel.disabled = false;
        return;
      }
    }
    note(log, `switch to ${name} timed out — check the server`, "err");
    switching = false; modelSel.disabled = false;
    refreshModels();
  }

  modelSel.addEventListener("change", () => switchModel(modelSel.value));

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
    const mr = metaReqs.get(id);
    if (mr) { mr.buf += text; return; }
    const s = live.get(id);
    if (!s) return;
    if (text.indexOf("\u0000MALABR:cap") !== -1) {
      s.capped = true;
      text = text.replace("\u0000MALABR:cap", "");
      if (!text) return;
    }
    s.raw += text;
    const { think, body } = splitThink(s.raw);
    if (think) {
      if (!s.thinkEl) { s.thinkEl = thinkBlock(""); s.turn.insertBefore(s.thinkEl, s.bodyEl); }
      s.thinkEl.querySelector(".inner").textContent = think;
    }
    setBody(s.bodyEl, body);
    s.bodyEl.classList.add("caret");
    log.scrollTop = log.scrollHeight;
  });

  chrome.malabr.onComplete.addListener((id, error) => {
    const mr = metaReqs.get(id);
    if (mr) {
      metaReqs.delete(id);
      if (error) { mr.reject(new Error(error)); return; }
      try { mr.resolve(JSON.parse(mr.buf || "{}")); }
      catch (e) { mr.reject(e); }
      return;
    }
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
      if (s.capped) note(s.turn, "stopped at the length limit — ask it to continue", "note");
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

  // ---- /bench : MALABR vs Chrome's built-in Prompt API ------------------
  function chromeLM() {
    return self.LanguageModel
        || (self.ai && (self.ai.languageModel || self.ai.assistant))
        || null;
  }
  async function runChromeLM(prompt, onTok) {
    const LM = chromeLM();
    if (!LM) throw new Error("Chrome Prompt API not present in this build");
    const avail = LM.availability ? await LM.availability()
                : LM.capabilities ? (await LM.capabilities()).available : "unknown";
    if (avail === "unavailable" || avail === "no")
      throw new Error("Prompt API present but model unavailable (" + avail + ")");
    const s = await LM.create();
    const stream = s.promptStreaming ? s.promptStreaming(prompt)
                : s.prompt ? null : null;
    if (stream) {
      let prev = "";
      for await (const chunk of stream) {
        // some versions yield cumulative text, some yield deltas
        const delta = chunk.startsWith(prev) ? chunk.slice(prev.length) : chunk;
        prev = chunk.length >= prev.length ? chunk : prev + chunk;
        onTok(delta);
      }
    } else {
      onTok(await s.prompt(prompt));
    }
    s.destroy && s.destroy();
  }
  function runMalabr(prompt, onTok) {
    return new Promise((resolve, reject) => {
      chrome.malabr.generate({ prompt }, (id) => {
        if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
        const h = (rid, text) => { if (rid === id) onTok(text); };
        const done = (rid, err) => {
          if (rid !== id) return;
          chrome.malabr.onToken.removeListener(h);
          chrome.malabr.onComplete.removeListener(done);
          err && !/stop|supersed/i.test(err) ? reject(new Error(err)) : resolve();
        };
        chrome.malabr.onToken.addListener(h);
        chrome.malabr.onComplete.addListener(done);
      });
    });
  }
  async function timeOne(label, fn, prompt) {
    const t0 = performance.now();
    let first = null, chars = 0;
    try {
      await fn(prompt, (t) => { if (first === null) first = performance.now(); chars += (t || "").length; });
    } catch (e) {
      return { label, error: e.message };
    }
    const total = performance.now() - t0;
    const gen = first === null ? total : total - (first - t0);
    return {
      label,
      ttft_ms: first === null ? null : Math.round(first - t0),
      total_ms: Math.round(total),
      chars,
      cps: gen > 0 ? +(chars / (gen / 1000)).toFixed(1) : 0,   // chars/sec (~3-4 chars/token)
    };
  }
  async function bench(prompt) {
    prompt = prompt || "Write two short paragraphs about how tides work.";
    const turn = render("malabr", "");
    const body = turn.querySelector(".body");
    body.textContent = "benchmarking… (MALABR, then Chrome Prompt API)";
    const rows = [];
    rows.push(await timeOne("MALABR (" + (modelSel.value || "?") + ")", runMalabr, prompt));
    rows.push(await timeOne("Chrome Prompt API (Gemini Nano)", runChromeLM, prompt));
    const fmt = (r) => r.error
      ? `${r.label}: ${r.error}`
      : `${r.label}: ttft ${r.ttft_ms ?? "–"}ms · ${r.cps} chars/s · ${r.total_ms}ms total · ${r.chars} chars`;
    setBody(body, "**benchmark** — prompt: _" + prompt + "_\n\n"
                 + rows.map((r) => "- " + fmt(r)).join("\n"));
  }

  function ask() {
    const typed = ta.value.trim();
    if (!typed) return;
    if (/^\/bench\b/.test(typed)) {
      // Same single-generation rule as a normal turn: a benchmark runs two
      // full generations back to back, and an Enter meanwhile must not open a
      // parallel one. Held via `sending` for the whole run.
      if (inFlight || sending) return;
      sending = true;
      ta.value = ""; ta.style.height = "auto";
      render("you", typed);
      bench(typed.replace(/^\/bench\b\s*/, "") || null)
        .finally(() => { sending = false; });
      return;
    }
    if (inFlight || sending) {
      // A generation is already in flight for this tab. A second one must not
      // start alongside it: if a real stop is available, treat this as "stop
      // the current answer"; otherwise just swallow the keystroke.
      if (inFlight && typeof chrome.malabr.stop === "function") chrome.malabr.stop();
      return;
    }
    sending = true;
    // Safety net: if generate()'s callback never comes back (dropped extension
    // message), don't leave the composer locked forever. On expiry also tear
    // down this turn's half-open UI, so the next Enter starts genuinely fresh
    // instead of racing a callback that might still be in flight.
    const sendingGuard = setTimeout(() => {
      if (!sending) return;
      sending = false;
      note(turn, "the server did not respond — try again", "err");
      turn.querySelector(".body")?.classList.remove("caret");
      setBusy(false);
    }, GENERATE_ACK_TIMEOUT_MS);
    ta.value = ""; ta.style.height = "auto";

    let prompt = typed;
    let chip = null;
    if (usePage.checked) {
      const body = pageText();
      // Send the page ONCE per session. The server keeps the conversation in
      // its KV cache, so re-sending the whole page every turn just re-prefills
      // it, burns the context budget, and makes the model re-read it. Only
      // re-attach if the page text actually changed (SPA navigation, new tab
      // content) or the conversation was cleared.
      const h = body ? hashStr(body) : null;
      if (body && h !== pageHash) {
        prompt = `Here is the text of the page the user is viewing (${location.href}):\n\n`
               + body + `\n\n---\nUser question: ${typed}`;
        chip = `page ${pageHash === null ? "attached" : "re-attached (changed)"} (${body.length} chars)`;
        pageHash = h;
      } else if (body) {
        chip = "page already in context";
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
      sending = false; clearTimeout(sendingGuard);
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
  // Toggling the checkbox does NOT force a re-read: the page is already in the
  // server's context, and only a real change to the page text (hashed in ask())
  // or a "New chat" re-attaches it.
  $(".clear").addEventListener("click", async () => {
    // §3: a real teardown, not a display clear. The display goes now (instant
    // feedback); the server-side session + KV are ended over the meta channel
    // and the note reports whether that was confirmed.
    log.innerHTML = ""; transcript = []; save(); pageHash = null;
    inFlight = null; sending = false;
    let acked = false;
    try { const r = await meta("new"); acked = !!(r && r.ok); } catch (e) {}
    note(log, acked
      ? "new chat — the previous conversation was released on the server"
      : "display cleared, but the server did not confirm the teardown — the model may still hold this conversation",
      acked ? "note" : "err");
  });

  restore();
  // The server may still be loading its model when the panel appears (cold
  // start, or MalabrManager just spawned it). A single failed list would leave
  // the model picker empty for the life of the tab, so retry with backoff
  // until it answers -- bounded, and only until the first success.
  (async function primeModels(tries) {
    for (let i = 0; i < tries; i++) {
      if (await refreshModels()) return;
      await new Promise((r) => setTimeout(r, Math.min(1500 * (i + 1), 8000)));
    }
  })(20);
  ta.focus();
})();
