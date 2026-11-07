/* R3 labeling page. The server renders every card; this script wires the
   keys, keeps the session's memory (which jobs were shown, what to undo) in
   sessionStorage, and adds it to every request HTMX makes from the page.
   Nothing here parses HTML. */
(() => {
  "use strict";

  const root = document.getElementById("r3-label");
  if (!root || typeof htmx === "undefined") return;
  const base = root.dataset.basePath || "";
  const url = (path) => `${base}${path}`;
  const STORE = { seen: "r3.label.seen.v1", undo: "r3.label.undo.v1", session: "r3.label.session.v1" };
  const SEEN_KEPT = 2000; // ids remembered for the session
  const SEEN_SENT = 1000; // the most recent ones a request carries (keeps the URL short)
  const UNDO_KEPT = 200;

  const $ = (sel, el = root) => el.querySelector(sel);
  const $$ = (sel, el = root) => Array.from(el.querySelectorAll(sel));

  const store = {
    get(key) {
      try { return JSON.parse(sessionStorage.getItem(key)); } catch (_) { return null; }
    },
    set(key, value) {
      try { sessionStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* private mode: the session forgets */ }
    },
  };

  function randomId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, "").slice(0, 24);
    return Math.random().toString(36).slice(2) + Date.now().toString(36);
  }

  // --- session memory ---------------------------------------------------------

  const seen = new Set((store.get(STORE.seen) || []).map(Number).filter(Number.isFinite));
  let undoStack = (store.get(STORE.undo) || []).filter((e) => e && Number.isFinite(e.id));
  let sessionId = store.get(STORE.session);
  if (typeof sessionId !== "string" || !sessionId) {
    sessionId = randomId();
    store.set(STORE.session, sessionId);
  }
  let shownAt = performance.now();
  let pending = null; // {kind, id}: the card a request is leaving behind
  let mode = null; // "load" | "undo": a request that leaves no card behind

  const saveSeen = () => store.set(STORE.seen, Array.from(seen).slice(-SEEN_KEPT));
  const saveUndo = () => store.set(STORE.undo, (undoStack = undoStack.slice(-UNDO_KEPT)));

  function card() { return $("#lab-card[data-job-id]"); }
  function currentId() {
    const c = card();
    const id = c ? Number(c.dataset.jobId) : NaN;
    return Number.isFinite(id) ? id : null;
  }
  function sinceMidnight() {
    const d = new Date();
    d.setHours(0, 0, 0, 0);
    return d.toISOString();
  }
  function secondsSpent() { return Math.max(0, Math.round((performance.now() - shownAt) / 1000)); }

  // --- the questions ----------------------------------------------------------

  let active = 0;
  const questions = () => $$(".lab-q");
  const answered = (q) => !!q.querySelector("input:checked");

  function setActive(i) {
    const qs = questions();
    if (!qs.length) return;
    active = Math.max(0, Math.min(i, qs.length - 1));
    qs.forEach((q, n) => q.classList.toggle("active", n === active));
  }

  function firstUnanswered() {
    const qs = questions();
    const i = qs.findIndex((q) => !answered(q));
    return i < 0 ? qs.length - 1 : i;
  }

  function exclusive(q, changed) {
    // A multi-select's stand-alone option ("none") clears the others and is cleared by them.
    const only = q.dataset.exclusive;
    if (!only || !changed.checked) return;
    $$("input", q).forEach((input) => {
      if (input !== changed && (changed.value === only || input.value === only)) input.checked = false;
    });
  }

  function pick(q, index) {
    const opt = $$(".opt", q)[index - 1];
    const input = opt && opt.querySelector("input");
    if (!input) return;
    input.checked = input.type === "checkbox" ? !input.checked : true;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function refreshStatus() {
    const status = $("#label-status");
    const qs = questions();
    if (!status || !qs.length) return;
    const first = qs[0];
    if (!answered(first)) {
      const keys = $$(".opt", first).map((o) => `${o.dataset.key} ${o.querySelector("span").textContent}`).join(" · ");
      status.textContent = `Start with “${first.querySelector("legend").firstChild.textContent.trim()}”: ${keys}.`;
      return;
    }
    const open = qs.slice(1).filter((q) => !answered(q)).length;
    status.textContent = open
      ? `Enter saves · ${open} unanswered question${open === 1 ? "" : "s"} stored as skipped`
      : "Enter saves";
  }

  root.addEventListener("change", (e) => {
    const input = e.target;
    if (!(input instanceof HTMLInputElement)) return;
    const q = input.closest(".lab-q");
    if (!q) return;
    if (input.type === "checkbox") exclusive(q, input);
    const i = questions().indexOf(q);
    setActive(input.type === "radio" ? i + 1 : i);
    refreshStatus();
  });

  // --- actions -----------------------------------------------------------------

  function flash(text) {
    const el = $("#label-flash");
    if (!el) return;
    el.textContent = text;
    el.hidden = !text;
  }

  function save() {
    const form = $("#label-form");
    if (!form) return;
    const qs = questions();
    if (qs.length && !answered(qs[0])) {
      flash("Answer “Would I apply to this?” first (1, 2 or 3).");
      setActive(0);
      return;
    }
    form.requestSubmit();
  }

  function skip() {
    const button = $("#label-skip");
    if (button) button.click();
  }

  function undo() {
    const last = undoStack.pop();
    saveUndo();
    if (!last) {
      flash("Nothing to undo in this session.");
      return;
    }
    seen.delete(last.id);
    saveSeen();
    mode = "undo";
    htmx.ajax("GET", url(`/label/${last.id}`), { target: "#label-card", swap: "innerHTML" });
  }

  function kindOf(elt) {
    if (mode) return mode;
    if (!elt) return null;
    if (elt.id === "label-form") return "save";
    if (elt.id === "label-skip") return "skip";
    if (elt.id === "label-filters") return "filter";
    return null;
  }

  // Every request from the page carries the session: the ids shown (plus the card being left),
  // the labeler's midnight for "today", the session id and the time spent on the card.
  document.body.addEventListener("htmx:configRequest", (e) => {
    const elt = e.detail.elt;
    if (!(elt && (root.contains(elt) || elt === document.body))) return;
    const kind = kindOf(elt);
    mode = null;
    const id = currentId();
    const carried = new Set(seen);
    if (kind === "save" || kind === "skip" || kind === "filter") {
      if (id !== null) carried.add(id);
      pending = id !== null ? { kind, id } : null;
    } else {
      pending = null;
    }
    const p = e.detail.parameters;
    p.seen = Array.from(carried).slice(-SEEN_SENT).join(",");
    p.since = sinceMidnight();
    p.session_id = sessionId;
    p.seconds_spent = String(secondsSpent());
  });

  document.body.addEventListener("htmx:afterSwap", (e) => {
    if (e.detail.target !== $("#label-card")) return;
    if (pending) {
      seen.add(pending.id);
      undoStack.push(pending);
      saveSeen();
      saveUndo();
      pending = null;
    }
    const trigger = e.detail.requestConfig && e.detail.requestConfig.elt;
    if (trigger && trigger.id === "label-filters") {
      // The filter lives in the URL, as on /jobs; the session memory does not.
      const params = new URLSearchParams();
      new FormData(trigger).forEach((value, key) => { if (value) params.append(key, value); });
      const qs = params.toString();
      history.replaceState(null, "", url("/label") + (qs ? `?${qs}` : ""));
    }
    shownAt = performance.now();
    setActive(firstUnanswered());
    refreshStatus();
    const desc = $(".lab-desc");
    if (desc) desc.scrollTop = 0;
  });

  document.body.addEventListener("htmx:responseError", (e) => {
    const xhr = e.detail.xhr;
    let message = `Request failed (HTTP ${xhr.status}).`;
    try {
      const body = JSON.parse(xhr.responseText);
      if (body && body.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* not JSON */ }
    flash(message);
    pending = null;
  });

  root.addEventListener("click", (e) => {
    const t = e.target instanceof Element ? e.target.closest("#label-undo") : null;
    if (t) undo();
  });

  // --- keys -----------------------------------------------------------------------

  document.addEventListener("keydown", (e) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target instanceof HTMLElement ? e.target : null;
    const typing = !!t && t.matches("input:not([type=radio]):not([type=checkbox]), textarea, select");
    if (e.key === "Enter") {
      if (t && t.matches("button, a, select, summary")) return; // the element's own Enter
      if (typing && !(t && t.id === "label-note")) return;
      e.preventDefault();
      save();
      return;
    }
    if (typing) {
      if (e.key === "Escape" && t) t.blur();
      return;
    }
    if (/^[0-9]$/.test(e.key)) {
      const qs = questions();
      if (!qs.length) return;
      e.preventDefault();
      pick(qs[active], e.key === "0" ? 10 : Number(e.key));
      return;
    }
    switch (e.key) {
      case "s": e.preventDefault(); skip(); break;
      case "u": e.preventDefault(); undo(); break;
      case "n": {
        const note = $("#label-note");
        if (note) { e.preventDefault(); note.focus(); }
        break;
      }
      case "ArrowDown": case "j": e.preventDefault(); setActive(active + 1); break;
      case "ArrowUp": case "k": e.preventDefault(); setActive(active - 1); break;
      default: break;
    }
  });

  // --- start ----------------------------------------------------------------------

  setActive(firstUnanswered());
  refreshStatus();
  // Re-ask for the card with this session's memory: the server-rendered one did not know what was
  // skipped before a reload, nor the labeler's midnight for "today". The order is stable, so a job
  // not yet seen comes back unchanged.
  const filters = $("#label-filters");
  if (filters) {
    mode = "load";
    htmx.ajax("GET", url("/label/next"), { source: filters, target: "#label-card", swap: "innerHTML" });
  }
})();
