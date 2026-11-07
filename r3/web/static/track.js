/* R3's click trail: every click, form submit, page view and menu choice on
   an R3 page goes to /api/events as an audit event (r3/web/routes_events.py).

   What is sent: the visible label of what was clicked, the element, the page,
   and for links where they lead. What is never sent: anything typed into a
   field (text boxes, passwords, the chat box), only which control changed.

   Each click starts a trace id, and the requests it sets off within the next
   few seconds carry it, so R3 can join the click to the request, the model
   calls inside it and the records it changed: fetch and HTMX requests in the
   X-R3-Trace header, and a form post or link (a navigation, which cannot
   carry a header) in the short-lived r3_trace cookie, cleared again once the
   next page loads. */
(() => {
  "use strict";
  const script = document.currentScript;
  const base = (script && script.dataset.base) || "";
  const ENDPOINT = `${base}/api/events`;
  const LINK_MS = 4000;
  const queue = [];
  let trace = "";
  let traceUntil = 0;

  const hex = () => {
    const b = new Uint8Array(16);
    (window.crypto || window.msCrypto).getRandomValues(b);
    return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  };
  const here = () => location.pathname.slice(base.length) || "/";
  const clip = (s, n) => {
    const t = String(s || "").replace(/\s+/g, " ").trim();
    return t.length > n ? `${t.slice(0, n - 1)}…` : t;
  };
  const describe = (el) => {
    const tag = el.tagName.toLowerCase();
    const id = el.id ? `#${el.id}` : "";
    const name = el.getAttribute("name") ? `[name=${el.getAttribute("name")}]` : "";
    return clip(`${tag}${id}${name}`, 120);
  };
  const labelOf = (el) => {
    const aria = el.getAttribute("aria-label");
    if (aria) return clip(aria, 120);
    if (el.matches("input[type=checkbox], input[type=radio]")) {
      const lab = el.labels && el.labels[0] ? el.labels[0].textContent : el.name;
      return clip(`${lab}: ${el.checked ? "on" : "off"}`, 120);
    }
    if (el.matches("input[type=submit], input[type=button]")) return clip(el.value, 120);
    if (el.tagName === "SELECT") {
      const opt = el.options[el.selectedIndex];
      const lab = el.labels && el.labels[0] ? el.labels[0].textContent : el.name || el.id;
      return clip(`${lab}: ${opt ? opt.textContent : ""}`, 120);
    }
    return clip(el.innerText || el.textContent || el.title || el.name || el.id || el.tagName, 120);
  };
  const internalHref = (el) => {
    const raw = el.getAttribute && el.getAttribute("href");
    if (!raw || raw.startsWith("#")) return "";
    try {
      const u = new URL(raw, location.href);
      if (u.origin !== location.origin) return u.origin;  // the site, not the full address
      return clip(u.pathname.slice(base.length) + u.search, 300);
    } catch {
      return "";
    }
  };

  function push(type, el, label) {
    queue.push({ type, label: label || (el ? labelOf(el) : ""), target: el ? describe(el) : "",
                 path: here(), href: el ? internalHref(el) : "", trace });
    if (queue.length >= 20) send();
  }

  function send(beacon) {
    if (!queue.length) return;
    const body = JSON.stringify({ events: queue.splice(0, 50) });
    try {
      if (beacon && navigator.sendBeacon) {
        navigator.sendBeacon(ENDPOINT, new Blob([body], { type: "application/json" }));
      } else {
        realFetch(ENDPOINT, { method: "POST", body, keepalive: true, credentials: "same-origin",
                              headers: { "Content-Type": "application/json" } }).catch(() => {});
      }
    } catch { /* the trail never breaks the page */ }
  }

  const COOKIE_PATH = base || "/";
  const setTraceCookie = (t) => {
    document.cookie = `r3_trace=${t}; Max-Age=${Math.ceil(LINK_MS / 1000)}; Path=${COOKIE_PATH}; SameSite=Lax`;
  };
  const freshTrace = () => (Date.now() < traceUntil ? trace : "");
  const realFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const t = freshTrace();
    if (t) {
      try {
        const url = new URL(typeof input === "string" ? input : input.url, location.href);
        if (url.origin === location.origin && !url.pathname.endsWith("/api/events")) {
          const headers = new Headers((init && init.headers) || (input instanceof Request ? input.headers : undefined));
          if (!headers.has("X-R3-Trace")) headers.set("X-R3-Trace", t);
          init = { ...(init || {}), headers };
        }
      } catch { /* leave the request as it was */ }
    }
    return realFetch(input, init);
  };
  document.addEventListener("htmx:configRequest", (e) => {
    const t = freshTrace();
    if (t && e.detail && e.detail.headers) e.detail.headers["X-R3-Trace"] = t;
  });

  const CLICKABLE = "button, a[href], summary, [role=button], [role=tab], label, " +
    "input[type=checkbox], input[type=radio], input[type=submit], input[type=button]";
  document.addEventListener("click", (e) => {
    const el = e.target instanceof Element ? e.target.closest(CLICKABLE) : null;
    if (!el) return;
    // A click on a label also clicks its checkbox; record the checkbox only.
    if (el.tagName === "LABEL" && el.control) return;
    trace = hex();
    traceUntil = Date.now() + LINK_MS;
    setTraceCookie(trace);
    push("click", el);
  }, true);
  document.addEventListener("submit", (e) => {
    const form = e.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (!freshTrace()) { trace = hex(); traceUntil = Date.now() + LINK_MS; setTraceCookie(trace); }
    const label = form.getAttribute("aria-label") || form.id || form.getAttribute("action") || "form";
    push("submit", form, clip(label, 120));
  }, true);
  document.addEventListener("change", (e) => {
    const el = e.target;
    if (el instanceof HTMLSelectElement) push("change", el);
  }, true);

  // The navigation that carried the last click's cookie is done; later
  // requests from this page start their own traces.
  document.cookie = `r3_trace=; Max-Age=0; Path=${COOKIE_PATH}; SameSite=Lax`;
  trace = hex();
  traceUntil = 0;
  push("view", null, clip(document.title, 120));
  setInterval(() => send(false), 2000);
  addEventListener("pagehide", () => send(true));
  addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") send(true); });
})();
