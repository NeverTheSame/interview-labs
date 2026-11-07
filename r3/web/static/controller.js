/* R3 Controller: a streaming chat on the left, the pipelines panel on the
   right (suggestions, runs, jobs, statistics). Two rules hold throughout:
   server text only reaches the page through textContent, never parsed as
   HTML, and every URL is built from the base path the template renders,
   because the portal may be served under a path prefix (R3_ROOT_PATH). */
(() => {
  "use strict";
  const root = document.getElementById("r3-controller");
  if (!root) return;

  /* ================================================================ helpers */

  const $ = (id) => document.getElementById(id);
  const BASE = (root.dataset.basePath || "").replace(/\/+$/, "");
  const api = (path) => BASE + path;

  /* Build an element. String children become text nodes and props become
     attributes or listeners, so nothing built here is ever parsed as markup. */
  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "on") for (const [type, fn] of Object.entries(value)) node.addEventListener(type, fn);
      else if (key === "hidden" || key === "disabled") node[key] = true;
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat(Infinity)) if (child != null && child !== false && child !== "") node.append(child);
    return node;
  }

  // Server links: absolute http(s) only, so no javascript: or data: URL ever becomes clickable.
  const webUrl = (u) => (typeof u === "string" && /^https?:\/\/\S+$/i.test(u) ? u : "");
  // Portal paths ("/jobs?status=new") join the base path; "//host" is refused.
  // A path on this portal; "//host" and "/\host" (which browsers read as "//host") are other sites.
  const portalUrl = (p) => (typeof p === "string" && /^\/(?![\/\\])/.test(p) ? BASE + p : "");
  const jobUrl = (id) => `${BASE}/jobs/${encodeURIComponent(id)}`;

  // Storage can be disabled or full; the page works without it.
  const store = (area) => ({
    get: (key) => { try { return window[area].getItem(key); } catch { return null; } },
    set: (key, value) => { try { window[area].setItem(key, value); } catch { /* not kept */ } },
    remove: (key) => { try { window[area].removeItem(key); } catch { /* nothing to remove */ } },
  });
  const session = store("sessionStorage");
  const local = store("localStorage");

  /* Who the page is for: admins get the operator page, everyone else the job
     seeker's Assistant. The chat and the panel tab live in sessionStorage
     under keys that carry the account id, so a second account signing in on
     this tab never sees the first one's conversation. */
  const ADMIN = root.dataset.admin !== "false";
  const ACCOUNT = root.dataset.userId || "0";
  const STATE_PREFIXES = ["r3.controller.chat.v1", "r3.controller.tab.v1", "r3.controller.actions.v1",
    "r3.controller.resume.v1"];
  const ownKey = (base) => `${base}:${ACCOUNT}`;
  function dropControllerState(keepOwn) {
    try {
      const keys = [];
      for (let i = 0; i < window.sessionStorage.length; i += 1) keys.push(window.sessionStorage.key(i));
      keys.filter((k) => k && STATE_PREFIXES.some((b) => k === b || k.startsWith(`${b}:`)))
        .filter((k) => !keepOwn || !k.endsWith(`:${ACCOUNT}`)).forEach((k) => session.remove(k));
    } catch { /* storage off: nothing kept, nothing to drop */ }
  }
  dropControllerState(true); // another account's (or the pre-account) chat goes
  // Signing out forgets this tab's chat before the next person signs in.
  document.querySelectorAll('form[action$="/logout"]').forEach((f) =>
    f.addEventListener("submit", () => dropControllerState(false)));

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  // FastAPI reports errors as {detail: "text"} or {detail: [{msg}, ...]};
  // a middleware's refusal can be plain text, which is shown as sent.
  function detailOf(data, status, plain = "") {
    const detail = data && data.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    const msgs = Array.isArray(detail) ? detail.map((d) => d && d.msg).filter((m) => typeof m === "string") : [];
    if (msgs.length) return msgs.join("; ");
    if (plain.trim()) return plain.trim();
    return status === 503 ? "Runs are unavailable right now."
      : status === 404 ? "Not found (this part of the Controller may not be deployed yet)." : `The server answered ${status}.`;
  }

  async function fetchJSON(path, { method = "GET", body } = {}) {
    const headers = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    let res;
    try {
      res = await fetch(api(path), { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
    } catch {
      throw new ApiError("Couldn’t reach R3. Check your connection.", 0);
    }
    const plain = !res.ok && (res.headers.get("Content-Type") || "").startsWith("text/plain");
    const data = plain ? null : await res.json().catch(() => null);
    if (!res.ok) throw new ApiError(detailOf(data, res.status, plain ? await res.text().catch(() => "") : ""), res.status);
    return data;
  }

  const numberFormat = new Intl.NumberFormat();
  const num = (n) => (typeof n === "number" && Number.isFinite(n) ? numberFormat.format(n) : "—");
  const plural = (n, word) => `${num(n)} ${n === 1 ? word : `${word}s`}`;
  const cap = (s) => String(s || "").charAt(0).toUpperCase() + String(s || "").slice(1);
  const stamp = (iso) => { const t = Date.parse(iso); return Number.isNaN(t) ? null : t; };
  const narrow = () => window.matchMedia("(max-width: 1000px)").matches;

  function ago(iso) {
    const t = stamp(iso);
    if (t === null) return "never";
    const s = Math.max(0, (Date.now() - t) / 1000);
    return s < 45 ? "just now" : s < 3600 ? `${Math.max(1, Math.round(s / 60))}m ago`
      : s < 86400 ? `${Math.floor(s / 3600)}h ago` : `${Math.floor(s / 86400)}d ago`;
  }

  function span(s) {
    if (typeof s !== "number" || !Number.isFinite(s) || s < 0) return "—";
    return s < 1 ? "<1s" : s < 60 ? `${Math.floor(s)}s` : s < 3600 ? `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`
      : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  }

  // Relative times age in place (see tick) instead of re-rendering whole cards.
  const agoEl = (iso, cls) => el("time", { class: cls, datetime: stamp(iso) === null ? null : iso, "data-ago": "",
    title: stamp(iso) === null ? null : new Date(iso).toLocaleString(), text: ago(iso) });
  const since = (node) => `${node.dataset.verb} ${span((Date.now() - (stamp(node.dataset.since) || Date.now())) / 1000)}`;

  function tick() {
    root.querySelectorAll("time[data-ago]").forEach((t) => { t.textContent = ago(t.getAttribute("datetime")); });
    root.querySelectorAll("[data-since]").forEach((node) => { node.textContent = since(node); });
  }

  // On a phone the panel sits below the chat, so a click that concerns it brings it into view.
  function reveal(node) {
    if (node && node.isConnected) node.scrollIntoView({ block: narrow() ? "start" : "nearest", behavior: "smooth" });
  }

  /* =========================================================== model picker */

  /* The server renders the picker, so it shows the default with JS off;
     /api/models refreshes it. The choice persists per browser. */
  const MODEL_KEY = "r3.controller.model.v1";
  const CUSTOM = "__custom__";
  const MODEL_ID = /^(openrouter|openai):[A-Za-z0-9][A-Za-z0-9._:/-]{0,120}$/;
  const PROVIDERS = { openrouter: "OpenRouter", openai: "OpenAI", typesafe: "TypeSafe" };
  const picker = $("chat-model");
  const custom = $("chat-model-custom");
  const ready = root.dataset.chatReady === "true";
  let defaultModel = root.dataset.defaultModel || "";
  const modelLabels = new Map(); // id -> label, for the header and the reply footnotes
  picker.querySelectorAll("option[data-label]").forEach((o) => modelLabels.set(o.value, o.dataset.label));

  function customId() {
    const raw = custom.value.trim();
    const id = /^(openrouter|openai):/.test(raw) ? raw : `openrouter:${raw}`;
    return raw && MODEL_ID.test(id) ? id : "";
  }

  // What a turn sends: null means the server's default; undefined means the custom id is not valid yet.
  const chosenModel = () => (picker.value === CUSTOM ? customId() || undefined
    : picker.value && picker.value !== defaultModel ? picker.value : null);

  function showModel() {
    const isCustom = picker.value === CUSTOM;
    custom.hidden = !isCustom;
    custom.setAttribute("aria-invalid", String(isCustom && Boolean(custom.value.trim()) && !customId()));
    $("chat-model-name").textContent = !ready ? "Setup required" : isCustom
      ? customId().replace(/^openrouter:/, "") || "Custom model" : modelLabels.get(picker.value) || picker.value;
  }

  function applyStoredModel() {
    const saved = local.get(MODEL_KEY) || "";
    const values = [...picker.options].map((o) => o.value);
    if (saved && saved !== CUSTOM && values.includes(saved)) picker.value = saved;
    else if (MODEL_ID.test(saved) && values.includes(CUSTOM)) {
      picker.value = CUSTOM;
      custom.value = saved.replace(/^openrouter:/, "");
    }
  }

  async function loadModels() {
    if (!ADMIN) return; // the server rendered this account's list; no custom ids, no per-provider duplicates
    let data;
    try { data = await fetchJSON("/api/models"); } catch { return; } // keep the server-rendered list
    if (!data || !Array.isArray(data.models) || !data.models.length) return;
    if (typeof data.default === "string" && data.default) defaultModel = data.default;
    const groups = new Map(); // provider -> models, in the server's order
    for (const m of data.models.filter((x) => x && typeof x.id === "string")) {
      const provider = typeof m.provider === "string" && m.provider ? m.provider : m.id.split(":")[0];
      groups.set(provider, [...(groups.get(provider) || []), m]);
    }
    modelLabels.clear();
    const options = [...groups].map(([provider, list]) => {
      const name = PROVIDERS[provider] || provider;
      return el("optgroup", { label: name }, list.map((m) => {
        const label = typeof m.label === "string" && m.label ? m.label : m.id;
        modelLabels.set(m.id, label);
        return el("option", { value: m.id, "data-label": label, text: label.replace(`${name} · `, "") });
      }));
    });
    const openrouter = !data.providers || data.providers.openrouter !== false;
    if (data.custom_allowed && openrouter) options.push(el("option", { value: CUSTOM, text: "Custom OpenRouter model…" }));
    picker.replaceChildren(...options);
    picker.value = defaultModel;
    if (picker.selectedIndex < 0) picker.selectedIndex = 0;
    applyStoredModel();
    showModel();
    controls();
  }

  // OpenRouter's catalogue (?live=1) suggests custom ids; fetched once, when "Custom" is first picked.
  let catalogue = null;
  function loadCatalogue() {
    catalogue = catalogue || fetchJSON("/api/models?live=1").then((data) => {
      const list = Array.isArray(data && data.catalogue) ? data.catalogue.filter((c) => c && typeof c.id === "string") : [];
      $("chat-model-catalogue").replaceChildren(...list.slice(0, 500).map((c) => el("option", {
        value: c.id.replace(/^openrouter:/, ""),
        label: `${c.name || c.id}${c.prompt_price === 0 && c.completion_price === 0 ? " · free" : ""}` })));
    }).catch(() => { catalogue = null; }); // try again the next time
  }

  picker.addEventListener("change", () => {
    showModel();
    if (picker.value === CUSTOM) { custom.focus(); loadCatalogue(); } else local.set(MODEL_KEY, picker.value);
    controls();
  });
  custom.addEventListener("input", () => { showModel(); controls(); });
  custom.addEventListener("change", () => { if (customId()) local.set(MODEL_KEY, customId()); });

  /* =============================================================== markdown */

  /* Models answer in Markdown. A small, safe subset is rebuilt as DOM nodes
     through el(), never parsed as HTML: paragraphs and line breaks, **bold**,
     *italic*, `code`, fenced code, lists (one nested level), headings as bold
     lines, http(s) links (bare URLs too), links to R3's own pages
     ("[Build your resume](/resume/build)", a job as "[Title at Company](/jobs/7)"),
     pipe tables whose rows all have the same number of cells, and "#13656" job
     references linking to the job. */
  const INLINE = new RegExp([
    /(`+)([^`\n]+?)\1/.source, // 1-2 inline code
    /\*\*(?=\S)(.+?)\*\*|(?<![\w.\/])__(?=\S)(.+?)__(?!\w)/.source, // 3-4 bold; __ never inside identifiers
    /(?<![\w*])\*(?=[^\s*])([^*\n]+?)\*(?!\w)|(?<!\w)_(?=\S)([^_\n]+?)_(?!\w)/.source, // 5-6 italic
    /\[([^\]\n]+)\]\(((?:[^()\s]|\([^()\s]*\))+)\)/.source, // 7-8 [text](url), one level of parens in the url
    /(https?:\/\/[^\s<>()]*[^\s<>().,;:!?'"*_])/.source, // 9 bare URL, without trailing punctuation
    /(?<![\w&/#])#(\d{3,9})\b/.source, // 10 job reference
  ].join("|"), "g");
  const LIST_ITEM = /^(\s*)([-*+]|\d{1,3}[.)])\s+(.*)$/;
  const TABLE_ROW = /^\s*\|.*\|\s*$/;
  const TABLE_RULE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
  const hostOf = (u) => { try { return new URL(u).host.replace(/^www\./, ""); } catch { return u; } };
  const outLink = (href, children) => el("a", { href, target: "_blank", rel: "noopener noreferrer", title: href }, children);
  // A portal page the reply links to: a same-site path only, joined to the base path and opened in this tab.
  // "//host" and "/\host" (which browsers read as another host), any scheme (javascript:, data:), whitespace,
  // backslashes and control characters never become a link.
  const pagePath = (p) => (typeof p === "string" && /^\/(?![/\\])[^\s\\\u0000-\u001f\u007f]*$/.test(p) ? BASE + p : "");

  function inline(text) {
    const out = [];
    let at = 0;
    for (const m of text.matchAll(INLINE)) { // matchAll copies the regex, so the recursion below is safe
      out.push(text.slice(at, m.index));
      at = m.index + m[0].length;
      if (m[2] !== undefined) out.push(el("code", { text: m[2] }));
      else if (m[3] !== undefined || m[4] !== undefined) out.push(el("strong", {}, inline(m[3] ?? m[4])));
      else if (m[5] !== undefined || m[6] !== undefined) out.push(el("em", {}, inline(m[5] ?? m[6])));
      else if (m[7] !== undefined) {
        const href = webUrl(m[8]);
        const page = href ? "" : pagePath(m[8]); // any other target keeps just its text
        if (href) out.push(outLink(href, [inline(m[7]), el("span", { class: "md-host", text: ` (${hostOf(href)})` })]));
        else out.push(page ? el("a", { class: "md-page", href: page }, inline(m[7])) : inline(m[7]));
      } else if (m[9] !== undefined) out.push(outLink(m[9], hostOf(m[9])));
      else out.push(el("a", { class: "md-job", href: jobUrl(m[10]), text: `#${m[10]}` }));
    }
    out.push(text.slice(at));
    return out;
  }

  function mdList(lines) {
    const indent = lines[0].match(/^\s*/)[0].length;
    const top = [];
    for (const line of lines) {
      const m = line.match(LIST_ITEM);
      const last = top[top.length - 1];
      if (!m) { if (last) (last.children[last.children.length - 1] || last).text += ` ${line.trim()}`; continue; }
      const item = { ordered: /\d/.test(m[2]), start: parseInt(m[2], 10), text: m[3], children: [] };
      if (m[1].length > indent && last) last.children.push(item); else top.push(item); // deeper levels flatten to one
    }
    const build = (items) => el(items[0].ordered ? "ol" : "ul", { class: "md-list",
      start: items[0].ordered && items[0].start !== 1 ? items[0].start : null },
    items.map((it) => el("li", {}, inline(it.text), it.children.length ? build(it.children) : null)));
    // A bullet list followed directly by a numbered one (or the reverse) is two lists.
    const groups = [];
    for (const item of top) {
      const group = groups[groups.length - 1];
      if (group && group[0].ordered === item.ordered) group.push(item); else groups.push([item]);
    }
    return groups.map(build);
  }

  function mdTable(rows) {
    const split = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    const body = rows.filter((r) => !TABLE_RULE.test(r)).map(split);
    const width = body.length > 1 ? body[0].length : 0;
    if (width < 2 || body.some((r) => r.length !== width)) return null; // ragged: leave it as text
    const head = TABLE_RULE.test(rows[1] || "") ? body.shift() : null;
    return el("div", { class: "md-table" }, el("table", {},
      head ? el("thead", {}, el("tr", {}, head.map((c) => el("th", { scope: "col" }, inline(c))))) : null,
      el("tbody", {}, body.map((r) => el("tr", {}, r.map((c) => el("td", {}, inline(c))))))));
  }

  function markdown(text) {
    const lines = text.replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let para = [];
    const flush = () => {
      if (para.length) out.push(el("p", {}, para.map((line, i) => [i ? el("br") : null, inline(line)])));
      para = [];
    };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const fence = line.match(/^\s*(```|~~~)/);
      const heading = line.match(/^\s{0,3}#{1,6}\s+(.*?)[\s#]*$/);
      if (fence) {
        flush();
        const code = [];
        while (++i < lines.length && !lines[i].trim().startsWith(fence[1])) code.push(lines[i]);
        out.push(el("pre", { class: "md-code" }, el("code", { text: code.join("\n") })));
      } else if (!line.trim()) {
        flush();
      } else if (heading) {
        flush();
        out.push(el("p", { class: "md-heading" }, el("strong", {}, inline(heading[1]))));
      } else if (TABLE_ROW.test(line)) {
        const rows = [];
        while (i < lines.length && TABLE_ROW.test(lines[i])) rows.push(lines[i++]);
        i -= 1;
        const table = mdTable(rows);
        if (table) { flush(); out.push(table); } else para.push(...rows);
      } else if (LIST_ITEM.test(line)) {
        flush();
        const items = [];
        while (i < lines.length && (LIST_ITEM.test(lines[i]) || /^\s{2,}\S/.test(lines[i]))) items.push(lines[i++]);
        i -= 1;
        out.push(...mdList(items));
      } else {
        para.push(line);
      }
    }
    flush();
    return out;
  }

  /* While a reply streams the whole buffer is re-rendered at most once a
     frame (cheap at chat sizes); the final text is rendered at once. */
  function paint(v, text, now = false) {
    v.source = text;
    if (v.frame) cancelAnimationFrame(v.frame);
    v.frame = 0;
    const draw = () => {
      v.frame = 0;
      const follow = nearBottom();
      v.body.replaceChildren(...markdown(v.source));
      if (follow) toBottom();
    };
    if (now) draw(); else v.frame = requestAnimationFrame(draw);
  }

  /* =================================================================== chat */

  const CHAT_KEY = ownKey("r3.controller.chat.v1");
  const chat = {
    input: $("chat-input"), send: $("chat-send"), stop: $("chat-stop"), form: $("chat-form"), log: $("chat-messages"),
    welcome: $("chat-welcome"), scroll: $("chat-scroll"), error: $("chat-error"), context: $("chat-context"),
    advice: $("chat-advice"), fresh: $("chat-new"),
  };
  let history = []; // finished exchanges: [{role, content, meta?}], always user/assistant pairs
  let pending = null; // {controller, turn} while a reply streams

  const validMeta = (m) => m === undefined || (m && typeof m === "object" && !Array.isArray(m) &&
    Object.values(m).every((v) => typeof v === "string" && v.length <= 300));
  const validHistory = (list) => Array.isArray(list) && list.length <= 40 && list.length % 2 === 0 &&
    list.every((m, i) => m && m.role === (i % 2 ? "assistant" : "user") && typeof m.content === "string" &&
      m.content.length > 0 && m.content.length <= 16000 && validMeta(m.meta));
  const saveHistory = () => session.set(CHAT_KEY, JSON.stringify(history));
  const syncWelcome = () => { chat.welcome.hidden = chat.log.childElementCount > 0; };
  const nearBottom = () => chat.scroll.scrollHeight - chat.scroll.scrollTop - chat.scroll.clientHeight < 90;
  const toBottom = () => { chat.scroll.scrollTop = chat.scroll.scrollHeight; };
  const showError = (message) => { chat.error.textContent = message; chat.error.hidden = !message; };

  const addUser = (content) => chat.log.appendChild(el("article", { class: "msg user" },
    el("div", { class: "msg-label", text: "You" }), el("div", { class: "msg-body", text: content })));

  /* An assistant message is built live: a route badge, tool and action chips,
     a thinking indicator until the first token, the streamed text (a single
     text node, appended to) and a footnote with the model and the tokens. */
  function addAssistant(meta = {}) {
    const v = {
      route: el("span", { class: `msg-route ${meta.kind || ""}`, text: meta.route, hidden: !meta.route }),
      chips: el("div", { class: "msg-chips", hidden: true }),
      label: el("span", { text: "Reading your message…" }),
      body: el("div", { class: "msg-body md" }),
      foot: el("p", { class: "msg-foot", text: meta.foot, hidden: !meta.foot }),
    };
    v.thinking = el("div", { class: "msg-thinking", hidden: true },
      el("span", { class: "dots", "aria-hidden": "true" }, el("i"), el("i"), el("i")), v.label);
    v.article = chat.log.appendChild(el("article", { class: "msg assistant" }, el("div", { class: "msg-label" }, "R3", v.route),
      v.chips, v.thinking, v.body, v.foot));
    return v;
  }

  const addChip = (v, node) => { v.chips.hidden = false; return v.chips.appendChild(node); };

  // Tool names are the advisor's own; the chips say what happened in plain words.
  const TOOL_WORDS = [
    [/search|list_jobs|find/, "searching jobs", "searched jobs"], [/overview|summary|snapshot|stats/, "reading overview", "read overview"],
    [/job/, "reading a job", "read a job"], [/run/, "checking runs", "checked runs"],
  ];
  function toolWords(name) {
    const plain = String(name || "tool").toLowerCase().replace(/[_-]+/g, " ").trim();
    const hit = TOOL_WORDS.find(([re]) => re.test(String(name || "").toLowerCase()));
    return hit ? hit.slice(1) : [`using ${plain}`, `used ${plain}`];
  }

  function onChatEvent(v, turn, type, data) {
    if (type === "route" && data.decision && typeof data.decision.intent === "string") {
      const d = data.decision;
      const via = typeof d.router === "string" && d.router ? ` · via ${d.router}` : "";
      const sure = typeof d.confidence === "number" ? ` · ${Math.round(d.confidence * 100)}%` : "";
      turn.kind = d.intent === "advice" ? "advice" : "workflow";
      turn.route = d.intent === "advice" ? "Advice" : `Workflow: ${d.intent}${via}${sure}`;
      Object.assign(v.route, { textContent: turn.route, className: `msg-route ${turn.kind}`, hidden: false,
        title: typeof d.reason === "string" ? d.reason : "" });
      v.label.textContent = d.intent === "advice" ? "Thinking…" : `Preparing the ${d.intent} workflow…`;
    } else if (type === "action" && data.action && typeof data.action === "object") {
      // The person described their background: once the reply is in, the page makes the resume.
      if (data.action.kind === "resume_text" && typeof data.action.text === "string") turn.background = data.action.text;
      const a = panelAction(data.action);
      if (a) {
        const chip = addChip(v, el("button", { class: `chip action ${a.tone || ""}`, type: "button", text: `${a.label} →`,
          on: { click: a.go } }));
        if (a.bind) a.bind(chip);
      }
    } else if (type === "tool_call") {
      const [doing] = toolWords(data.name);
      addChip(v, el("span", { class: "chip tool pending", "data-tool": String(data.name), text: `${doing}…` }));
      v.label.textContent = `${cap(doing)}…`;
    } else if (type === "tool_result") {
      const done = toolWords(data.name)[1];
      const summary = typeof data.summary === "string" ? data.summary.trim() : "";
      const node = [...v.chips.querySelectorAll(".tool.pending")].reverse().find((c) => c.dataset.tool === String(data.name))
        || addChip(v, el("span"));
      Object.assign(node, { className: "chip tool done", title: summary,
        textContent: summary && summary.length <= 40 ? `${done} · ${summary}` : done });
      v.label.textContent = "Thinking…";
    } else if (type === "delta" && typeof data.text === "string" && data.text) {
      v.thinking.hidden = true;
      turn.text += data.text;
      paint(v, turn.text); // follows the bottom only if the reader has not scrolled up
    } else if (type === "final") {
      turn.final = typeof data.text === "string" ? data.text : turn.text;
      paint(v, turn.final, true);
      v.thinking.hidden = true;
      const u = data.usage || {};
      const model = typeof data.model === "string" && data.model ? modelLabels.get(data.model) || data.model : "";
      const tokens = u.input_tokens || u.output_tokens ? `${num(u.input_tokens || 0)} in · ${num(u.output_tokens || 0)} out tokens` : "";
      turn.foot = [model, tokens].filter(Boolean).join(" · ");
      Object.assign(v.foot, { textContent: turn.foot, hidden: !turn.foot });
    } else if (type === "error") {
      turn.error = { message: typeof data.message === "string" && data.message ? data.message : "R3 could not answer.",
        retryable: data.retryable !== false };
    } // "done" ends the stream; unknown event types are ignored
  }

  /* Server-Sent Events over fetch, because EventSource cannot POST. A frame
     ends at a blank line and may arrive split across chunks, so the tail of
     each chunk waits in `buffer` until the rest of its frame arrives. */
  async function readEvents(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const drain = (end) => {
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = end ? "" : frames.pop();
      for (const frame of frames) {
        let type = "";
        const data = [];
        for (const line of frame.split(/\r?\n/)) {
          if (line.startsWith("event:")) type = line.slice(6).trim();
          else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
        }
        if (!data.length) continue; // comments and keep-alives
        let payload;
        try { payload = JSON.parse(data.join("\n")); } catch { continue; }
        if (payload && typeof payload === "object") onEvent(type || payload.type || "", payload);
      }
    };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      drain(false);
    }
    buffer += decoder.decode();
    drain(true);
  }

  async function sendTurn(content) {
    const model = chosenModel();
    if (pending || !ready || !content) return;
    if (model === undefined) {
      showError("Enter an OpenRouter model id such as vendor/model, or pick a listed model.");
      custom.focus();
      return;
    }
    // A failed exchange never reached the history; it goes before the retry or the next message.
    chat.log.querySelectorAll("[data-failed]").forEach((n) => n.remove());
    showError("");
    const user = { role: "user", content };
    const userNode = addUser(content);
    const v = addAssistant();
    v.thinking.hidden = false;
    v.article.setAttribute("aria-busy", "true"); // screen readers wait for the finished reply
    syncWelcome();
    toBottom();

    // At most the last 10 exchanges plus the new message, trimmed further to fit 80,000 characters.
    const messages = [...history.slice(-20), user].map((m) => ({ role: m.role, content: m.content }));
    while (messages.length > 1 && messages.reduce((n, m) => n + m.content.length, 0) > 80000) messages.splice(0, 2);

    const turn = { text: "", final: null, error: null, route: "", kind: "", foot: "", stopped: false, timedOut: false };
    const controller = new AbortController();
    pending = { controller, turn };
    // The proxy in front of the hosted copy cuts streams at ~110 s; give up just before it does.
    const timer = setTimeout(() => { turn.timedOut = true; controller.abort(); }, 100000);
    controls();
    try {
      const res = await fetch(api("/api/chat/stream"), {
        method: "POST", signal: controller.signal,
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ messages, model, include_context: chat.context.checked, mode: chat.advice.checked ? "advice" : "auto" }),
      });
      if (!res.ok || !res.body) throw new ApiError(detailOf(await res.json().catch(() => null), res.status), res.status);
      await readEvents(res, (type, data) => onChatEvent(v, turn, type, data));
      if (turn.final === null && !turn.error) turn.error = { message: "The reply was cut off before it finished.", retryable: true };
    } catch (err) {
      if (turn.timedOut) turn.error = { message: "The reply took longer than 100 seconds and was stopped.", retryable: true };
      else if (err && err.name === "AbortError") turn.stopped = true;
      else {
        const message = err instanceof ApiError ? err.message : "Couldn’t reach R3. Check your connection and try again.";
        turn.error = { message, retryable: true };
      }
    } finally {
      clearTimeout(timer);
      pending = null;
    }
    finishTurn(user, userNode, v, turn);
    controls();
    if (turn.background && turn.final !== null) resumeFromText(turn.background);
  }

  function finishTurn(user, userNode, v, turn) {
    v.thinking.hidden = true;
    v.article.removeAttribute("aria-busy");
    if (turn.final === null) paint(v, turn.text, true); // a stopped or failed reply keeps what it said
    const reply = turn.final !== null ? turn.final : turn.text;
    if (turn.final !== null || (turn.stopped && reply.trim())) {
      // A finished reply, or a stopped one that already said something, joins the conversation.
      if (turn.final === null) Object.assign(v.foot, { textContent: (turn.foot = "Stopped"), hidden: false });
      const meta = turn.route ? { route: turn.route, kind: turn.kind } : {};
      if (turn.foot) meta.foot = turn.foot;
      history = [...history, user, { role: "assistant", content: reply.trim() ? reply.slice(0, 16000) : "(no reply)", meta }].slice(-40);
      saveHistory();
      return;
    }
    // Nothing usable came back: the exchange stays on screen, marked failed, and the text returns to the composer.
    const retry = turn.stopped || turn.error.retryable;
    userNode.dataset.failed = v.article.dataset.failed = "";
    v.article.append(el("p", { class: "msg-error", role: "alert" }, turn.stopped ? "Stopped before R3 answered." : turn.error.message,
      retry ? el("button", { class: "btn tiny", type: "button", text: "Retry", on: { click: () => {
        if (chat.input.value.trim() === user.content) { chat.input.value = ""; autosize(); }
        sendTurn(user.content);
      } } }) : null));
    if (retry && !chat.input.value.trim()) { chat.input.value = user.content; autosize(); }
  }

  function controls() {
    const busy = Boolean(pending);
    // While a resume's follow-ups are open, Send answers them (no model needed).
    const answering = Boolean(resume.mode);
    chat.send.hidden = busy;
    chat.stop.hidden = !busy;
    chat.input.disabled = !ready && !answering;
    chat.input.placeholder = answering ? "Answer R3’s question about your resume…" : "Message R3…";
    chat.send.disabled = !chat.input.value.trim() || resume.busy
      || (!answering && (!ready || busy || chosenModel() === undefined));
    chat.fresh.disabled = chat.context.disabled = chat.advice.disabled = custom.disabled = busy || resume.busy;
    up.button.disabled = busy || resume.busy;
    picker.disabled = busy || !ready || picker.options.length === 0;
    chat.welcome.querySelectorAll("[data-prompt]").forEach((b) => { b.disabled = !ready || busy; });
  }

  function autosize() {
    chat.input.style.height = "auto";
    chat.input.style.height = `${Math.min(chat.input.scrollHeight, 180)}px`;
    controls();
  }

  // Starters and "prompt" suggestions fill the composer; the operator still presses Send.
  function usePrompt(text) {
    if (!ready) return;
    chat.input.value = text;
    autosize();
    chat.input.focus();
    chat.input.setSelectionRange(text.length, text.length);
    if (narrow()) chat.input.scrollIntoView({ block: "center", behavior: "smooth" });
  }

  chat.input.addEventListener("input", autosize);
  chat.input.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
    e.preventDefault();
    if (!chat.send.disabled) chat.form.requestSubmit();
  });
  chat.form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = chat.input.value.trim();
    if (resume.mode) {
      if (text && !resume.busy) { chat.input.value = ""; autosize(); sendAnswer(text); }
      return;
    }
    if (!text || pending || !ready) return;
    if (chosenModel() !== undefined) { chat.input.value = ""; autosize(); } // an invalid model keeps the text
    sendTurn(text);
  });
  chat.stop.addEventListener("click", () => { if (pending) { pending.turn.stopped = true; pending.controller.abort(); } });
  chat.fresh.addEventListener("click", () => {
    if (pending || resume.busy) return;
    resume.mode = null;
    session.remove(RESUME_KEY);
    history = [];
    saveHistory();
    chat.log.replaceChildren();
    showError("");
    chat.input.value = "";
    autosize();
    syncWelcome();
    chat.input.focus();
  });
  chat.welcome.querySelectorAll("[data-prompt]").forEach((b) => b.addEventListener("click", () => usePrompt(b.dataset.prompt)));

  function restoreHistory() {
    let stored;
    try { stored = JSON.parse(session.get(CHAT_KEY) || "[]"); } catch { return; } // damaged: start fresh
    if (!validHistory(stored) || !stored.length) return;
    history = stored;
    for (const m of history) {
      if (m.role === "user") addUser(m.content);
      else paint(addAssistant(m.meta || {}), m.content, true);
    }
    syncWelcome();
    toBottom();
  }

  /* ========================================================= resume upload */

  /* The Assistant takes a resume file: picked with "Upload resume" or dropped
     on the chat, staged as a chip until the seeker presses Upload, then saved,
     read into R3's structure and followed up here (role dates, a profile
     link) until it is complete. The checks here are a convenience; the
     server applies every rule again. */
  const RESUME_KEY = ownKey("r3.controller.resume.v1");
  const RESUME_TYPES = [".pdf", ".docx", ".txt", ".md"];
  const MAX_UPLOAD = Number(root.dataset.maxUpload) || 5 * 1024 * 1024;
  const up = { button: $("resume-upload"), input: $("resume-file"), chip: $("resume-chip"), drop: $("resume-drop"),
    chat: $("r3-chat") };
  const resume = { staged: null, mode: null, busy: false }; // mode: {id} while follow-ups are open

  const sizeText = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
  const extOf = (name) => { const m = /\.[^.]+$/.exec(String(name || "").toLowerCase()); return m ? m[0] : ""; };

  function resumeProblem(file) {
    if (!RESUME_TYPES.includes(extOf(file.name))) {
      return `“${file.name}” is not a file R3 can read as a resume. Use a PDF, DOCX, TXT or MD file.`;
    }
    if (file.size > MAX_UPLOAD) return `“${file.name}” is ${sizeText(file.size)}; the limit is ${sizeText(MAX_UPLOAD)}.`;
    if (file.size === 0) return `“${file.name}” is empty.`;
    return "";
  }

  function stageResume(files) {
    if (resume.busy) return;
    const list = [...(files || [])];
    if (!list.length) return;
    if (list.length > 1) { showError("Drop one resume at a time."); return; }
    const problem = resumeProblem(list[0]);
    if (problem) { showError(problem); return; }
    showError("");
    resume.staged = list[0];
    renderChip(true);
  }

  function renderChip(focus = false) {
    const f = resume.staged;
    up.chip.hidden = !f;
    if (!f) { up.chip.replaceChildren(); controls(); return; }
    const send = el("button", { class: "btn go tiny", type: "button", "aria-label": `Upload ${f.name}`, text: "Upload",
      on: { click: uploadResume } });
    up.chip.replaceChildren(el("span", { "aria-hidden": "true", text: "📄" }),
      el("span", { class: "name", text: f.name }), el("span", { class: "size", text: sizeText(f.size) }),
      el("span", { class: "actions" }, send, el("button", { class: "btn ghost tiny", type: "button",
        "aria-label": `Remove ${f.name}`, text: "Remove",
        on: { click: () => { resume.staged = null; renderChip(); up.button.focus(); } } })));
    controls();
    if (focus) send.focus();
  }

  // An assistant bubble tagged "Resume", built from text and nodes (never parsed as HTML).
  function addResumeNote(...children) {
    const v = addAssistant({ route: "Resume", kind: "workflow" });
    v.body.append(...children.filter(Boolean).map((c) => (typeof c === "string" ? el("p", { text: c }) : c)));
    syncWelcome();
    toBottom();
    return v;
  }

  // The exchange joins the chat history, so the assistant knows about the resume.
  function remember(userText, assistantText) {
    history = [...history, { role: "user", content: userText.slice(0, 16000) },
      { role: "assistant", content: assistantText.slice(0, 16000), meta: { route: "Resume", kind: "workflow" } }].slice(-40);
    saveHistory();
  }

  const failText = (err, fallback) => (err instanceof ApiError ? err.message : fallback);

  async function uploadResume() {
    const file = resume.staged;
    if (!file || resume.busy) return;
    resume.busy = true;
    controls();
    showError("");
    addUser(`📎 ${file.name}`);
    const v = addResumeNote();
    v.thinking.hidden = false;
    v.label.textContent = "Uploading…";
    v.article.setAttribute("aria-busy", "true");
    const form = new FormData();
    form.append("file", file, file.name);
    let saved;
    try {
      let res;
      try {
        res = await fetch(api("/api/resume/upload"), { method: "POST", body: form, headers: { Accept: "application/json" } });
      } catch {
        throw new ApiError("Couldn’t reach R3. Check your connection and try again.", 0);
      }
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new ApiError(detailOf(data, res.status), res.status);
      saved = data;
    } catch (err) {
      v.thinking.hidden = true;
      v.article.removeAttribute("aria-busy");
      v.body.append(el("p", { class: "msg-error", role: "alert", text: failText(err, "The upload failed. Try again.") }));
      resume.busy = false;
      controls(); // the chip stays: Upload again, or Remove
      return;
    }
    resume.staged = null;
    renderChip();
    const first = saved.primary ? " It is now your primary resume, so your job matches use it." : "";
    v.body.append(el("p", { text: `Uploaded “${saved.title}”.${first}` }));
    const card = await readInto(saved.resume_id, v, saved.title);
    resume.busy = false;
    controls();
    if (card) afterRead(card);
  }

  // A resume from what the person wrote in the chat: saved and read in one call, then the same card
  // and follow-ups as an upload. Nothing is saved when R3 cannot read it; the reason says what to do.
  async function resumeFromText(text) {
    if (resume.busy || resume.mode) return;
    resume.busy = true;
    controls();
    const v = addResumeNote();
    v.thinking.hidden = false;
    v.label.textContent = "Reading what you wrote… this can take up to a minute.";
    v.article.setAttribute("aria-busy", "true");
    let card = null;
    try {
      card = await fetchJSON("/api/resume/from-text", { method: "POST", body: { text } });
      const first = card.primary ? " It is now your primary resume, so your job matches use it." : "";
      v.body.append(el("p", { text: `Saved “${card.title}”.${first}` }), cardNode(card));
      if (card.message) v.body.append(el("p", { class: "note", text: card.message }));
      remember("Make my resume from what I wrote.", summaryText(card.title, card).replace(/^Uploaded/, "Made")
        + (card.done ? "" : ` Asked next: ${card.question}`));
    } catch (err) {
      const again = el("button", { class: "btn tiny", type: "button", text: "Try again", on: { click: () => {
        again.disabled = true;
        resumeFromText(text);
      } } });
      v.body.append(el("div", { class: "msg-error md", role: "alert" },
        ...markdown(failText(err, "R3 couldn’t make the resume.")), again));
    } finally {
      v.thinking.hidden = true;
      v.article.removeAttribute("aria-busy");
      resume.busy = false;
      controls();
    }
    if (card) afterRead(card);
  }

  // Reads a saved resume into R3's structure and shows the card in ``v``.
  async function readInto(id, v, title) {
    v.thinking.hidden = false;
    v.label.textContent = "Reading it… this can take a minute or two.";
    v.article.setAttribute("aria-busy", "true");
    try {
      const card = await fetchJSON(`/api/resume/${encodeURIComponent(id)}/read`, { method: "POST" });
      v.body.append(cardNode(card));
      if (card.message) v.body.append(el("p", { class: "note", text: card.message }));
      remember(`I uploaded my resume (“${title}”).`, summaryText(title, card) + (card.done ? "" : ` Asked next: ${card.question}`));
      return card;
    } catch (err) {
      const again = el("button", { class: "btn tiny", type: "button", text: "Read again", on: { click: async () => {
        again.disabled = true;
        resume.busy = true;
        controls();
        const card = await readInto(id, v, title);
        resume.busy = false;
        controls();
        if (card) afterRead(card);
      } } });
      v.body.append(el("p", { class: "msg-error", role: "alert" },
        failText(err, "Reading was interrupted."), " Your resume is saved. ", again));
      return null;
    } finally {
      v.thinking.hidden = true;
      v.article.removeAttribute("aria-busy");
    }
  }

  function cardNode(c) {
    const rows = [];
    const add = (term, value) => { if (value) rows.push(el("dt", { text: term }), el("dd", {}, value)); };
    add("Name", [c.name, c.headline].filter(Boolean).join(" · "));
    add("Contact", [c.email, c.phone, c.location].filter(Boolean).join(" · "));
    add("Links", (c.links || []).join(", ") || "none yet");
    if ((c.roles || []).length) {
      add("Experience", el("ul", { class: "roles" }, c.roles.map((r) => el("li", {
        text: `${[r.title, r.company].filter(Boolean).join(" — ") || "A role"} · ${[r.start, r.end].filter(Boolean).join(" – ") || "no dates"}` }))));
    }
    if (c.skills_count) {
      add("Skills", `${plural(c.skills_count, "skill")}: ${(c.skills || []).slice(0, 8).join(", ")}${c.skills_count > 8 ? "…" : ""}`);
    }
    add("Education", (c.education || []).join("; "));
    return el("section", { class: "resume-card", "aria-label": "What R3 read from your resume" },
      el("h3", { text: "Here’s what I read" }), el("dl", {}, rows),
      el("ul", { class: "resume-checks", "aria-label": "Resume checklist" }, (c.checklist || []).map((i) =>
        el("li", { class: `chip ${i.ok ? "ok" : "warn"}`, text: `${i.ok ? "✓" : "missing:"} ${i.section}` }))),
      c.read && !c.faithful ? el("p", { class: "note", text: "I kept your original text: R3’s clean layout would have lost part of it." }) : null,
      c.is_resume === false ? el("p", { class: "note", text: "This may not be a resume. Check it on the Resume page." }) : null,
      el("p", {}, el("a", { href: portalUrl(c.url), text: "Open this resume" })));
  }

  function summaryText(title, c) {
    const roles = (c.roles || []).map((r) => `${[r.title, r.company].filter(Boolean).join(" at ") || "a role"} (${[r.start, r.end].filter(Boolean).join("–") || "no dates"})`).join("; ");
    const missing = (c.checklist || []).filter((i) => !i.ok).map((i) => i.section).join(", ");
    return `Uploaded “${title}” and read it: ${c.name || "no name found"}; ${roles || "no roles found"}; `
      + `${plural(c.skills_count || 0, "skill")}.${missing ? ` Still missing: ${missing}.` : ""}`;
  }

  function afterRead(card) {
    if (card.done) finishResume(card);
    else enterResumeMode(card.resume_id, card);
  }

  function enterResumeMode(id, state) {
    resume.mode = { id: String(id) };
    session.set(RESUME_KEY, JSON.stringify(resume.mode));
    askNext(state);
    controls();
    chat.input.focus();
  }

  function askNext(state) {
    const said = state.note || (state.changed ? `Got it: ${state.changed}.` : "");
    addResumeNote(said, state.question, el("p", {}, el("button", { class: "chip action", type: "button",
      text: "Skip for now", on: { click: leaveResumeMode } })));
  }

  function leaveResumeMode() {
    if (!resume.mode || resume.busy) return;
    const id = resume.mode.id;
    resume.mode = null;
    session.remove(RESUME_KEY);
    addResumeNote("OK. Your resume is saved; what is still missing stays listed on it, and you can finish it any time.",
      el("p", {}, el("a", { href: portalUrl(`/resume/${encodeURIComponent(id)}`), text: "Open the resume" })));
    remember("Skip the resume questions for now.", "The resume is saved; some required details are still open.");
    controls();
  }

  function finishResume(state) {
    resume.mode = null;
    session.remove(RESUME_KEY);
    addResumeNote(state.changed ? `Got it: ${state.changed}.` : "",
      "Your resume is complete. Next: open Jobs, where your best matches are at the top, and press “I applied” on the ones you apply to.",
      el("p", {}, el("a", { class: "btn go tiny", href: portalUrl("/jobs"), text: "Open Jobs" })));
    controls();
  }

  async function sendAnswer(text) {
    if (!resume.mode || resume.busy) return;
    resume.busy = true;
    controls();
    showError("");
    const userNode = addUser(text);
    syncWelcome();
    toBottom();
    try {
      const state = await fetchJSON(`/api/resume/${encodeURIComponent(resume.mode.id)}/answer`,
        { method: "POST", body: { answer: text } });
      if (state.off_topic && !ready) state.note = "Without an AI model I can only take the answer itself.";
      else if (state.off_topic) {
        // A question back ("what do you need?") goes to the assistant, which
        // has the pending question in its history; the follow-up stays open.
        resume.busy = false;
        userNode.remove();
        controls();
        sendTurn(text);
        return;
      }
      remember(text, [state.note || (state.changed ? `Got it: ${state.changed}.` : ""),
        state.done ? "The resume is complete." : state.question].filter(Boolean).join(" "));
      resume.busy = false;
      if (state.done) finishResume(state);
      else askNext(state);
    } catch (err) {
      resume.busy = false;
      userNode.remove();
      showError(`${failText(err, "Couldn’t reach R3.")} Your answer is back in the box; try again.`);
      if (!chat.input.value.trim()) { chat.input.value = text; autosize(); }
    }
    controls();
  }

  // Back after a reload, or from the Resume page ("Finish in the Assistant").
  async function resumeFollowups() {
    const param = new URLSearchParams(location.search).get("resume") || "";
    let stored = null;
    try { stored = JSON.parse(session.get(RESUME_KEY) || "null"); } catch { stored = null; }
    const id = /^\d+$/.test(param) ? param : stored && /^\d+$/.test(String(stored.id)) ? String(stored.id) : "";
    if (!id) return;
    try {
      const state = await fetchJSON(`/api/resume/${encodeURIComponent(id)}/followup`);
      if (!state.read) {
        session.remove(RESUME_KEY);
        if (!param) return;
        resume.busy = true;
        controls();
        const card = await readInto(id, addResumeNote(`Reading “${state.title}”…`), state.title);
        resume.busy = false;
        controls();
        if (card) afterRead(card);
        return;
      }
      if (state.done) {
        session.remove(RESUME_KEY);
        if (param) addResumeNote(`“${state.title}” has nothing left to fill in.`);
        return;
      }
      enterResumeMode(state.resume_id, state);
    } catch {
      session.remove(RESUME_KEY);
    }
  }

  const hasFiles = (e) => Boolean(e.dataTransfer) && [...(e.dataTransfer.types || [])].includes("Files");
  let dragDepth = 0;
  up.chat.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth += 1;
    up.drop.hidden = resume.busy;
  });
  up.chat.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = resume.busy ? "none" : "copy";
  });
  up.chat.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) up.drop.hidden = true;
  });
  up.chat.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.stopPropagation();
    dragDepth = 0;
    up.drop.hidden = true;
    stageResume(e.dataTransfer.files);
  });
  // A file dropped anywhere else must not make the browser open it (and wipe the chat).
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    showError("Drop the file on the chat to upload it.");
  });
  up.button.addEventListener("click", () => { if (!resume.busy) up.input.click(); });
  up.input.addEventListener("change", () => { stageResume(up.input.files); up.input.value = ""; });

  /* ============================================================ panel: tabs */

  const TAB_KEY = ownKey("r3.controller.tab.v1");
  const tabs = [...document.querySelectorAll("#r3-panel [role=tab]")];
  const panelScroll = $("panel-scroll");

  function selectTab(name, focus = false) {
    const target = tabs.find((t) => t.dataset.tab === name) || tabs[0];
    for (const tab of tabs) {
      tab.setAttribute("aria-selected", String(tab === target));
      tab.tabIndex = tab === target ? 0 : -1;
      $(tab.getAttribute("aria-controls")).hidden = tab !== target;
    }
    if (focus) target.focus();
    session.set(TAB_KEY, target.dataset.tab);
    // Jobs and statistics load the first time their tab opens, not with the page.
    if (target.dataset.tab === "jobs" && !jobsLoaded) loadJobs();
    if (target.dataset.tab === "stats" && !statsLoaded) loadStats();
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
    tab.addEventListener("keydown", (e) => { // arrows, Home and End move between tabs (the ARIA tabs pattern)
      const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
      if (to === undefined) return;
      e.preventDefault();
      selectTab(tabs[(to + tabs.length) % tabs.length].dataset.tab, true);
    });
  });

  /* ======================================================= panel: notices */

  function notice({ tone = "neutral", title, text, hint }) {
    const node = el("section", { class: `notice ${tone}`, role: tone === "warn" ? "alert" : "status" },
      el("header", {}, el("h3", { text: title }),
        el("button", { class: "notice-close", type: "button", "aria-label": "Dismiss", text: "×", on: { click: () => node.remove() } })),
      text ? el("p", { text }) : null, hint ? el("p", { class: "notice-hint", text: hint }) : null);
    $("panel-notices").prepend(node);
    return node;
  }

  /* ============================================================ panel: runs */

  const KINDS = { stats: ["▦", "Stats"] };
  const kindLabel = (k) => (KINDS[k] ? KINDS[k][1] : cap(k) || "Run");
  const OPEN = new Set(["queued", "running"]);
  const runMap = new Map(); // id -> WorkflowRun
  const runNodes = new Map(); // id -> {node, sig}: a poll re-renders only the runs that changed
  const flashUntil = new Map(); // id -> ms: runs that just arrived are highlighted for a moment
  let runsOn = $("r3-panel").dataset.runsConfigured === "true";
  let panelTimer = 0;

  async function startRun(request) {
    const run = await fetchJSON("/api/runs", { method: "POST", body: request });
    addRuns([run], run.id);
    showRun(run.id);
    return run;
  }

  function addRuns(list, highlight) {
    let finished = false;
    for (const run of list) {
      if (!run || typeof run.id !== "string" || !run.id || typeof run.kind !== "string") continue;
      const before = runMap.get(run.id);
      runMap.set(run.id, run);
      if (!before || !OPEN.has(before.status) || OPEN.has(run.status)) continue;
      finished = true;
      $("runs-live").textContent = `${kindLabel(run.kind)} run ${String(run.status).replace("_", " ")}.`;
      // A finished stats run carries the freshest snapshot there is.
      if (run.kind === "stats" && run.result && typeof run.result.total_jobs === "number") renderStats(run.result, run);
    }
    if (highlight) flashUntil.set(highlight, Date.now() + 4000);
    renderRuns();
    if (finished) { clearTimeout(panelTimer); panelTimer = setTimeout(loadPanel, 1500); } // the numbers moved
    if (!polling) schedulePoll(); // a new open run switches polling to every 3 s
  }

  function renderRuns() {
    const list = [...runMap.values()].sort((a, b) => (stamp(b.started_at) || 0) - (stamp(a.started_at) || 0)).slice(0, 30);
    const box = $("runs-list");
    const keep = new Set(list.map((r) => r.id));
    for (const [id, entry] of runNodes) if (!keep.has(id)) { entry.node.remove(); runNodes.delete(id); }
    list.forEach((run, i) => {
      const sig = JSON.stringify(run);
      let entry = runNodes.get(run.id);
      if (!entry || entry.sig !== sig) {
        const node = runCard(run);
        if (entry) { // keep keyboard focus on the same control when a card is redrawn under it
          const key = entry.node.contains(document.activeElement) ? document.activeElement.dataset.focusKey : "";
          entry.node.replaceWith(node);
          if (key) node.querySelector(`[data-focus-key="${key}"]`)?.focus();
        }
        runNodes.set(run.id, (entry = { node, sig }));
      }
      if (box.children[i] !== entry.node) box.insertBefore(entry.node, box.children[i] || null);
    });
    $("runs-empty").hidden = list.length > 0;
    const open = list.filter((r) => OPEN.has(r.status)).length;
    Object.assign($("runs-badge"), { textContent: open ? String(open) : "", hidden: !open });
  }

  function runCard(run) {
    const [icon, label] = KINDS[run.kind] || ["•", cap(run.kind)];
    const open = OPEN.has(run.status);
    const started = stamp(run.started_at);
    const closed = stamp(run.closed_at);
    const took = closed !== null && started !== null ? `took ${span((closed - started) / 1000)}` : "";
    const model = run.model ? run.model.replace(/^[a-z]+:/, "") : "";
    const who = ADMIN ? [run.requested_by, run.source ? `via ${run.source}` : "", model].filter(Boolean) : [];
    const verb = run.status === "queued" ? "queued" : "running";
    const clock = open ? el("span", { "data-since": run.started_at, "data-verb": verb }) : null;
    if (clock) clock.textContent = since(clock);
    const temporal = webUrl(run.url);
    const flash = (flashUntil.get(run.id) || 0) > Date.now();
    return el("li", { class: `run ${open ? "open" : ""} ${flash ? "is-flash" : ""}`, "data-run": run.id },
      el("header", { class: "run-head" },
        el("span", { class: "run-icon", "aria-hidden": "true", text: icon }), el("strong", { class: "run-kind", text: label }),
        el("span", { class: `run-status ${run.status}`, text: String(run.status).replace("_", " ") }), agoEl(run.started_at, "run-time")),
      el("p", { class: "run-meta" }, clock || took, who.length ? `${clock || took ? " · " : ""}${who.join(" · ")}` : null),
      run.note ? el("p", { class: "run-note", text: `“${run.note}”` }) : null,
      open ? runProgress(run.progress) : runResult(run),
      run.error ? el("p", { class: "run-error", text: run.error }) : null,
      el("footer", { class: "run-foot" },
        ADMIN ? el("a", { class: "run-id", href: `#run=${encodeURIComponent(run.id)}`, title: "Link to this run", text: run.id }) : null,
        ADMIN && temporal ? el("a", { href: temporal, target: "_blank", rel: "noopener", text: "Open in Temporal ↗" }) : null,
        open ? el("button", { class: "btn tiny", type: "button", "data-cancel": run.id, "data-focus-key": "cancel",
          text: "Cancel" }) : null));
  }

  function runProgress(p) {
    const total = p && p.total > 0 ? p.total : 0;
    const fill = el("i");
    if (total) fill.style.width = `${Math.min(100, Math.round((p.done / total) * 100))}%`;
    const words = p && (p.message || p.stage) ? p.message || p.stage : "Waiting for a worker…";
    return el("div", { class: "run-progress" },
      el("div", { class: `run-bar ${total ? "" : "indeterminate"}`, role: "progressbar", "aria-label": "Progress",
        "aria-valuemin": total ? 0 : null, "aria-valuemax": total || null, "aria-valuenow": total ? p.done : null }, fill),
      el("p", { text: total ? `${words} · ${p.done}/${total}` : words }));
  }

  const numbers = (pairs, bad = "") => el("dl", { class: "run-nums" }, pairs.filter(([, v]) => typeof v === "number").map(([k, v]) =>
    el("div", { class: k === bad && v > 0 ? "bad" : "" }, el("dt", { text: k }), el("dd", { text: num(v) }))));

  function runResult(run) {
    const r = run.result;
    if (!r || typeof r !== "object") return null;
    const kind = r.kind || run.kind;
    if (kind === "stats") {
      return el("div", { class: "run-result" }, numbers([["jobs", r.total_jobs], [ADMIN ? "unreviewed" : "new", r.new_jobs],
        ["applied", r.applied_jobs]]),
        el("button", { class: "btn tiny ghost", type: "button", "data-stats-run": run.id, "data-focus-key": "stats",
          text: "Open in Stats" }));
    }
    return typeof r.message === "string" && r.message ? el("p", { class: "run-msg", text: r.message }) : null;
  }

  async function showRun(id) {
    selectTab("runs");
    if (!runMap.has(id)) {
      try {
        addRuns([await fetchJSON(`/api/runs/${encodeURIComponent(id)}`)]);
      } catch (err) {
        notice({ tone: "warn", title: "Run not found", text: err.status === 404 ? `No run with the id ${id}.` : err.message });
        return;
      }
    }
    const entry = runNodes.get(id);
    if (!entry) return;
    flashUntil.set(id, Date.now() + 4000);
    entry.node.classList.remove("is-flash");
    void entry.node.offsetWidth; // restart the highlight animation
    entry.node.classList.add("is-flash");
    reveal(entry.node);
  }

  /* Poll every 3 s while a run is open, every 20 s otherwise, and not at all
     while the browser tab is hidden (it catches up when shown again). */
  let pollTimer = 0;
  let polling = false;
  let pollFailures = 0;

  function schedulePoll() {
    clearTimeout(pollTimer);
    if (document.visibilityState !== "visible") return;
    const open = [...runMap.values()].some((r) => OPEN.has(r.status));
    pollTimer = setTimeout(pollRuns, pollFailures > 2 ? 60000 : open ? 3000 : 20000);
  }

  async function pollRuns() {
    if (polling) return;
    polling = true;
    try {
      const list = await fetchJSON("/api/runs?limit=20");
      if (Array.isArray(list)) {
        const seen = new Set(list.map((r) => r && r.id));
        addRuns(list);
        // Workflow visibility can lag behind a start, so open runs the list missed are asked for one by one.
        const missing = [...runMap.values()].filter((r) => OPEN.has(r.status) && !seen.has(r.id)).slice(0, 5);
        const fresh = await Promise.all(missing.map((r) => fetchJSON(`/api/runs/${encodeURIComponent(r.id)}`).catch(() => null)));
        addRuns(fresh.filter(Boolean));
      }
      pollFailures = 0;
    } catch {
      pollFailures += 1;
    } finally {
      polling = false;
      schedulePoll();
    }
  }

  $("runs-refresh").addEventListener("click", pollRuns);
  $("runs-list").addEventListener("click", async (e) => {
    const statsButton = e.target.closest("[data-stats-run]");
    const run = statsButton && runMap.get(statsButton.dataset.statsRun);
    if (run && run.result) { renderStats(run.result, run); selectTab("stats"); return; }
    const button = e.target.closest("button[data-cancel]");
    if (!button) return;
    Object.assign(button, { disabled: true, textContent: "Cancelling…" });
    try {
      const res = await fetchJSON(`/api/runs/${encodeURIComponent(button.dataset.cancel)}/cancel`, { method: "POST" });
      if (!res || !res.cancelled) notice({ title: "Nothing to cancel", text: "The run had already finished." });
    } catch (err) {
      notice({ tone: "warn", title: "Couldn’t cancel the run", text: err.message });
    }
    pollRuns();
  });

  /* ============================================================ panel: jobs */

  const jobsForm = $("jobs-filter");
  let jobsLoaded = false;
  let jobsTicket = 0; // answers to superseded requests are dropped
  const STATUS_TITLES = { new: ADMIN ? "Unreviewed" : "New", applied: "Applied", skipped: "Skipped", expired: "Expired" };

  function jobsParams() {
    const f = jobsForm.elements;
    const params = new URLSearchParams();
    if (f.status.value && f.status.value !== "all") params.set("status", f.status.value);
    f.role.value.split(",").map((r) => r.trim()).filter(Boolean).forEach((r) => params.append("role", r)); // "DevOps, SRE" = either
    if (f.remote.checked) params.set("remote", "1");
    if (f.days.value) params.set("days", f.days.value);
    if (f.q.value.trim()) params.set("q", f.q.value.trim());
    return params;
  }

  function jobsTitle(params) {
    const roles = params.getAll("role").join(" / ");
    const bits = [params.get("remote") ? "remote" : "", params.get("days") ? `last ${params.get("days")}d` : ""].filter(Boolean);
    const what = `${STATUS_TITLES[params.get("status")] || "All"} ${roles ? `${roles} ` : ""}jobs`;
    return bits.length ? `${what} · ${bits.join(" · ")}` : what;
  }

  // A job list from the chat puts its filters in the form, so the operator can refine it from there.
  function syncForm(query) {
    const p = new URLSearchParams(query || "");
    const f = jobsForm.elements;
    const status = p.get("status") || "all";
    f.status.value = [...f.status.options].some((o) => o.value === status) ? status : "all";
    f.role.value = p.getAll("role").join(", ");
    f.remote.checked = p.get("remote") === "1";
    const days = p.get("days") || "";
    if (days && ![...f.days.options].some((o) => o.value === days)) f.days.append(el("option", { value: days, text: `${days} days` }));
    f.days.value = days;
    f.q.value = p.get("q") || "";
  }

  async function loadJobs(title) {
    jobsLoaded = true;
    const params = jobsParams();
    const ticket = ++jobsTicket;
    $("jobs-count").textContent = "Loading…";
    try {
      const query = params.toString();
      // The contract calls the page size `limit`; the /jobs query string calls it `size`. Both are sent.
      const data = (await fetchJSON(`/api/jobs?${query ? `${query}&` : ""}limit=50&size=50`)) || {};
      if (ticket !== jobsTicket) return;
      showJobList({ ...data, title: title || jobsTitle(params), filter_query: data.filter_query || `?${query}` });
    } catch (err) {
      if (ticket !== jobsTicket) return;
      showJobList({ title: title || jobsTitle(params), jobs: [], filter_query: `?${params}` });
      Object.assign($("jobs-empty"), { textContent: `Couldn’t load jobs: ${err.message}`, hidden: false });
      $("jobs-count").textContent = "";
    }
  }

  function showJobList({ title, jobs, total, filter_query: query }) {
    jobsLoaded = true;
    const list = Array.isArray(jobs) ? jobs.filter((j) => j && Number.isInteger(j.id)) : [];
    const count = typeof total === "number" ? total : list.length;
    $("jobs-title").textContent = title || "Jobs";
    const q = typeof query === "string" && query.length > 1 ? (query.startsWith("?") ? query : `?${query}`) : "";
    $("jobs-open").href = `${BASE}/jobs${q}`;
    $("jobs-count").textContent = list.length < count ? `Showing ${num(list.length)} of ${num(count)}` : plural(count, "job");
    $("jobs-list").replaceChildren(...list.map(jobItem));
    Object.assign($("jobs-empty"), { textContent: "No jobs match these filters.", hidden: list.length > 0 });
  }

  function jobItem(job) {
    const status = /^[a-z_]+$/.test(job.status || "") ? job.status : "";
    const posting = webUrl(job.job_url);
    return el("li", { class: "job" }, el("div", { class: "job-main" },
      el("a", { class: "job-title", href: jobUrl(job.id), text: job.position || `Job ${job.id}` }),
      el("p", { class: "job-meta", text: [job.company, job.location, job.salary].filter(Boolean).join(" · ") }),
      el("p", { class: "job-tags" },
        status ? el("span", { class: `pill ${status}`, text: status }) : null,
        job.role_category ? el("span", { class: "tag", text: job.role_category }) : null,
        job.site_name ? el("span", { class: "tag", text: job.site_name }) : null,
        job.scraped_at ? agoEl(job.scraped_at, "job-age") : null,
        posting ? el("a", { class: "job-posting", href: posting, target: "_blank", rel: "noopener", text: "Posting ↗" }) : null)));
  }

  jobsForm.addEventListener("submit", (e) => { e.preventDefault(); loadJobs(); });
  jobsForm.addEventListener("change", (e) => { if (e.target.matches("select, input[type=checkbox]")) loadJobs(); });

  /* =========================================================== panel: stats */

  let statsLoaded = false;
  let statsTicket = 0;

  async function loadStats() {
    statsLoaded = true;
    const ticket = ++statsTicket;
    $("stats-refresh").disabled = true;
    try {
      const snapshot = await fetchJSON(`/api/stats?days=${encodeURIComponent($("stats-days").value)}`);
      if (ticket === statsTicket) renderStats(snapshot);
    } catch (err) {
      const text = `Couldn’t load statistics: ${err.message}`;
      if (ticket === statsTicket) $("stats-body").replaceChildren(el("p", { class: "pane-empty bad", text }));
    } finally {
      $("stats-refresh").disabled = false;
    }
  }

  const tile = (label, value, href, tone) => el(href ? "a" : "div", { class: `tile ${tone || ""}`, href: href ? BASE + href : null },
    el("span", { class: "tile-n", text: value }), el("span", { class: "tile-l", text: label }));

  // Plain DOM bars in the overview page's .bars style; each links to the jobs it counts.
  function bars(items, total, key) {
    const list = (Array.isArray(items) ? items : []).filter((i) => i && typeof i.count === "number" && i.value != null);
    if (!list.length) return el("p", { class: "pane-empty", text: "Nothing counted yet." });
    const whole = total || Math.max(1, ...list.map((i) => i.count));
    return el("ul", { class: "bars" }, list.map((i) => {
      const fill = el("i");
      fill.style.width = `${Math.min(100, Math.max(1, Math.round((i.count / whole) * 100)))}%`;
      return el("li", {}, el("a", { href: `${BASE}/jobs?${key}=${encodeURIComponent(i.value)}` },
        el("span", { class: "k", title: String(i.value), text: String(i.value) }), el("span", { class: "bar" }, fill),
        el("span", { class: "v", text: num(i.count) })));
    }));
  }

  function runStatsSection(r) {
    const kinds = Object.keys(r.by_kind || {});
    const row = (kind) => {
      const c = r.by_kind[kind] || {};
      const failed = (c.failed || 0) + (c.timed_out || 0) + (c.terminated || 0);
      return el("tr", {}, el("th", { scope: "row", text: kindLabel(kind) }), el("td", { text: num(c.completed || 0) }),
        el("td", { text: num((c.queued || 0) + (c.running || 0)) }), el("td", { class: failed ? "bad" : "", text: num(failed) }),
        el("td", { text: span((r.avg_duration_s || {})[kind]) }), el("td", {}, agoEl((r.last_run_at || {})[kind] || null)));
    };
    return el("section", { class: "stats-section" }, el("h4", { text: `Runs · last ${plural(r.window_days, "day")}` }),
      el("div", { class: "tiles three" }, tile("runs", num(r.total)), tile("running", num(r.running), null, r.running ? "accent" : ""),
        tile("failed", num(r.failed), null, r.failed ? "bad" : "")),
      kinds.length ? el("table", { class: "run-table" },
        el("thead", {}, el("tr", {}, ["Kind", "Done", "Open", "Failed", "Avg", "Last"].map((h) => el("th", { scope: "col", text: h })))),
        el("tbody", {}, kinds.map(row))) : el("p", { class: "pane-empty", text: "No runs in this window." }));
  }

  function renderStats(s, run) {
    if (!s || typeof s !== "object") return;
    statsLoaded = true;
    if ([7, 30, 90].includes(s.window_days)) $("stats-days").value = String(s.window_days);
    const statuses = Object.entries(s.statuses || {}).map(([value, count]) => ({ value, count })).sort((a, b) => b.count - a.count);
    $("stats-body").replaceChildren(
      el("p", { class: "stats-captured" }, "Captured ", agoEl(s.captured_at), ` · last ${plural(s.window_days, "day")}`,
        run && run.id ? el("a", { href: `#run=${encodeURIComponent(run.id)}`, text: " · from a stats run" }) : null),
      el("div", { class: "tiles" },
        ...(ADMIN ? [
          tile("jobs stored", num(s.total_jobs), "/jobs"), tile("unreviewed", num(s.new_jobs), "/jobs?status=new", "accent"),
          tile("applied", num(s.applied_jobs), "/jobs?status=applied"),
          tile("added 7d", num(s.jobs_last7), "/jobs?days=7"), tile("added 30d", num(s.jobs_last30), "/jobs?days=30")]
        : [tile("jobs", num(s.total_jobs), "/jobs"), tile("new", num(s.new_jobs), "/jobs?status=new", "accent"),
          tile("applied", num(s.applied_jobs), "/jobs?status=applied"),
          tile("new this week", num(s.jobs_last7), "/jobs?days=7")])),
      !ADMIN ? null : el("div", { class: "fresh" }, el("p", {}, "Newest job added ", agoEl(s.last_scrape_at))),
      el("section", { class: "stats-section" }, el("h4", { text: "By status" }), bars(statuses, s.total_jobs, "status")),
      el("section", { class: "stats-section" }, el("h4", { text: "Top roles" }), bars(s.top_roles, s.total_jobs, "role")),
      ADMIN && s.runs && typeof s.runs === "object" ? runStatsSection(s.runs) : null);
  }

  $("stats-refresh").addEventListener("click", loadStats);
  $("stats-days").addEventListener("change", loadStats);

  /* ======================================================== panel: overview */

  function renderRunner(info) {
    runsOn = Boolean(info.configured);
    const temporal = runsOn && info.backend === "temporal";
    const ui = webUrl(info.ui_url);
    const chip = temporal && ui ? el("a", { href: ui, target: "_blank", rel: "noopener", title: "Open Temporal" })
      : temporal ? el("span") : el("a", { href: "#setup", title: info.reason || null });
    Object.assign(chip, { id: "runner-chip", className: `chip ${temporal ? "ok" : runsOn ? "" : "warn"}`,
      textContent: temporal ? `Temporal · ${info.namespace || "default"}` : runsOn ? "Local runs" : "Runs off" });
    if (!$("runner-chip")) { syncRunButtons(); return; } // the Assistant shows no runner internals
    $("runner-chip").replaceWith(chip);
    $("setup-reason").textContent = info.reason
      || (temporal ? "Runs execute on Temporal." : runsOn ? "Runs execute in-process." : "Runs are off.");
    $("setup-backend").textContent = info.backend || "—";
    $("setup-namespace").textContent = info.namespace || "—";
    $("setup-queue").textContent = info.task_queue || "—";
    syncRunButtons();
  }

  function syncRunButtons() {
    root.querySelectorAll("[data-quick=stats]").forEach((b) => {
      b.disabled = !runsOn;
      b.title = runsOn ? "" : ADMIN ? "Runs are off: see Runner setup" : "Not available right now";
    });
  }

  function renderFacts(facts) {
    for (const card of document.querySelectorAll("#r3-pipelines .pipe")) {
      const list = facts[card.dataset.pipeline];
      if (!Array.isArray(list)) continue;
      const dl = el("dl", { class: "pipe-facts" }, list.filter((f) => f && typeof f === "object").map((f) =>
        el("div", { class: f.warn ? "warn" : "" }, el("dt", { text: String(f.label ?? "") }), el("dd", { text: String(f.value ?? "") }))));
      const old = card.querySelector(".pipe-facts, .pipe-unavailable");
      if (old) old.replaceWith(dl); else card.querySelector("h3").after(dl);
    }
  }

  const RUN_VERBS = { stats: "Snapshot" };

  function suggestionCard(s) {
    let action = null;
    if (s.kind === "run" && s.request && s.request.params) {
      const request = { ...s.request, source: s.request.source || "panel" };
      const kind = request.params.kind;
      const run = async (e) => {
        const button = e.currentTarget;
        button.disabled = true;
        try { await startRun(request); } catch (err) { notice({ tone: "warn", title: `Couldn’t start: ${s.title}`, text: err.message }); }
        button.disabled = !runsOn;
      };
      action = el("button", { class: "btn tiny go", type: "button", disabled: !runsOn, text: RUN_VERBS[kind] || "Run", on: { click: run } });
    } else if (s.kind === "prompt" && typeof s.prompt === "string" && s.prompt) {
      action = el("button", { class: "btn tiny", type: "button", disabled: !ready, text: "Ask R3",
        on: { click: () => usePrompt(s.prompt) } });
    } else if (s.kind === "link" && portalUrl(s.href)) {
      action = el("a", { class: "btn tiny", href: portalUrl(s.href), text: "Open" });
    }
    return el("article", { class: `suggestion ${["accent", "warn"].includes(s.tone) ? s.tone : "neutral"}` },
      el("div", {}, el("h3", { text: s.title }), s.detail ? el("p", { text: s.detail }) : null), action);
  }

  function renderSuggestions(list) {
    const items = (Array.isArray(list) ? list : []).filter((s) => s && typeof s.title === "string" && s.title).slice(0, 6);
    Object.assign($("panel-suggestions"), { hidden: !items.length }).replaceChildren(...items.map(suggestionCard));
  }

  function panelNote(text) {
    let note = $("panel-note");
    if (!note) $("pane-overview").prepend((note = el("p", { class: "pane-note", id: "panel-note" })));
    Object.assign(note, { textContent: text, hidden: !text });
  }

  async function loadPanel() {
    let data;
    try {
      data = await fetchJSON("/api/panel");
    } catch (err) {
      panelNote(`Live panel data is unavailable (${err.message}); the numbers below are from page load.`);
      return;
    }
    if (!data || typeof data !== "object") return;
    panelNote("");
    if (data.runs_info && typeof data.runs_info === "object") renderRunner(data.runs_info);
    if (data.facts && typeof data.facts === "object") renderFacts(data.facts);
    renderSuggestions(data.suggestions);
    if (Array.isArray(data.recent_runs)) addRuns(data.recent_runs);
    if (data.stats && typeof data.stats === "object") {
      const roles = Array.isArray(data.stats.top_roles) ? data.stats.top_roles : [];
      const options = roles.filter((r) => r && typeof r.value === "string").map((r) => el("option", { value: r.value }));
      $("jobs-roles").replaceChildren(...options); // suggestions for the role filter
      if (!statsLoaded) renderStats(data.stats);
    }
  }

  async function quickAction(kind, button) {
    if (kind === "review") {
      syncForm("?status=new");
      selectTab("jobs");
      loadJobs(ADMIN ? "Unreviewed jobs" : "New jobs");
      reveal($("pane-jobs"));
      return;
    }
    if (kind !== "stats") return;
    const params = { kind: "stats", days: Number($("stats-days").value) || 30 };
    button.disabled = true;
    try {
      await startRun({ params, source: "panel" });
    } catch (err) {
      notice({ tone: "warn", title: `Couldn’t start the ${kindLabel(params.kind).toLowerCase()} run`, text: err.message });
    }
    button.disabled = !runsOn;
  }

  function openSetup() {
    const setup = $("setup");
    if (!setup) return;
    setup.open = true;
    setup.scrollIntoView({ block: "nearest" });
    setup.querySelector("summary").focus({ preventScroll: true });
  }

  $("r3-panel").addEventListener("click", (e) => {
    const quick = e.target.closest("[data-quick]");
    if (quick && !quick.disabled) quickAction(quick.dataset.quick, quick);
    if (e.target.closest('a[href="#setup"]')) { e.preventDefault(); openSetup(); }
  });

  /* ================================================= chat actions -> panel */

  /* A chat turn's `action` lands here: the panel does the work and returns
     a chip for the chat message that brings the result into view. */
  function panelAction(action) {
    switch (action.kind) {
      case "jobs": {
        syncForm(action.filter_query);
        jobsTicket += 1; // a slower filter request must not replace this list
        showJobList(action);
        selectTab("jobs");
        const total = typeof action.total === "number" ? action.total : (action.jobs || []).length;
        return { label: `${plural(total, "job")} in the panel`, go: () => { selectTab("jobs"); reveal($("pane-jobs")); } };
      }
      case "stats": {
        if (action.run) addRuns([action.run]);
        statsTicket += 1;
        renderStats(action.snapshot, action.run);
        selectTab("stats");
        return { label: "Statistics in the panel", go: () => { selectTab("stats"); reveal($("pane-stats")); } };
      }
      case "unavailable": {
        const what = kindLabel(action.intent);
        const node = notice({ tone: "warn", title: `${what} is unavailable`, text: action.reason, hint: action.hint });
        return { label: `${what} unavailable`, tone: "warn", go: () => reveal(node) };
      }
      default:
        return null; // advice: nothing for the panel
    }
  }

  /* ================================================================= start */

  function onHash() {
    let hash = "";
    try { hash = decodeURIComponent(location.hash.replace(/^#/, "")); } catch { return; }
    if (hash.startsWith("run=") && hash.length > 4) showRun(hash.slice(4));
    else if (hash === "setup") openSetup();
  }

  // The previous Controller kept an actions log; runs replace it.
  session.remove("r3.controller.actions.v1");
  session.remove("r3.controller.pipeline");

  restoreHistory();
  resumeFollowups();
  applyStoredModel();
  showModel();
  controls();
  selectTab(/^#run=/.test(location.hash) ? "runs" : session.get(TAB_KEY) || "overview");
  syncRunButtons();
  loadModels();
  loadPanel().finally(() => { pollRuns(); onHash(); });
  window.addEventListener("hashchange", onHash);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") pollRuns(); else clearTimeout(pollTimer);
  });
  setInterval(tick, 5000);
})();
