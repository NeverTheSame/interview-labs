/* R3 Settings page: save values per group, clear portal overrides, and run
   connection tests. The server renders everything; this script only wires
   the controls. Secrets are sent when typed and never read back. */
(() => {
  "use strict";

  const root = document.getElementById("r3-settings");
  if (!root) return;
  const base = root.dataset.basePath || "";
  const api = (path) => `${base}/api/settings${path}`;
  const SOURCE = { portal: "set in portal", env: "from environment", default: "default" };
  const results = new Map(); // test id -> last result

  let state = {};
  try {
    state = JSON.parse(document.getElementById("set-state").textContent || "{}");
  } catch (_) {
    state = {};
  }
  const storeOk = !!(state.store && state.store.available) && !state.pinned;

  const $ = (sel, el = root) => el.querySelector(sel);
  const $$ = (sel, el = root) => Array.from(el.querySelectorAll(sel));

  // --- values ----------------------------------------------------------------

  function control(field) {
    return field.querySelector("input, select, textarea");
  }

  function isReplacing(field) {
    const c = control(field);
    return field.dataset.secret === "true" && c && !c.hidden;
  }

  function dirtyValue(field) {
    // The value to send for this field, or undefined when unchanged.
    const c = control(field);
    if (!c) return undefined;
    if (field.dataset.secret === "true") {
      if (!isReplacing(field)) return undefined;
      const v = c.value.trim();
      return v === "" ? undefined : c.value;
    }
    return c.value !== (c.dataset.initial || "") ? c.value : undefined;
  }

  function dirtyValues(scope) {
    const out = {};
    $$(".set-field", scope).forEach((f) => {
      const v = dirtyValue(f);
      if (v !== undefined) out[f.dataset.name] = v;
    });
    return out;
  }

  function refreshDirty(form) {
    const n = Object.keys(dirtyValues(form)).length;
    const btn = form.querySelector('button[type="submit"]');
    btn.disabled = !storeOk || n === 0;
    btn.textContent = n ? `Save ${n} change${n === 1 ? "" : "s"}` : "Save";
    $$(".set-field", form).forEach((f) => f.classList.toggle("is-dirty", dirtyValue(f) !== undefined));
  }

  // --- rendering state after a save -----------------------------------------------

  function keyState(name) {
    for (const g of state.groups || []) for (const k of g.keys) if (k.name === name) return k;
    return null;
  }

  function renderField(field, k) {
    field.dataset.source = k.source;
    const src = field.querySelector("[data-src]");
    src.textContent = SOURCE[k.source] || k.source;
    src.className = `set-src src-${k.source}`;
    field.querySelector('[data-action="clear"]').hidden = !(k.source === "portal" && storeOk);
    const err = field.querySelector("[data-error]");
    err.hidden = true;
    err.textContent = "";
    const c = control(field);
    if (k.secret) {
      const wrap = field.querySelector(".set-secret");
      wrap.classList.toggle("is-set", k.set);
      wrap.classList.toggle("is-unset", !k.set);
      const masked = field.querySelector("[data-masked]");
      masked.hidden = !k.set;
      masked.querySelector(".set-dots").textContent = k.masked || "";
      masked.querySelector("[data-masked-src]").textContent = SOURCE[k.source] || "";
      const replace = field.querySelector('[data-action="replace"]');
      replace.hidden = !k.set;
      c.value = "";
      c.hidden = k.set;
      if (c.tagName === "INPUT") c.placeholder = k.set ? "Paste a new value" : "Not set";
      field.querySelector('[data-action="cancel"]').hidden = true;
    } else {
      c.value = k.value || "";
      c.dataset.initial = k.value || "";
    }
  }

  function renderTestsConfigured() {
    for (const g of state.groups || []) {
      for (const t of g.tests) {
        const li = root.querySelector(`.set-test[data-test="${t.id}"]`);
        if (!li) continue;
        li.dataset.configured = t.configured ? "true" : "false";
        if (!results.has(t.id)) setPill(t.id, "idle", t.configured ? "not tested" : "not configured");
      }
    }
  }

  function showApplied(apply) {
    const el = document.getElementById("set-applied");
    if (!apply) return;
    const time = new Date(apply.applied_at).toLocaleTimeString();
    const parts = [`Applied at ${time}`, `rebuilt ${apply.rebuilt.join(", ")}`];
    (apply.messages || []).forEach((m) => parts.push(m));
    el.textContent = parts.join(" · ");
    el.className = `set-notice ${apply.ok ? "ok" : "warn"}`;
    el.hidden = false;
  }

  // --- API ------------------------------------------------------------------------

  async function call(method, path, body) {
    const resp = await fetch(api(path), {
      method,
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: "same-origin",
    });
    let data = null;
    try {
      data = await resp.json();
    } catch (_) {
      data = null;
    }
    return { ok: resp.ok, status: resp.status, data };
  }

  function errorText(res) {
    const d = res.data && res.data.detail;
    if (typeof d === "string") return d;
    if (d && d.message) return d.message;
    return `The server answered HTTP ${res.status}.`;
  }

  function showFieldErrors(res) {
    const errors = (res.data && res.data.detail && res.data.detail.errors) || {};
    Object.entries(errors).forEach(([name, msg]) => {
      const field = root.querySelector(`.set-field[data-name="${CSS.escape(name)}"]`);
      if (!field) return;
      const err = field.querySelector("[data-error]");
      err.textContent = `${field.querySelector("label").textContent} ${msg}.`;
      err.hidden = false;
    });
    return Object.keys(errors).length;
  }

  async function save(form, values) {
    const status = form.querySelector("[data-status]");
    const btn = form.querySelector('button[type="submit"]');
    $$("[data-error]", form).forEach((e) => { e.hidden = true; });
    btn.disabled = true;
    status.className = "set-status";
    status.textContent = "Saving and applying…";
    let res;
    try {
      res = await call("PUT", "", { values });
    } catch (_) {
      status.textContent = "Could not reach the portal.";
      status.className = "set-status bad";
      refreshDirty(form);
      return;
    }
    if (!res.ok) {
      showFieldErrors(res);
      status.textContent = errorText(res);
      status.className = "set-status bad";
      refreshDirty(form);
      return;
    }
    state = res.data;
    Object.keys(values).forEach((name) => {
      const field = root.querySelector(`.set-field[data-name="${CSS.escape(name)}"]`);
      const k = keyState(name);
      if (field && k) renderField(field, k);
    });
    renderTestsConfigured();
    const apply = res.data.apply;
    showApplied(apply);
    status.textContent = apply ? (apply.ok ? "Saved and applied." : "Saved; see the notice above.") : "Nothing to save.";
    status.className = `set-status ${apply && !apply.ok ? "bad" : "good"}`;
    refreshDirty(form);
  }

  // --- tests ----------------------------------------------------------------------

  function setPill(id, kind, text) {
    const pill = root.querySelector(`[data-pill="${id}"]`);
    if (!pill) return;
    pill.className = `set-pill ${kind}`;
    pill.textContent = text;
  }

  function updateSummary() {
    const all = $$(".set-test").map((li) => li.dataset.test);
    const done = all.filter((id) => results.has(id));
    const ok = done.filter((id) => results.get(id).ok).length;
    const unset = done.filter((id) => !results.get(id).ok && results.get(id).configured === false).length;
    const count = document.getElementById("set-summary-count");
    const detail = document.getElementById("set-summary-detail");
    if (!done.length) {
      count.textContent = `${all.length} connections`;
      detail.textContent = "not tested yet";
      return;
    }
    count.textContent = `${ok} of ${all.length} connections OK`;
    const failed = done.length - ok - unset;
    const bits = [];
    if (failed) bits.push(`${failed} failing`);
    if (unset) bits.push(`${unset} not configured`);
    if (done.length < all.length) bits.push(`${all.length - done.length} not tested`);
    detail.textContent = bits.join(" · ") || "all good";
    count.className = failed ? "bad" : "";
  }

  async function runTest(id) {
    const btn = root.querySelector(`button[data-action="test"][data-test="${id}"]`);
    const msg = root.querySelector(`[data-result="${id}"]`);
    if (btn) btn.disabled = true;
    setPill(id, "busy", "testing…");
    let res;
    try {
      res = await call("POST", `/test/${encodeURIComponent(id)}`, { values: dirtyValues(root) });
    } catch (_) {
      res = { ok: false, status: 0, data: null };
    }
    let result;
    if (res.ok && res.data) {
      result = res.data;
    } else {
      if (res.status === 400) showFieldErrors(res);
      result = { ok: false, configured: true, latency_ms: 0,
        message: res.status ? errorText(res) : "Could not reach the portal." };
    }
    results.set(id, result);
    if (result.ok) setPill(id, "good", `OK · ${result.latency_ms} ms`);
    else if (result.configured === false) setPill(id, "idle", "not configured");
    else setPill(id, "bad", result.latency_ms ? `failed · ${result.latency_ms} ms` : "failed");
    msg.textContent = result.message;
    msg.classList.toggle("bad", !result.ok && result.configured !== false);
    if (btn) btn.disabled = false;
    updateSummary();
    return result;
  }

  // --- wiring ---------------------------------------------------------------------

  $$(".set-form").forEach((form) => {
    form.addEventListener("input", (ev) => {
      // An edited field's old error no longer applies.
      const field = ev.target.closest(".set-field");
      if (field) field.querySelector("[data-error]").hidden = true;
      const status = form.querySelector("[data-status]");
      if (status.classList.contains("bad")) { status.textContent = ""; status.className = "set-status"; }
      refreshDirty(form);
    });
    form.addEventListener("change", () => refreshDirty(form));
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const values = dirtyValues(form);
      if (Object.keys(values).length) save(form, values);
    });
    form.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-action]");
      if (!btn) return;
      const field = btn.closest(".set-field");
      const action = btn.dataset.action;
      if (action === "replace") {
        const c = control(field);
        c.hidden = false;
        btn.hidden = true;
        field.querySelector("[data-masked]").hidden = true;
        field.querySelector('[data-action="cancel"]').hidden = false;
        c.focus();
      } else if (action === "cancel") {
        const k = keyState(field.dataset.name);
        if (k) renderField(field, k);
      } else if (action === "clear") {
        save(form, { [field.dataset.name]: null });
      }
      refreshDirty(form);
    });
    // Enter in a single-line input saves the group, as a form would.
    form.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && ev.target.matches(".set-secret input, .set-secret textarea")) {
        const field = ev.target.closest(".set-field");
        const k = keyState(field.dataset.name);
        if (k && k.set) { renderField(field, k); refreshDirty(form); }
      }
    });
    refreshDirty(form);
  });

  $$('button[data-action="test"]').forEach((btn) => {
    btn.disabled = false;
    btn.addEventListener("click", () => runTest(btn.dataset.test));
  });
  $$('button[data-action="replace"]').forEach((btn) => { btn.disabled = !storeOk; });
  $$('button[data-action="clear"]').forEach((btn) => { if (!storeOk) btn.hidden = true; });

  if (state.applied) showApplied(state.applied);

  const testAll = document.getElementById("set-test-all");
  testAll.disabled = false;
  testAll.addEventListener("click", async () => {
    testAll.disabled = true;
    testAll.textContent = "Testing…";
    const ids = $$(".set-test").map((li) => li.dataset.test);
    await Promise.all(ids.map((id) => runTest(id)));
    testAll.disabled = false;
    testAll.textContent = "Test all";
  });

  window.addEventListener("beforeunload", (ev) => {
    if (Object.keys(dirtyValues(root)).length) {
      ev.preventDefault();
      ev.returnValue = "";
    }
  });
})();
