/* htmx behaviours that would otherwise need a trigger filter, which htmx
   evaluates with eval and the Content-Security-Policy (script-src 'self')
   forbids. tests/test_safe_url.py keeps filters out of the templates. */
(() => {
  "use strict";
  // data-pause-while-edited="<textarea id>": skip a poll while that box holds
  // unsaved typing, so a refresh never throws away the person's edits.
  document.addEventListener("htmx:beforeRequest", (e) => {
    const el = e.detail && e.detail.elt;
    const id = el && el.dataset ? el.dataset.pauseWhileEdited : "";
    if (!id) return;
    const box = document.getElementById(id);
    if (box && box.value !== box.defaultValue) e.preventDefault();
  });
})();
