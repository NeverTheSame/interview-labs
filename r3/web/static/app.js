// Site-wide behaviour that must not be an inline handler (the page's
// Content-Security-Policy forbids inline scripts). Delegated, so rows that
// htmx swaps in later work without re-binding.
document.addEventListener("click", function (ev) {
  var trigger = ev.target.closest ? ev.target.closest("[data-toggle-open]") : null;
  if (!trigger) return;
  var target = document.getElementById(trigger.getAttribute("data-toggle-open"));
  if (target) target.classList.toggle("open");
});
