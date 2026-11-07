// The resume builder (/resume/build): send one answer, show the next
// question, swap in the re-rendered preview and checklist. The server
// renders both fragments (escaped by Jinja), so this file never builds
// HTML from user text: messages are added with textContent only.
(() => {
  "use strict";
  const root = document.getElementById("rb");
  if (!root) return;
  const base = root.dataset.basePath || "";
  const form = document.getElementById("rb-form");
  const input = document.getElementById("rb-input");
  const send = document.getElementById("rb-send");
  const log = document.getElementById("rb-log");
  const scroll = document.getElementById("rb-scroll");
  const thinking = document.getElementById("rb-thinking");
  const error = document.getElementById("rb-error");
  let busy = false;

  const message = (role, text, note) => {
    const article = document.createElement("article");
    article.className = `msg ${role}`;
    const label = document.createElement("div");
    label.className = "msg-label";
    label.textContent = role === "user" ? "You" : "R3";
    const body = document.createElement("div");
    body.className = "msg-body";
    body.textContent = text;
    article.append(label, body);
    if (note) {
      const foot = document.createElement("p");
      foot.className = "msg-foot";
      foot.textContent = note;
      article.append(foot);
    }
    log.append(article);
    return article;
  };

  const toBottom = () => { scroll.scrollTop = scroll.scrollHeight; };

  const showError = (text) => {
    error.textContent = text;
    error.hidden = !text;
  };

  const setBusy = (on) => {
    busy = on;
    thinking.hidden = !on;
    send.disabled = on;
    document.getElementById("rb-log").setAttribute("aria-busy", on ? "true" : "false");
    if (on) toBottom();
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    const answer = input.value.trim();
    if (!answer) { showError("Type an answer first (or \"skip\" to move on)."); return; }
    showError("");
    // Echo the answer and clear the box at once: on a slow model turn the
    // message must read as sent, not swallowed. If the send fails, the
    // bubble goes and the answer goes back into the box, so nothing is lost.
    const userNode = message("user", answer);
    input.value = "";
    setBusy(true);
    try {
      const res = await fetch(`${base}/resume/build/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ answer }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        userNode.remove();
        input.value = answer;
        showError(data.error || `Could not send that answer (HTTP ${res.status}). Try again.`);
        return;
      }
      message("assistant", data.question, data.note);
      document.getElementById("rb-preview").innerHTML = data.draft_html;
      document.getElementById("rb-progress").innerHTML = data.progress_html;
      const save = document.querySelector(".rb-save .btn");
      if (save) save.classList.toggle("primary", Boolean(data.done));
    } catch (e) {
      userNode.remove();
      input.value = answer;
      showError("Network error: your answer is back in the box. Try again.");
    } finally {
      setBusy(false);
      toBottom();
      input.focus();
    }
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  document.getElementById("rb-reset").addEventListener("submit", (event) => {
    if (!window.confirm("Start over? This clears every answer and the draft resume.")) event.preventDefault();
  });

  toBottom();
  input.focus();
})();
