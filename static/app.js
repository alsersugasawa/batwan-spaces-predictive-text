// The search box: suggestions as you type, the best one as grey text after
// the cursor. Every suggestion is set as text, never as markup.
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const input = $("q"), list = $("list"), ghost = $("ghost"), status = $("status");
  const DEBOUNCE_MS = 140;
  let lang = "any", items = [], index = -1, timer = 0, inflight = null, lastQuery = null;

  const norm = s => s.normalize("NFC").toLowerCase().replace(/[’‘ʼ]/g, "'")
                     .replace(/["“”«»?!.,;:()[\]{}]/g, " ").replace(/\s+/g, " ").replace(/^ /, "");

  function setGhost(completion) {
    const typed = input.value;
    const show = completion && norm(typed) === typed && input.selectionStart === typed.length
                 && input.scrollLeft === 0;
    ghost.querySelector(".typed").textContent = show ? typed : "";
    ghost.querySelector(".rest").textContent = show ? completion : "";
  }

  function close() {
    list.hidden = true; input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant"); index = -1;
  }

  function render(query) {
    list.textContent = "";
    if (!items.length) { close(); setGhost(""); return; }
    const top = Math.max(...items.map(i => i.score), 1e-9);
    items.forEach((it, i) => {
      const li = document.createElement("li");
      li.id = `opt-${i}`; li.setAttribute("role", "option"); li.setAttribute("aria-selected", String(i === index));
      const text = document.createElement("span"); text.className = "text";
      const head = it.text.startsWith(query) ? query : "";
      text.append(document.createTextNode(head));
      const b = document.createElement("b"); b.textContent = it.text.slice(head.length); text.append(b);
      const meter = document.createElement("span"); meter.className = "meter"; meter.title = `${Math.round(it.score * 100)}%`;
      const bar = document.createElement("span"); bar.style.width = `${Math.max(6, 100 * it.score / top)}%`; meter.append(bar);
      const tag = document.createElement("span"); tag.className = "tag"; tag.textContent = "AI";
      tag.title = "Predicted by the model";
      li.append(text, meter, tag);
      li.addEventListener("mousedown", e => { e.preventDefault(); accept(it.text); });
      list.append(li);
    });
    list.hidden = false; input.setAttribute("aria-expanded", "true");
    if (index >= 0) input.setAttribute("aria-activedescendant", `opt-${index}`);
    else input.removeAttribute("aria-activedescendant");
    setGhost(items[0] && items[0].text.startsWith(input.value) ? items[0].text.slice(input.value.length) : "");
  }

  async function fetchSuggestions() {
    const typed = input.value;
    const q = norm(typed);
    if (!q.trim()) { items = []; render(""); return; }
    if (q === lastQuery) return;
    if (inflight) inflight.abort();
    inflight = new AbortController();
    lastQuery = q;
    try {
      const r = await fetch(`/api/suggest?${new URLSearchParams({ q, lang, limit: 4 })}`, { signal: inflight.signal });
      const data = await r.json();
      if (norm(input.value) !== data.query) return;          // typed on since
      if (data.stale) return;
      if (data.status && data.status !== "up") showStatus(data.status);
      if (typeof data.ms === "number" && !data.cached && data.items.length) showStatus("up", data.ms);
      items = data.items || []; index = -1; render(data.query);
    } catch (e) {
      if (e.name !== "AbortError") { lastQuery = null; }
    }
  }

  function schedule() { clearTimeout(timer); timer = setTimeout(fetchSuggestions, DEBOUNCE_MS); }

  function accept(text) {
    input.value = text + " "; close(); setGhost(""); items = []; lastQuery = null;
    input.focus(); input.setSelectionRange(input.value.length, input.value.length);
    fetchSuggestions();   // the next word, at once
  }

  function search() {
    const q = input.value.trim();
    if (q) window.open(`https://batwan.org/#/search/${encodeURIComponent(q)}`, "_blank", "noopener");
  }

  input.addEventListener("input", () => { setGhost(""); schedule(); });
  input.addEventListener("keydown", e => {
    const atEnd = input.selectionStart === input.value.length && input.selectionEnd === input.value.length;
    const rest = ghost.querySelector(".rest").textContent;
    if ((e.key === "Tab" && !e.shiftKey || e.key === "ArrowRight" && atEnd) && rest) {
      e.preventDefault(); accept(input.value + rest); return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (list.hidden || !items.length) { if (e.key === "ArrowDown") { lastQuery = null; fetchSuggestions(); } return; }
      e.preventDefault();
      index = e.key === "ArrowDown" ? (index + 1) % items.length : (index <= 0 ? items.length - 1 : index - 1);
      render(norm(input.value));
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      if (index >= 0 && items[index]) accept(items[index].text); else { close(); search(); }
      return;
    }
    if (e.key === "Escape") { close(); setGhost(""); }
  });
  input.addEventListener("blur", () => setTimeout(close, 120));
  input.addEventListener("focus", () => { if (items.length) render(norm(input.value)); });
  input.addEventListener("scroll", () => setGhost(""));
  $("form").addEventListener("submit", e => { e.preventDefault(); search(); });

  document.querySelectorAll(".directions button").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(".directions button").forEach(x => x.setAttribute("aria-checked", String(x === b)));
    lang = b.dataset.lang; lastQuery = null; input.focus(); fetchSuggestions();
  }));
  document.querySelectorAll(".tries button").forEach(b => b.addEventListener("click", () => {
    input.value = b.dataset.try; lastQuery = null; input.focus();
    input.setSelectionRange(input.value.length, input.value.length); fetchSuggestions();
  }));

  function showStatus(state, ms) {
    status.className = `status ${state === "up" ? "up" : state === "down" ? "down" : ""}`;
    status.querySelector(".label").textContent =
      state === "up" ? (ms ? `Model ready · last suggestion in ${(ms / 1000).toFixed(1)} s` : "Model ready")
      : state === "down" ? "The model is not running. Try again later."
      : "The model is loading - a minute or two after the Space starts.";
  }

  async function poll() {
    let state = "loading";
    try {
      const r = await fetch("/health/model");
      const data = await r.json();
      state = data.model;
      if (data.name) $("modelname").textContent = data.name.replace(/\.gguf$/, "");
    } catch (e) { /* the Space is starting */ }
    if (!status.classList.contains("up") || state !== "up") showStatus(state);
    setTimeout(poll, state === "up" ? 60000 : 4000);
  }
  poll();
})();
