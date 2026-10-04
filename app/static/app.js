// StreamClearance UI. Plain JS, no dependencies.
// All server/model text is inserted with textContent (via el()), never innerHTML.
"use strict";

const $ = (id) => document.getElementById(id);
const state = { videoId: null, jobId: null, job: null, countries: [], pollTimer: null };

const RISK_LABEL = { red: "High risk", amber: "Review", green: "Clear" };
const CATEGORY_LABEL = (c) => c.replace(/_/g, " ").replace(/\bor\b/g, "/");

// ------------------------------------------------------------------ helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function fmtTime(s) {
  s = Math.max(0, Math.floor(Number(s) || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function setMsg(id, text, isError = false) {
  const n = $(id);
  n.textContent = text || "";
  n.classList.toggle("error", !!isError);
}

function overlaps(a, b) {
  return (a.start_s < b.end_s && b.start_s < a.end_s) || a.start_s === b.start_s;
}

// ------------------------------------------------------------------ access code

function askAccessCode() {
  return new Promise((resolve) => {
    const dlg = $("access-dialog");
    const input = $("access-input");
    input.value = "";
    dlg.addEventListener("close", function onClose() {
      dlg.removeEventListener("close", onClose);
      const code = input.value.trim();
      if (code) {
        document.cookie = `sc_access=${encodeURIComponent(code)}; path=/; max-age=${60 * 60 * 24 * 7}; SameSite=Strict`;
      }
      resolve(!!code);
    });
    dlg.showModal();
    input.focus();
  });
}

async function api(path, opts = {}) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const res = await fetch(path, { credentials: "same-origin", ...opts });
    if (res.status === 401) {
      if (!(await askAccessCode())) throw new Error("Access code required");
      continue;
    }
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) throw new Error((body && body.detail && String(body.detail)) || `HTTP ${res.status}`);
    return body;
  }
  throw new Error("Access code rejected");
}

// ------------------------------------------------------------------ init

async function init() {
  try {
    const status = await api("/api/status");
    $("mock-banner").hidden = !status.mock;
    $("dz-limits").textContent =
      `MP4, MOV or WebM · up to ${status.max_upload_mb} MB · ${Math.round(status.max_duration_s / 60)} min`;
    state.countries = await api("/api/countries");
    renderCountries();
  } catch (e) {
    setMsg("analyze-msg", e.message, true);
  }
  setupDropzone();
  $("analyze-btn").addEventListener("click", startAnalysis);
  $("download-btn").addEventListener("click", downloadJson);
}

function renderCountries() {
  const list = $("country-list");
  list.replaceChildren(...state.countries.map((c) => el("label", { class: "country" },
    el("input", { type: "checkbox", value: c.code, checked: true, onchange: updateAnalyzeBtn }),
    el("span", { class: "name" }, `${c.name} (${c.code})`),
    el("span", { class: "law" },
      c.law,
      c.label ? el("span", { class: "note" }, ` · ${c.label}`) : null,
      c.rules_status !== "reviewed" ? el("span", { class: "note" }, " · draft rules, unreviewed") : null),
  )));
  updateAnalyzeBtn();
}

function selectedCountries() {
  return [...document.querySelectorAll("#country-list input:checked")].map((i) => i.value);
}

function updateAnalyzeBtn() {
  $("analyze-btn").disabled = !state.videoId || selectedCountries().length === 0;
}

// ------------------------------------------------------------------ upload

function setupDropzone() {
  const dz = $("dropzone");
  const input = $("file-input");
  input.addEventListener("change", () => input.files[0] && upload(input.files[0]));
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => {
    e.preventDefault(); dz.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => {
    e.preventDefault(); dz.classList.remove("over");
  }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f) upload(f);
  });
}

function upload(file) {
  const ok = /\.(mp4|mov|webm)$/i.test(file.name);
  if (!ok) { setMsg("upload-msg", "Please choose an MP4, MOV or WebM file.", true); return; }
  state.videoId = null;
  updateAnalyzeBtn();
  const bar = $("upload-progress");
  bar.hidden = false;
  bar.firstElementChild.style.width = "0";
  setMsg("upload-msg", `Uploading ${file.name}…`);

  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) bar.firstElementChild.style.width = `${(100 * e.loaded) / e.total}%`;
  };
  xhr.onload = async () => {
    bar.hidden = true;
    let body = null;
    try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
    if (xhr.status === 401) {
      if (await askAccessCode()) upload(file);
      return;
    }
    if (xhr.status !== 200 || !body) {
      setMsg("upload-msg", (body && body.detail) || `Upload failed (HTTP ${xhr.status})`, true);
      return;
    }
    state.videoId = body.video_id;
    setMsg("upload-msg", "Uploaded. Pick countries and press Analyze.");
    const player = $("player");
    player.src = `/api/videos/${encodeURIComponent(body.video_id)}`;
    $("player-card").hidden = false;
    const dur = body.duration_s != null ? ` · ${fmtTime(body.duration_s)}` : "";
    $("video-meta").textContent = `${body.filename}${dur} · ${(body.size_bytes / 1e6).toFixed(1)} MB`;
    updateAnalyzeBtn();
  };
  xhr.onerror = () => { bar.hidden = true; setMsg("upload-msg", "Upload failed (network error).", true); };
  xhr.send(form);
}

// ------------------------------------------------------------------ analysis

async function startAnalysis() {
  const countries = selectedCountries();
  if (!state.videoId || !countries.length) return;
  $("analyze-btn").disabled = true;
  setMsg("analyze-msg", "");
  $("report").hidden = true;
  try {
    const { job_id } = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: state.videoId, countries }),
    });
    state.jobId = job_id;
    $("progress-card").hidden = false;
    poll();
  } catch (e) {
    setMsg("analyze-msg", e.message, true);
    updateAnalyzeBtn();
  }
}

async function poll() {
  clearTimeout(state.pollTimer);
  let job;
  try {
    job = await api(`/api/jobs/${encodeURIComponent(state.jobId)}`);
  } catch (e) {
    setMsg("analyze-msg", e.message, true);
    state.pollTimer = setTimeout(poll, 3000);
    return;
  }
  state.job = job;
  renderStages(job);
  if (job.mock) $("mock-banner").hidden = false;
  if (job.status === "done") {
    renderReport(job.report);
    updateAnalyzeBtn();
  } else if (job.status === "error") {
    setMsg("analyze-msg", `Analysis failed: ${job.error || "unknown error"}`, true);
    updateAnalyzeBtn();
  } else {
    state.pollTimer = setTimeout(poll, 1200);
  }
}

function renderStages(job) {
  $("stages").replaceChildren(...job.stages.map((s) => el("li", { class: s.status },
    el("span", { class: "dot", "aria-hidden": "true" }),
    el("span", {}, s.label, el("span", { class: "visually-hidden" }, ` (${s.status})`)),
    el("span", { class: "detail" }, s.detail || (s.status === "pending" ? "" : s.status)),
  )));
}

// ------------------------------------------------------------------ report

function seek(t) {
  const player = $("player");
  if (!player.src) return;
  player.currentTime = Number(t) || 0;
  player.play().catch(() => {});
  if (window.matchMedia("(max-width: 860px)").matches) {
    player.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function renderReport(report) {
  $("report").hidden = false;
  if (report.mock) $("mock-banner").hidden = false;

  $("summary").replaceChildren(...report.countries.map((c) => el("a", { href: `#cr-${c.country}` },
    el("span", { class: "pill" }, c.country, el("span", { class: `badge ${c.overall_risk}` }, c.overall_risk)))));

  const errs = $("report-errors");
  errs.hidden = !report.errors.length;
  errs.textContent = report.errors.length ? `Some checks failed: ${report.errors.join("; ")}` : "";

  $("country-reports").replaceChildren(...report.countries.map(renderCountry));
  $("disclaimer").textContent = report.disclaimer;
  $("report").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderCountry(c) {
  const meta = state.countries.find((x) => x.code === c.country) || {};
  const cards = Object.fromEntries(c.cited_cards.map((k) => [k.id, k]));
  const findings = c.findings.length
    ? el("ol", { class: "findings" }, c.findings.map((f) => renderFinding(c, f, cards)))
    : el("p", { class: "empty" }, "No moments in this video matched this country's rule cards.");

  return el("article", { class: "card country-report", id: `cr-${c.country}` },
    el("div", { class: "cr-head" },
      el("h3", {}, meta.name || c.country),
      el("span", { class: `badge ${c.overall_risk}` }, `${c.overall_risk} · ${RISK_LABEL[c.overall_risk]}`),
      c.rules_status !== "reviewed"
        ? el("span", { class: "badge draft", title: "Rule cards have not been reviewed by a person yet" }, "draft rules")
        : null),
    el("p", { class: "cr-law" }, c.law, c.label ? el("span", { class: "note" }, ` · ${c.label}`) : null),
    el("p", { class: "framing" }, c.risk_framing),
    findings,
    c.dropped_findings
      ? el("p", { class: "empty" }, `${c.dropped_findings} model suggestion(s) were dropped by the citation validator.`)
      : null,
  );
}

function renderFinding(c, f, cards) {
  const drift = c.translation.filter((d) => overlaps(d, f));
  return el("li", { class: `finding ${f.risk}` },
    el("div", { class: "f-head" },
      el("button", { class: "chip", type: "button", onclick: () => seek(f.start_s),
        title: "Jump to this moment" }, `▶ ${fmtTime(f.start_s)}–${fmtTime(f.end_s)}`),
      el("span", { class: `badge ${f.risk}` }, f.risk),
      el("span", { class: "cat" }, CATEGORY_LABEL(f.category)),
      f.creator_liability ? el("span", { class: "badge creator" }, "creator liability") : null,
      drift.map((d) => el("span", { class: `badge drift-${d.drift}` }, `${d.language}: drift ${d.drift}`)),
    ),
    el("div", { class: "f-row" }, el("span", { class: "k" }, "Why"), f.rationale),
    el("div", { class: "f-row" }, el("span", { class: "k" }, "Suggested fix"), f.suggested_fix),
    f.rule_card_ids.map((id) => renderCite(c.country, cards[id], id)),
    drift.map(renderDrift),
  );
}

function renderCite(cc, card, id) {
  if (!card) return el("div", { class: "cite" }, `Rule ${id}`);
  const href = `/api/laws/${encodeURIComponent(cc)}/pdf#page=${encodeURIComponent(card.page)}`;
  return el("div", { class: "cite" },
    el("div", { class: "src" },
      el("strong", {}, card.section),
      el("span", {}, card.law),
      el("a", { href, target: "_blank", rel: "noopener" }, `PDF p. ${card.page}`),
      el("span", { class: "cat" }, `[${card.id}]`)),
    card.quote ? el("blockquote", {}, `“${card.quote}”`) : null,
    el("div", { class: "duty" }, `Platform duty: ${card.platform_duty}`),
    card.applies_to !== "platform" ? el("div", { class: "duty" }, `Penalty: ${card.penalty}`) : null,
  );
}

function renderDrift(d) {
  return el("details", { class: "drift", open: d.drift === "increased" },
    el("summary", {},
      el("span", { class: `badge drift-${d.drift}` }, `translation drift: ${d.drift}`),
      `${d.language} · ${fmtTime(d.start_s)}`),
    el("dl", {},
      el("dt", {}, "Original"), el("dd", {}, d.original),
      el("dt", {}, d.language), el("dd", {}, d.translated),
      el("dt", {}, "Back-translated"), el("dd", {}, d.back_translated),
      el("dt", {}, "Assessment"), el("dd", {}, d.explanation)),
  );
}

function downloadJson() {
  if (!state.job || !state.job.report) return;
  const blob = new Blob([JSON.stringify(state.job.report, null, 2)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: `streamclearance-${state.job.job_id}.json` });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

init();
