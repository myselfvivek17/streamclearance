// StreamClearance UI. Plain JS, no dependencies, no build step.
// All server/model text is inserted with textContent (via el()), never parsed as HTML.
"use strict";

const $ = (id) => document.getElementById(id);
const LS = { theme: "sc.theme", video: "sc.video", meta: "sc.videoMeta", job: "sc.lastJob", countries: "sc.countries" };
const POLL_MS = 2000;

const S = {
  status: null, countries: [], taxonomy: { leaves: {}, categories: {} },
  video: null, // upload metadata of the video still on the server (not yet analysed)
  local: null, // { id, url }: object URL of the file the user picked; the server deletes its copy after the check
  reportVideo: null, // upload metadata (from this browser) of the job being shown
  job: null, // last job fetched
  pollTimer: null, tickTimer: null, advanceTimer: null,
  seq: 0, // bumps on every route change so stale async renders are dropped
};

// ------------------------------------------------------------------ helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const icon = (name, style) => el("i", { class: `ic ic-${name}`, "aria-hidden": "true", style });
const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch { /* blocked */ } },
};

function fmt(s) {
  s = Math.max(0, Math.floor(Number(s) || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
const range = (a, b) => `${fmt(a)}–${fmt(b)}`;
const mb = (bytes) => `${Math.max(0.1, bytes / (1024 * 1024)).toFixed(1)} MB`;
const overlaps = (a, b) => (a.start_s < b.end_s && b.start_s < a.end_s) || a.start_s === b.start_s;

function joinWords(list) {
  if (list.length <= 1) return list.join("");
  return `${list.slice(0, -1).join(", ")} and ${list[list.length - 1]}`;
}

let toastTimer = null;
function toast(msg, sticky = false) {
  const t = $("toast");
  t.lastElementChild.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  if (!sticky) toastTimer = setTimeout(() => { t.hidden = true; }, 2800);
}

window.addEventListener("error", (e) => toast(`Something went wrong: ${e.message}`, true));
window.addEventListener("unhandledrejection", (e) => toast(`Something went wrong: ${(e.reason && e.reason.message) || e.reason}`, true));

// ------------------------------------------------------------------ labels

const RISK = {
  red: { cls: "risk-red", icon: "stop", verdict: "Don’t publish as is", level: "High risk", word: "Red", rank: 3, dot: "dot-red" },
  amber: { cls: "risk-amber", icon: "warn", verdict: "Fix recommended", level: "Medium risk", word: "Amber", rank: 2, dot: "dot-amber" },
  green: { cls: "risk-green", icon: "okc", verdict: "Looks clear", level: "Clear", word: "Green", rank: 0, dot: "dot-green" },
  grey: { cls: "risk-grey", icon: "help", verdict: "Check manually", level: "Check manually", word: "Check manually", rank: 1, dot: "dot-grey" },
  not_checked: { cls: "risk-grey", icon: "info", verdict: "Not checked: no verified law loaded", level: "Not checked", word: "Not checked", rank: 1, dot: "dot-grey" },
};
const NO_RISK = { cls: "", icon: "", verdict: "", level: "No issues", word: "", rank: -1, dot: "dot-none" };
const riskMeta = (r) => RISK[r === "check" ? "grey" : r] || NO_RISK;
const badgeOf = (c) => (c.badge === "check" ? "grey" : c.badge || c.overall_risk);

const LAW_STATUS = { in_force: ["In force", "inforce"], amended: ["As amended", "inforce"], guideline: ["Guideline", "guideline"], bill: ["Bill", "bill"] };
const lawStatus = (s) => LAW_STATUS[s] || [s, ""];
const LANG_CODE = { Hindi: "hi", Malay: "ms", "Mandarin Chinese": "zh", Urdu: "ur", Tamil: "ta" };
const DUTY = { creator: ["Creator"], advertiser: ["Advertiser"], platform: ["Platform"], both: ["Creator", "Platform"] };
const DUTY_TEXT = { creator: "Creator", advertiser: "Advertiser", platform: "Platform", both: "Creator and platform" };
const WHO_ORDER = ["Creator", "Advertiser", "Platform"];

const leafLabel = (x) => (x.leaf && S.taxonomy.leaves[x.leaf]) || S.taxonomy.categories[x.category] || String(x.leaf || x.category || "Sensitive content").replace(/_/g, " ");
const countryInfo = (cc) => S.countries.find((c) => c.code === cc) || { code: cc, name: cc, laws: [], drift_languages: [] };
const covLabel = (info) => (info.coverage === "deep" ? "Deep coverage" : "Basic coverage");
const covCls = (info) => (info.coverage === "deep" ? "tag tag-deep" : "tag");
const flag = (cc, lg = true) => el("span", { class: `flag${lg ? " flag-lg" : ""} flag-${cc}`, "aria-hidden": "true" });
const specified = (v) => v && !/^not specified/i.test(v);

// ------------------------------------------------------------------ API

function askAccessCode() {
  return new Promise((resolve) => {
    const dlg = $("access-dialog");
    const input = $("access-input");
    input.value = "";
    dlg.addEventListener("close", () => {
      const code = input.value.trim();
      if (code) document.cookie = `sc_access=${encodeURIComponent(code)}; path=/; max-age=${60 * 60 * 24 * 7}; SameSite=Strict`;
      resolve(!!code);
    }, { once: true });
    dlg.showModal();
    input.focus();
  });
}

class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function api(path, opts = {}) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const res = await fetch(path, { credentials: "same-origin", ...opts });
    if (res.status === 401) {
      if (!(await askAccessCode())) throw new ApiError("Access code required", 401);
      continue;
    }
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) throw new ApiError((body && body.detail && String(body.detail)) || `HTTP ${res.status}`, res.status);
    return body;
  }
  throw new ApiError("Access code rejected", 401);
}

let basics = null;
function loadBasics() {
  basics = basics || Promise.all([api("/api/status"), api("/api/countries"), api("/api/taxonomy")])
    .then(([status, countries, taxonomy]) => {
      S.status = status; S.countries = countries; S.taxonomy = taxonomy;
      if (status.mock) $("mock-banner").hidden = false;
    })
    .catch((e) => { basics = null; throw e; });
  return basics;
}

// File name, size and duration of the last upload stay in this browser, so a
// report can still name its video after the server deleted it.
function savedMeta(videoId) {
  try {
    const m = JSON.parse(store.get(LS.meta) || "null");
    return m && m.video_id === videoId ? m : null;
  } catch { return null; }
}

async function loadVideo(videoId) {
  if (S.video && S.video.video_id === videoId) return S.video;
  try {
    S.video = await api(`/api/videos/${encodeURIComponent(videoId)}/meta`);
    store.set(LS.meta, JSON.stringify(S.video));
    return S.video;
  } catch (e) {
    if (e.status === 404) return null;
    throw e;
  }
}

function setLocal(videoId, file) {
  if (S.local) URL.revokeObjectURL(S.local.url);
  S.local = file ? { id: videoId, url: URL.createObjectURL(file) } : null;
}
const localUrl = (videoId) => (S.local && S.local.id === videoId ? S.local.url : null);

const videoName = () => (S.reportVideo && S.reportVideo.filename) || "Your video";

// ------------------------------------------------------------------ shell: theme, stepper, routing

function setTheme(dark) {
  document.documentElement.classList.toggle("dark", dark);
  const btn = $("theme-btn");
  const label = dark ? "Switch to light mode" : "Switch to dark mode";
  btn.setAttribute("aria-label", label);
  btn.title = label;
  btn.firstElementChild.className = `ic ${dark ? "ic-sun" : "ic-moon"}`;
}

$("theme-btn").addEventListener("click", () => {
  const dark = !document.documentElement.classList.contains("dark");
  store.set(LS.theme, dark ? "dark" : "light");
  setTheme(dark);
});

function parseRoute() {
  const [name, id, cc] = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  return { name: name || "upload", id, cc: cc && cc.toUpperCase() };
}

function go(hash) {
  if (location.hash === hash) route(); else location.hash = hash;
}

function renderStepper(r) {
  const order = ["upload", "countries", "analysis", "results"];
  const labels = ["Upload", "Countries", "Analysis", "Results"];
  const cur = r.name === "country" ? 3 : Math.max(0, order.indexOf(r.name));
  const jobId = r.id || store.get(LS.job);
  const jobDone = S.job && S.job.job_id === jobId ? S.job.status === "done" : !!jobId;
  const target = {
    upload: "#upload",
    countries: S.video || store.get(LS.video) ? "#countries" : null,
    analysis: jobId ? `#analysis/${jobId}` : null,
    results: jobId && jobDone ? `#results/${jobId}` : null,
  };
  $("stepper").replaceChildren(...order.map((k, i) => {
    const isDone = i < cur;
    const isCur = i === cur && r.name !== "country";
    const href = target[k];
    const can = !!href && !isCur;
    return el("li", { class: `step ${isDone ? "done" : i === cur ? "current" : ""}` },
      i > 0 ? el("span", { class: "step-line", "aria-hidden": "true" }) : null,
      el("button", {
        class: `step-btn ${can ? "can" : ""}`, type: "button", disabled: !can && !isCur,
        "aria-current": i === cur ? "step" : null, onclick: () => { if (can) go(href); },
      },
      el("span", { class: "step-num" }, isDone ? [icon("check"), el("span", { class: "sr-only" }, "Done: ")] : String(i + 1)),
      el("span", { class: "step-label" }, labels[i])));
  }));
}

function clearTimers() {
  clearTimeout(S.pollTimer); clearInterval(S.tickTimer); clearTimeout(S.advanceTimer);
}

function mount(...nodes) {
  $("view").replaceChildren(...nodes);
}

async function route() {
  clearTimers();
  const seq = ++S.seq;
  const r = parseRoute();
  renderStepper(r);
  const stale = () => seq !== S.seq;
  try {
    await loadBasics();
    if (stale()) return;
    if (r.name === "countries") await screenCountries(stale);
    else if (r.name === "analysis" && r.id) await screenAnalysis(r.id, stale);
    else if (r.name === "results" && r.id) await screenResults(r.id, stale);
    else if (r.name === "country" && r.id && r.cc) await screenCountry(r.id, r.cc, stale);
    else await screenUpload(stale);
  } catch (e) {
    if (stale()) return;
    mount(el("main", { class: "main narrow" },
      el("div", { class: "alert risk-red", role: "alert" }, icon("warn"), el("span", {}, e.message || String(e))),
      el("div", { class: "footer-bar" }, el("button", { class: "btn", type: "button", onclick: () => go("#upload") }, icon("left"), "Start over"))));
  }
  if (!stale()) {
    renderStepper(r);
    const h1 = $("view").querySelector("h1");
    if (h1 && document.activeElement === document.body) { h1.tabIndex = -1; h1.focus({ preventScroll: true }); }
  }
}

window.addEventListener("hashchange", () => { window.scrollTo(0, 0); route(); });

function pageHead(eyebrow, title, lede) {
  return el("div", { class: "page-head" },
    el("span", { class: "eyebrow" }, eyebrow),
    el("h1", { class: "h1" }, title),
    lede ? el("p", { class: "lede" }, lede) : null);
}

const alertBox = (msg, cls = "risk-red", ic = "warn") =>
  el("div", { class: `alert ${cls}`, role: cls === "risk-red" ? "alert" : "note" }, icon(ic), el("span", {}, msg));

// ------------------------------------------------------------------ 1 · upload

// Only what the server says is true in this deployment (/api/status `privacy`).
function privacyNote() {
  const pv = S.status.privacy;
  if (!pv) return null;
  const days = pv.report_ttl_days;
  const parts = [
    pv.video_deleted_after_check ? "Your video is deleted right after the check." : null,
    days ? `Reports are deleted after ${plural(days, "day")}.` : null,
    pv.no_training ? "Content is never used to train models." : null,
  ].filter(Boolean);
  if (!parts.length) return null;
  return el("p", { class: "privacy-note" }, icon("shield"),
    el("span", {}, parts.join(" "), " ", el("a", { href: "/privacy" }, "Privacy")));
}

async function screenUpload(stale) {
  const saved = store.get(LS.video);
  if (saved && !S.video) await loadVideo(saved);
  if (stale()) return;
  if (saved && !S.video) store.set(LS.video, null);
  const maxMb = S.status.max_upload_mb;
  const maxMin = Math.round(S.status.max_duration_s / 60);

  const body = el("div");
  const main = el("main", { class: "main narrow" },
    pageHead("Step 1 of 4", "Check your video before you publish",
      "Upload a video and choose where you will post it. StreamClearance flags moments that could break local law, explains why, and suggests a fix."),
    body);
  mount(main);

  function showDrop(error) {
    const input = el("input", { type: "file", accept: ".mp4,.mov,.webm,video/mp4,video/quicktime,video/webm",
      onchange: () => input.files[0] && accept(input.files[0]) });
    const dz = el("div", { class: "dropzone" },
      el("div", { class: "dz-icon" }, icon("upload")),
      el("h2", { class: "h2" }, "Drag and drop your video here"),
      el("div", { class: "dz-or" }, "or"),
      el("label", { class: "btn btn-primary file-input-label" }, "Choose file", input),
      el("div", { class: "limits", style: "margin-top: 8px" },
        el("span", {}, icon("video"), "MP4, MOV or WebM"),
        el("span", {}, icon("file"), `Up to ${maxMb} MB`),
        el("span", {}, icon("clock"), `Up to ${plural(maxMin, "minute")}`)));
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("over"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault(); dz.classList.remove("over");
      const f = e.dataTransfer && e.dataTransfer.files[0];
      if (f) accept(f);
    });
    body.replaceChildren(...[dz, privacyNote(), error ? el("div", { style: "margin-top: 12px" }, alertBox(error)) : null].filter(Boolean));
  }

  function accept(f) {
    if (!/\.(mp4|mov|webm)$/i.test(f.name)) return showDrop("That file type isn’t supported. Upload an MP4, MOV or WebM video.");
    if (f.size > maxMb * 1024 * 1024) return showDrop(`That file is over ${maxMb} MB. Trim or compress it and try again.`);
    upload(f);
  }

  function upload(f) {
    const pct = el("span", { class: "mono small muted" }, "0%");
    const bar = el("span", { style: "width: 0%" });
    const prog = el("div", { class: "progress", role: "progressbar", "aria-label": "Upload progress",
      "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0" }, bar);
    const status = el("p", { class: "small muted" }, "Uploading…");
    body.replaceChildren(el("div", { class: "card card-pad stack", style: "gap: 14px", "aria-live": "polite" },
      el("div", { class: "row" }, icon("video", "color: var(--text-3)"), el("span", { class: "grow", style: "font-weight: 500" }, f.name), pct),
      prog, status));

    const form = new FormData();
    form.append("file", f);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const p = Math.round((100 * e.loaded) / e.total);
      bar.style.width = `${p}%`; pct.textContent = `${p}%`; prog.setAttribute("aria-valuenow", p);
      if (p >= 100) status.textContent = "Checking the video…";
    };
    xhr.onload = async () => {
      if (stale()) return;
      let res = null;
      try { res = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status === 401) {
        if (await askAccessCode()) upload(f); else showDrop("Access code required.");
        return;
      }
      if (xhr.status !== 200 || !res) {
        const detail = (res && res.detail) || `Upload failed (HTTP ${xhr.status}).`;
        const msg = xhr.status === 413 ? `That file is over ${maxMb} MB. Trim or compress it and try again.`
          : /limit is/.test(detail) ? `That video is too long: ${detail}. Trim it and try again.`
            : xhr.status === 415 ? "That file type isn’t supported. Upload an MP4, MOV or WebM video."
              : `Upload failed: ${detail}`;
        showDrop(msg);
        return;
      }
      S.video = res;
      store.set(LS.video, res.video_id);
      store.set(LS.meta, JSON.stringify(res));
      setLocal(res.video_id, f);
      showFile();
      renderStepper(parseRoute());
    };
    xhr.onerror = () => { if (!stale()) showDrop("Upload failed (network error). Check your connection and try again."); };
    xhr.send(form);
  }

  function showFile() {
    const v = S.video;
    const ext = (v.ext || "").replace(".", "").toUpperCase();
    const dur = v.duration_s != null ? fmt(v.duration_s) : "–";
    body.replaceChildren(
      el("div", { class: "card file-card" },
        el("div", { class: "thumb" },
          el("video", { src: `${localUrl(v.video_id) || `/api/videos/${encodeURIComponent(v.video_id)}`}#t=0.5`, preload: "metadata", muted: true, playsinline: true,
            "aria-hidden": "true", style: "position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover" }),
          el("span", { class: "thumb-dur" }, dur)),
        el("div", { class: "stack grow", style: "gap: 14px; justify-content: center; min-width: 220px" },
          el("div", { class: "row", style: "gap: 10px" }, el("span", { class: "badge risk-green" }, icon("okc"), "Uploaded")),
          el("div", { class: "h2", style: "word-break: break-all" }, v.filename),
          el("dl", { class: "meta-list", style: "margin: 0" },
            el("div", {}, el("dt", {}, "Duration"), el("dd", {}, dur)),
            el("div", {}, el("dt", {}, "Size"), el("dd", {}, mb(v.size_bytes))),
            el("div", {}, el("dt", {}, "Format"), el("dd", {}, ext))),
          el("div", {}, el("button", { class: "link-btn small", type: "button", onclick: () => {
            api(`/api/videos/${encodeURIComponent(v.video_id)}`, { method: "DELETE" }).catch(() => { /* the sweeper deletes it anyway */ });
            S.video = null; store.set(LS.video, null); store.set(LS.meta, null); setLocal(null); showDrop(); renderStepper(parseRoute());
          } }, "Replace video")))),
      el("div", { class: "footer-bar" },
        el("span", { class: "small muted" }, "Next, choose the countries you want to publish in."),
        el("button", { class: "btn btn-primary", type: "button", onclick: () => go("#countries") }, "Next", icon("right"))));
  }

  if (S.video) showFile(); else showDrop();
}

// ------------------------------------------------------------------ 2 · countries

async function screenCountries(stale) {
  const saved = store.get(LS.video);
  if (saved && !S.video) await loadVideo(saved);
  if (stale()) return;

  let selected;
  try { selected = new Set(JSON.parse(store.get(LS.countries) || "null")); } catch { selected = null; }
  if (!selected || !selected.size) {
    const deep = S.countries.filter((c) => c.coverage === "deep").map((c) => c.code);
    selected = new Set(deep.length ? deep : S.countries.map((c) => c.code));
  }
  for (const cc of [...selected]) if (!S.countries.some((c) => c.code === cc)) selected.delete(cc);

  const countLabel = el("span", { style: "font-weight: 600" });
  const names = el("span", { class: "small muted" });
  const analyse = el("button", { class: "btn btn-primary", type: "button", onclick: start }, "Analyse video", icon("right"));
  const msg = el("div", { style: "margin-top: 12px" });

  function update() {
    const codes = S.countries.map((c) => c.code).filter((c) => selected.has(c));
    countLabel.textContent = plural(codes.length, "country", "countries");
    names.textContent = codes.length ? codes.map((c) => countryInfo(c).name).join(", ") : "Select at least one country";
    analyse.disabled = !codes.length || !S.video;
    store.set(LS.countries, JSON.stringify(codes));
    return codes;
  }

  async function start() {
    const codes = update();
    if (!codes.length || !S.video) return;
    analyse.disabled = true;
    msg.replaceChildren();
    try {
      const { job_id } = await api("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: S.video.video_id, countries: codes }),
      });
      store.set(LS.job, job_id);
      S.job = null;
      // The server deletes the video when this check ends, so it can't be analysed again.
      S.video = null; store.set(LS.video, null);
      go(`#analysis/${job_id}`);
    } catch (e) {
      if (e.status === 404) { S.video = null; store.set(LS.video, null); }
      msg.replaceChildren(alertBox(e.status === 404 ? "The uploaded video is no longer available. Upload it again." : `Could not start the analysis: ${e.message}`));
      update();
    }
  }

  const cards = S.countries.map((c) => {
    const box = el("input", { type: "checkbox", checked: selected.has(c.code), onchange: () => {
      if (box.checked) selected.add(c.code); else selected.delete(c.code);
      card.classList.toggle("on", box.checked);
      update();
    } });
    const lawsLabel = c.coverage === "deep"
      ? `${plural(c.laws_count, "law")} · ${plural(c.provisions_count, "provision")}`
      : c.checked === false ? `${plural(c.laws_count, "law")} · none verified`
        : `${plural(c.laws_count, "law")} · basic rule cards`;
    const card = el("label", { class: `ccard ${selected.has(c.code) ? "on" : ""}` },
      box,
      el("div", { class: "row", style: "gap: 12px" },
        flag(c.code),
        el("div", { class: "stack grow" }, el("span", { class: "c-name" }, c.name), el("span", { class: "c-laws" }, lawsLabel)),
        el("span", { class: "check-box", "aria-hidden": "true" }, icon("check"))),
      el("div", { class: "tags" },
        c.checked === false ? el("span", { class: "tag risk-grey" }, icon("info"), "Not checked yet (no verified source)")
          : el("span", { class: covCls(c) }, covLabel(c)),
        c.bill ? el("span", { class: "tag tag-draft" }, icon("info"), "Bill, not yet enacted") : null,
        c.draft ? el("span", { class: "tag tag-draft" }, icon("draft"), "Draft rules: not yet reviewed by a lawyer") : null));
    return card;
  });

  mount(el("main", { class: "main" },
    pageHead("Step 2 of 4", "Where will you publish?",
      "Pick every country you plan to post in. We check your video against the laws loaded for each one."),
    S.video ? null : el("div", { style: "margin-bottom: 16px" },
      el("div", { class: "alert alert-grey", role: "note" }, icon("info"),
        el("span", {}, "Upload a video first. ", el("a", { href: "#upload" }, "Go to upload")))),
    el("fieldset", { style: "border: 0; padding: 0; margin: 0" },
      el("legend", { class: "sr-only" }, "Countries to check"),
      el("div", { class: "country-grid" }, cards)),
    el("div", { class: "sources" }, disclosure("Sources", "file", S.countries.map((c) =>
      el("div", { class: "sources-country" }, el("b", {}, c.name),
        el("ul", { class: "made-list" }, c.laws.map((l) => el("li", {}, el("span", {}, l.title), lawSource(l)))))))),
    el("div", { class: "card selection-bar" },
      el("div", { class: "stack", style: "gap: 2px" }, countLabel, names),
      el("div", { class: "row wrap" },
        el("button", { class: "btn btn-ghost", type: "button", onclick: () => go("#upload") }, icon("left"), "Back"),
        analyse)),
    msg,
    el("p", { class: "xsmall muted", style: "margin-top: 12px" },
      el("b", { style: "font-weight: 600" }, "Deep coverage"),
      " means the main laws and guidelines for video ads are loaded as checked provisions. ",
      el("b", { style: "font-weight: 600" }, "Basic coverage"),
      " means only general rule cards are loaded, so more moments will need a manual check.")));
  update();
}

// ------------------------------------------------------------------ 3 · analysis

// since > 0: fetch only the feed events after the first `since` (the rest are kept from S.job).
async function fetchJob(jobId, since = 0) {
  const prev = since && S.job && S.job.job_id === jobId ? (S.job.events || []).slice(0, since) : [];
  if (prev.length < since) since = 0;
  try {
    S.job = await api(`/api/jobs/${encodeURIComponent(jobId)}${since ? `?since=${since}` : ""}`);
    if (since) S.job.events = prev.concat(S.job.events || []);
  } catch (e) {
    if (e.status === 404) {
      if (store.get(LS.job) === jobId) store.set(LS.job, null);
      throw new Error(/expired|deleted/.test(e.message) ? `${e.message}. Start a new check.` : "That analysis was not found. It may have been deleted; start a new one.");
    }
    throw e;
  }
  if (S.job.mock) $("mock-banner").hidden = false;
  return S.job;
}

const STATUS_LABEL = { done: "Done", running: "Running", pending: "Pending", error: "Failed", skipped: "Skipped" };
const STATUS_ICON = { done: "check", running: "spin", error: "x", skipped: "right" };

function stageRow({ label, status, detail, sub, cc, dot }) {
  const ic = STATUS_ICON[status];
  return el("li", { class: `stage ${status}${sub ? " sub" : ""}` },
    el("span", { class: `st-icon st-${status}` }, ic ? icon(ic) : null),
    el("div", { class: "grow" },
      el("div", { class: "row", style: "gap: 8px" }, cc ? flag(cc, false) : null, el("span", { class: "st-label" }, label)),
      detail ? el("div", { class: "st-detail" }, dot ? el("span", { class: `dot ${dot}`, "aria-hidden": "true" }) : null, detail) : null),
    el("span", { class: "st-status" }, STATUS_LABEL[status] || status));
}

function buildStages(job) {
  const by = Object.fromEntries(job.stages.map((s) => [s.key, s]));
  const st = (k) => (by[k] ? by[k].status : "pending");
  const rows = [];
  const v = S.reportVideo;
  rows.push({ label: "Video uploaded", status: "done",
    detail: v ? [v.filename, mb(v.size_bytes), v.duration_s != null ? fmt(v.duration_s) : null].filter(Boolean).join(" · ") : "" });

  const inv = by.inventory || {};
  rows.push({ label: "Understanding the video (speech, visuals, on-screen text)", status: st("inventory"),
    detail: inv.status === "done" ? (inv.detail || "").replace(/segments?/, "moments").replace(", ", " · ")
      : inv.status === "running" ? "Transcribing speech and reading on-screen text…" : inv.detail });

  const cs = job.countries.map((cc) => st(`country:${cc}`));
  const lawsStatus = cs.every((s) => s === "done") ? "done" : cs.some((s) => s === "running") ? "running"
    : cs.every((s) => s === "error") ? "error" : cs.every((s) => s === "done" || s === "error") ? "done" : "pending";
  rows.push({ label: "Checking each country’s laws", status: lawsStatus });
  for (const cc of job.countries) {
    const s = by[`country:${cc}`] || { status: "pending", detail: "" };
    const info = countryInfo(cc);
    let detail = s.detail; let dot = null;
    if (s.status === "done") {
      const [badge, rest] = (s.detail || "").split(" · ");
      const m = riskMeta(badge);
      detail = [m.word || badge, rest].filter(Boolean).join(" · ");
      dot = m.dot;
    } else if (s.status === "running") {
      const [step, n] = [(s.detail || "").split(":")[0], ((s.detail || "").match(/\d+/) || [])[0]];
      detail = step === "assess" ? `Matching ${plural(Number(n), "sensitive moment")} to ${plural(info.laws_count || 1, "law")}…`
        : step === "critic" ? `Double-checking ${plural(Number(n), "finding")}…`
          : `Checking ${plural(info.laws_count || 1, "law")}…`;
    }
    else if (s.status === "pending") detail = "Waiting";
    rows.push({ label: info.name, cc, sub: true, status: s.status, detail, dot });
  }

  const langs = [...new Set(job.countries.flatMap((cc) => countryInfo(cc).drift_languages || []))];
  const tr = by.translation || {};
  rows.push({ label: `Translation check${langs.length ? ` (${langs.join(", ")})` : ""}`, status: st("translation"),
    detail: !langs.length ? "Not needed: English only"
      : tr.status === "skipped" ? `Not needed: ${tr.detail}`
        : tr.status === "running" ? "Translating and back-translating the spoken lines (runs alongside the country checks)…"
          : tr.status === "done" ? (tr.detail || "").replace(/^(\d+) line checks/, (_, n) => plural(Number(n), "spoken line") + " checked") : tr.detail });
  rows.push({ label: "Report ready", status: st("done"), detail: st("done") === "done" ? "Opening results…" : "" });
  return rows;
}

// ------------------------------------------------------------------ agent activity feed

// Every event is a real pipeline step (app/events.py); model and video text goes in via textContent.
const FEED = {
  watch: ["eye", "Watched"], search: ["globe", "Law search"], law: ["scale", "Law"], review: ["bot", "Review"],
  translate: ["lang", "Translation"], ai: ["shield", "AI-content check"], warn: ["warn", "Warning"], info: ["info", "Step"],
};

function feedItem(e) {
  const [ic, label] = FEED[e.kind] || FEED.info;
  return el("li", { class: `feed-item feed-${e.kind}` },
    el("span", { class: "feed-t", title: "Time since the check started" }, fmt(e.t)),
    el("span", { class: "feed-ic", title: label }, icon(ic), el("span", { class: "sr-only" }, `${label}: `)),
    el("span", { class: "feed-text" }, e.country ? flag(e.country, false) : null, e.text));
}

/** A list that appends new events and stays scrolled to the newest, unless the user scrolled up. */
function makeFeed({ live = true } = {}) {
  const list = el("ol", { class: "feed-list", role: live ? "log" : null, "aria-label": "Agent activity" });
  const empty = el("p", { class: "feed-empty" }, live ? "Waiting for the first step…" : "No steps were recorded for this report.");
  const box = el("div", { class: `feed ${live ? "feed-live" : ""}` }, empty, list);
  let shown = 0; let stick = true;
  box.addEventListener("scroll", () => { stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 12; });
  function update(evs) {
    evs = evs || [];
    if (evs.length < shown) { list.replaceChildren(); shown = 0; }
    for (const e of evs.slice(shown)) list.append(feedItem(e));
    shown = evs.length;
    empty.hidden = shown > 0;
    if (live && stick) box.scrollTop = box.scrollHeight;
  }
  return { node: box, update };
}

async function screenAnalysis(jobId, stale) {
  const job = await fetchJob(jobId);
  S.reportVideo = savedMeta(job.video_id);
  if (stale()) return;
  store.set(LS.job, jobId);

  const stages = el("ol", { class: "stages" });
  const feed = makeFeed();
  const elapsed = el("span", { class: "elapsed" });
  const foot = el("div", { class: "an-foot", "aria-live": "polite" });
  const err = el("div", { style: "margin-top: 16px" });
  mount(el("main", { class: "main narrow" },
    pageHead("Step 3 of 4", "Checking your video",
      "We are comparing what is said, shown and written on screen with each country’s rules. This usually takes 1–3 minutes."),
    el("div", { class: "card an-card" },
      el("div", { class: "an-head" },
        el("div", { class: "row", style: "gap: 10px" }, icon("video", "color: var(--text-3)"),
          el("span", { style: "font-weight: 600" }, videoName()),
          el("span", { class: "small muted" }, `· ${plural(job.countries.length, "country", "countries")}`)),
        elapsed),
      stages,
      el("div", { class: "feed-wrap" },
        el("div", { class: "feed-head" }, el("h2", { class: "h3" }, "Agent activity"),
          el("span", { class: "xsmall muted" }, "What the agent watched, searched and decided, as it happens")),
        feed.node),
      foot),
    err));

  const tick = () => {
    const j = S.job;
    const end = j.status === "running" || j.status === "queued" ? Date.now() / 1000 : j.updated_at;
    const label = fmt(end - j.created_at);
    elapsed.replaceChildren(icon("clock"), label);
    elapsed.setAttribute("aria-label", `Elapsed time ${label}`);
  };
  S.tickTimer = setInterval(tick, 1000);

  const draw = (j) => {
    stages.replaceChildren(...buildStages(j).map(stageRow));
    feed.update(j.events);
    tick();
    if (j.status === "done") {
      foot.replaceChildren(icon("okc"), el("span", { class: "grow" }, "Report ready. Opening results…"),
        el("button", { class: "btn btn-primary btn-sm", type: "button", onclick: () => go(`#results/${jobId}`) }, "View results"));
      clearInterval(S.tickTimer);
      S.advanceTimer = setTimeout(() => { if (!stale()) go(`#results/${jobId}`); }, 1200);
    } else if (j.status === "error" || j.status === "cancelled") {
      foot.replaceChildren(icon("x"), el("span", { class: "grow" }, "The analysis stopped."));
      err.replaceChildren(alertBox(`Analysis failed: ${j.error || "unknown error"} Your video was deleted from the server; upload it again to retry.`),
        el("div", { class: "footer-bar" },
          el("button", { class: "btn", type: "button", onclick: () => go("#upload") }, icon("upload"), "Upload again")));
      clearInterval(S.tickTimer);
    } else {
      foot.replaceChildren(icon("info"), el("span", {}, "Results open automatically when the report is ready."));
    }
    renderStepper(parseRoute());
  };
  draw(job);

  const poll = async () => {
    if (stale()) return;
    try {
      draw(await fetchJob(jobId, (S.job.events || []).length));
    } catch (e) {
      err.replaceChildren(alertBox(`Lost contact with the server, retrying: ${e.message}`, "alert-grey", "info"));
    }
    if (!stale() && (S.job.status === "running" || S.job.status === "queued")) S.pollTimer = setTimeout(poll, POLL_MS);
  };
  if (job.status === "running" || job.status === "queued") S.pollTimer = setTimeout(poll, POLL_MS);
}

// ------------------------------------------------------------------ report model

async function loadReport(jobId, stale) {
  const job = await fetchJob(jobId);
  if (stale()) return null;
  if (job.status !== "done" || !job.report) {
    location.replace(`#analysis/${encodeURIComponent(jobId)}`);
    return null;
  }
  S.reportVideo = savedMeta(job.video_id);
  store.set(LS.job, jobId);
  return job;
}

/** Timeline moments: stage A segments (fallback: the flagged intervals themselves). */
function moments(report) {
  const segs = (report.inventory && report.inventory.segments) || [];
  if (segs.length) return segs;
  const seen = new Map();
  for (const c of report.countries) {
    for (const x of [...c.findings, ...(c.uncovered || [])]) seen.set(`${x.start_s}-${x.end_s}`, { start_s: x.start_s, end_s: x.end_s });
  }
  return [...seen.values()].sort((a, b) => a.start_s - b.start_s);
}

function momentRisk(c, m) {
  let best = { risk: null, label: "" };
  for (const f of c.findings) {
    if (overlaps(f, m) && riskMeta(f.risk).rank > riskMeta(best.risk).rank) best = { risk: f.risk, label: leafLabel(f) };
  }
  if (best.risk) return best;
  const d = (c.translation || []).find((x) => x.drift === "increased" && overlaps(x, m));
  if (d) return { risk: "amber", label: `Riskier when translated (${d.language})` };
  const u = (c.uncovered || []).find((x) => overlaps(x, m));
  if (u) return { risk: "grey", label: `${leafLabel(u)}: check manually` };
  return best;
}

function worstRisk(countries, m) {
  let best = { risk: null, label: "" };
  for (const c of countries) {
    const r = momentRisk(c, m);
    if (riskMeta(r.risk).rank > riskMeta(best.risk).rank) best = r;
  }
  return best;
}

const segText = (s) => (s.transcript || "").trim();
const segScreen = (s) => (s.on_screen_text || "").trim().replace(/\s*\n\s*/g, " / ");

function quoteAt(report, x) {
  const s = moments(report).find((m) => overlaps(m, x));
  return s ? segText(s) || segScreen(s) : "";
}

// ------------------------------------------------------------------ video player + timeline

function makePlayer(job, scope, { legend = false } = {}) {
  const report = job.report;
  const ms = moments(report);
  let dur = Number(report.duration_s) || 0;
  // Plays the browser's own copy of the file: the server deleted its copy after the check.
  // Without one (e.g. after a reload) the timeline still works, just without the picture.
  const src = localUrl(job.video_id);
  const video = src ? el("video", { src, preload: "metadata", playsinline: true }) : null;
  let pos = 0; // current time when there is no video
  const now = () => (video ? video.currentTime || 0 : pos);
  const tc = el("span", { class: "player-tc" });
  const caption = el("div", { class: "caption", "aria-live": "off" });
  const big = el("button", { class: "play-big is-play", type: "button", "aria-label": "Play video" }, icon("play"));
  const small = el("button", { class: "play-sm", type: "button", "aria-label": "Play video", hidden: !video }, icon("play"));
  const head = el("div", { class: "tl-head" });
  const player = video
    ? el("div", { class: "player" }, video,
      el("span", { class: "player-file" }, videoName()), tc,
      el("div", { class: "player-center" }, big), caption)
    : el("div", { class: "player player-gone" },
      el("span", { class: "player-file" }, videoName()), tc,
      el("div", { class: "player-center" }, el("span", { class: "gone-note" }, icon("info"), "Video deleted after the check")), caption);

  const segs = [];
  for (const m of ms) {
    const w = worstRisk(scope, m);
    if (!w.risk || !dur) continue;
    const meta = riskMeta(w.risk);
    const aria = `Jump to ${range(m.start_s, m.end_s)}: ${w.label}, ${meta.level}`;
    const b = el("button", { class: `tl-seg ${w.risk === "check" ? "grey" : w.risk}`, type: "button", "aria-label": aria, title: aria,
      style: `left: ${(m.start_s / dur * 100).toFixed(2)}%; width: ${(Math.max(0.3, m.end_s - m.start_s) / dur * 100).toFixed(2)}%`,
      onclick: (e) => { e.stopPropagation(); seek(m.start_s); } });
    b.moment = m;
    segs.push(b);
  }
  const track = el("div", { class: "tl-track", role: "slider", tabindex: "0", "aria-label": "Video position",
    "aria-valuemin": "0", "aria-valuemax": String(Math.round(dur)) },
  el("div", { class: "tl-base" }), segs, head);
  track.addEventListener("click", (e) => {
    const rect = track.getBoundingClientRect();
    if (dur) seek(((e.clientX - rect.left) / rect.width) * dur, false);
  });
  track.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: -5, ArrowRight: 5, ArrowDown: -5, ArrowUp: 5 }[e.key];
    if (step) { e.preventDefault(); seek(now() + step, false); } else if (e.key === "Home") { e.preventDefault(); seek(0, false); } else if (e.key === "End") { e.preventDefault(); seek(dur, false); } else if (e.key === " " || e.key === "k") { e.preventDefault(); toggle(); }
  });

  const half = dur / 2;
  const timeline = el("div", { class: "timeline" },
    el("div", { class: "tl-controls" }, small, track),
    el("div", { class: "tl-ticks" }, el("span", {}, "0:00"), el("span", {}, fmt(half)), el("span", {}, fmt(dur))),
    legend ? el("div", { class: "tl-legend" },
      el("span", {}, el("i", { class: "legend-sw red", "aria-hidden": "true" }), "High risk"),
      el("span", {}, el("i", { class: "legend-sw amber", "aria-hidden": "true" }), "Medium risk (striped)"),
      el("span", {}, el("i", { class: "legend-sw grey", "aria-hidden": "true" }), "Check manually"),
      el("span", { class: "muted" }, "Select a marker to jump to that moment")) : null);

  const listeners = [];
  function update() {
    const t = now();
    head.style.left = `${dur ? (Math.min(dur, t) / dur * 100).toFixed(2) : 0}%`;
    tc.textContent = `${fmt(t)} / ${fmt(dur)}`;
    track.setAttribute("aria-valuenow", String(Math.round(t)));
    track.setAttribute("aria-valuetext", `${fmt(t)} of ${fmt(dur)}`);
    const cur = ms.find((m) => t >= m.start_s && t < m.end_s);
    caption.textContent = cur ? segText(cur) : "";
    for (const b of segs) b.classList.toggle("active", b.moment === cur);
    for (const fn of listeners) fn(t, cur);
  }
  function setPlaying(p) {
    player.classList.toggle("playing", p);
    for (const b of [big, small]) {
      b.setAttribute("aria-label", p ? "Pause video" : "Play video");
      b.classList.toggle("is-play", !p);
      b.firstElementChild.className = `ic ${p ? "ic-pause" : "ic-play"}`;
    }
  }
  function toggle() { if (!video) return; if (video.paused) video.play().catch(() => {}); else video.pause(); }
  function seek(t, play = true) {
    t = Math.max(0, Math.min(dur || t, t));
    if (video) video.currentTime = t; else pos = t;
    update();
    if (play && video) video.play().catch(() => {});
  }
  if (video) {
    big.addEventListener("click", toggle);
    small.addEventListener("click", toggle);
    video.addEventListener("click", toggle);
    video.addEventListener("timeupdate", update);
    video.addEventListener("seeked", update);
    video.addEventListener("play", () => setPlaying(true));
    video.addEventListener("pause", () => setPlaying(false));
    video.addEventListener("loadedmetadata", () => { if (!dur && video.duration) { dur = video.duration; update(); } });
    video.addEventListener("error", () => { caption.textContent = "Video not available (it was deleted after the check)."; });
  }
  update();
  return { player, timeline, seek, onTime: (fn) => listeners.push(fn) };
}

// ------------------------------------------------------------------ 4 · results

function reasonOf(c) {
  const b = badgeOf(c);
  if (c.findings.length) {
    const labels = [...new Set(c.findings.map(leafLabel))];
    const s = joinWords(labels.map((l, i) => (i ? l.charAt(0).toLowerCase() + l.slice(1) : l)));
    return s;
  }
  const riskier = [...new Set((c.translation || []).filter((d) => d.drift === "increased").map((d) => d.language))];
  if (riskier.length) return `A spoken line reads riskier in ${joinWords(riskier)}`;
  if (b === "grey") return "Sensitive content found, but no loaded law covers it";
  if (b === "not_checked") return "No verified law is loaded for this country, so nothing was checked";
  return "No moment matched a loaded law";
}

function reportNotes(job) {
  const r = job.report;
  const notes = [];
  if (r.mock) notes.push(alertBox("Mock mode: these results are canned examples, not a real analysis.", "alert-grey", "info"));
  if ((r.errors || []).length) notes.push(alertBox(`Some checks failed, so this report is incomplete: ${r.errors.join("; ")}`, "risk-amber", "warn"));
  return notes.length ? el("div", { class: "notes" }, notes) : null;
}

async function screenResults(jobId, stale) {
  const job = await loadReport(jobId, stale);
  if (!job || stale()) return;
  const r = job.report;
  const cs = r.countries;
  const failed = job.countries.filter((cc) => !cs.some((c) => c.country === cc));
  const errFor = (cc) => (r.errors || []).find((e) => e.startsWith(`${cc}:`));

  const needFix = cs.filter((c) => ["red", "amber"].includes(badgeOf(c))).length;
  const greyN = cs.filter((c) => badgeOf(c) === "grey").length;
  const notChecked = cs.filter((c) => badgeOf(c) === "not_checked").length;
  const n = cs.length;
  const checkedN = n - notChecked;
  const notCheckedPart = notChecked ? ` · ${notChecked} not checked (no verified law loaded)` : "";
  const headline = (needFix ? `${needFix} of ${n} ${n === 1 ? "country needs" : "countries need"} changes before you publish`
    : greyN ? `${greyN} of ${n} ${n === 1 ? "country needs" : "countries need"} a manual check`
      : !checkedN ? `Not checked: no verified law loaded for ${n === 1 ? "this country" : `these ${n} countries`}`
        : `Looks clear in ${checkedN === 1 ? "the 1 country" : `all ${checkedN} countries`} checked`) + (checkedN ? notCheckedPart : "");

  const p = makePlayer(job, cs, { legend: true });
  const ms = moments(r);
  const items = ms.map((m) => {
    const w = worstRisk(cs, m);
    const meta = riskMeta(w.risk);
    const speech = segText(m);
    const screen = segScreen(m);
    const btn = el("button", { class: "tr-item", type: "button", onclick: () => p.seek(m.start_s) },
      el("span", { class: "tr-time" }, range(m.start_s, m.end_s)),
      el("span", { class: `dot ${meta.dot}`, "aria-hidden": "true" }),
      el("span", { class: "tr-text" },
        speech ? `“${speech}”` : screen ? [el("span", { class: "tr-src" }, "On screen"), screen] : el("span", { class: "muted" }, m.visuals || "No speech or text"),
        el("span", { class: "tr-flag" }, w.risk ? `${meta.level} · ${w.label}` : "No issues")));
    btn.moment = m;
    return el("li", {}, btn);
  });
  p.onTime((t, cur) => { for (const li of items) li.firstElementChild.classList.toggle("active", li.firstElementChild.moment === cur && t > 0); });

  const cards = cs.map((c) => {
    const info = countryInfo(c.country);
    const meta = riskMeta(badgeOf(c));
    const nf = c.findings.length; const nu = (c.uncovered || []).length;
    const countLabel = c.overall_risk === "not_checked" ? "Not checked"
      : nf ? `${plural(nf, "issue")}${nu ? ` · ${nu} to check` : ""}` : nu ? `${nu} to check manually` : "Nothing to fix";
    const reason = reasonOf(c);
    return el("button", { class: "rcard", type: "button", "aria-label": `${info.name}: ${meta.verdict}. ${reason}. View details`,
      onclick: () => go(`#country/${jobId}/${c.country}`) },
    el("div", { class: "rcard-top" }, flag(c.country),
      el("span", { class: "stack grow" }, el("span", { class: "c-name" }, info.name), el("span", { class: "xsmall muted" }, covLabel(info)))),
    el("div", {}, el("span", { class: `badge badge-lg ${meta.cls}` }, icon(meta.icon), meta.verdict)),
    el("p", { class: "rcard-reason" }, reason),
    el("div", { class: "rcard-foot" }, el("span", {}, countLabel), el("span", { class: "go" }, "View details", icon("right"))));
  });
  for (const cc of failed) {
    const info = countryInfo(cc);
    const e = errFor(cc);
    cards.push(el("div", { class: "rcard", style: "cursor: default" },
      el("div", { class: "rcard-top" }, flag(cc),
        el("span", { class: "stack grow" }, el("span", { class: "c-name" }, info.name), el("span", { class: "xsmall muted" }, covLabel(info)))),
      el("div", {}, el("span", { class: "badge badge-lg risk-grey" }, icon("x"), "Check failed")),
      el("p", { class: "rcard-reason" }, e ? e.slice(cc.length + 1).trim() : "This country could not be checked. Run the analysis again."),
      el("div", { class: "rcard-foot" }, el("span", {}, "Not checked"))));
  }

  mount(el("main", { class: "main" },
    pageHead(`Results · ${videoName()}`, headline, "Select a country to see each problem moment, why it matters and how to fix it."),
    reportNotes(job),
    el("section", { class: "card player-wrap", "aria-label": "Video and problem moments" },
      el("div", { class: "player-col" }, p.player, p.timeline),
      el("div", { class: "transcript" },
        el("h2", { class: "h3" }, "What the video says"),
        el("ul", { class: "tr-list" }, items))),
    el("section", { class: "section", "aria-label": "Verdict by country" },
      el("div", { class: "section-head" }, el("h2", { class: "h2" }, "By country"),
        el("span", { class: "small muted" }, `${plural(job.countries.length - notChecked - failed.length, "country", "countries")} checked`)),
      el("div", { class: "results-grid" }, cards),
      el("div", { class: "disclaimer", style: "margin-top: 16px" }, icon("info"),
        el("span", {}, el("b", { style: "font-weight: 600" }, "Automated pre-check, not legal advice. "),
          "A person makes the publish decision."))),
    howReached(job),
    howMade(r),
    el("div", { class: "footer-bar" },
      el("button", { class: "btn", type: "button", onclick: () => go("#upload") }, icon("upload"), "Check another video"),
      el("button", { class: "btn", type: "button", onclick: () => download(job) }, icon("download"), "Download report (JSON)"),
      el("button", { class: "btn btn-danger", type: "button", onclick: () => deleteReport(job) }, icon("x"), "Delete this report now"))));
}

// "How the agent reached this": the same activity feed as on the analysis screen, collapsed.
function howReached(job) {
  const evs = job.events || [];
  if (!evs.length) return null;
  const feed = makeFeed({ live: false });
  feed.update(evs);
  return el("section", { class: "section", "aria-label": "How the agent reached this" },
    el("div", { class: "card", style: "padding: 0 18px" },
      disclosure(`How the agent reached this (${plural(evs.length, "step")})`, "bot", feed.node, { style: "border-top: 0" })));
}

// "How this report was made": laws (and their review status), model, app version.
function howMade(r) {
  const pv = r.provenance;
  if (!pv) return null;
  const m = pv.model;
  const when = (iso) => (iso ? new Date(iso).toLocaleString() : "unknown");
  const short = (h) => (h ? `${h.slice(0, 12)}…` : "unknown");
  const rows = [
    el("li", {}, el("b", {}, "Created: "), when(pv.created_at)),
    el("li", {}, el("b", {}, "Model: "), m.backend === "mock" ? "none (mock mode)"
      : `${m.smart}${m.fast !== m.smart ? ` / ${m.fast}` : ""}${m.location ? ` on Vertex AI, location ${m.location}` : ""}`),
    el("li", {}, el("b", {}, "App version: "), el("span", { class: "mono" }, pv.app_git_sha)),
    ...pv.laws.map((l) => el("li", {},
      el("b", {}, `${l.country} · ${l.title}: `),
      el("span", { class: `badge ${l.status === "reviewed" ? "risk-green" : "risk-grey"}` }, l.status === "reviewed" ? "Reviewed" : "Draft, not yet reviewed by a person"),
      ` · ${plural(l.provisions, "provision")}, built ${when(l.provisions_built)} · source `,
      el("span", { class: "mono", title: l.source_sha256 || "" }, `${l.source_file} sha256 ${short(l.source_sha256)}`),
      lawSource(l))),
  ];
  if (r.privacy && r.privacy.report_expires_at) {
    rows.push(el("li", {}, el("b", {}, "Video deleted: "), r.privacy.video_deleted_at ? when(r.privacy.video_deleted_at) : "not yet",
      " · ", el("b", {}, "Report will be deleted: "), when(r.privacy.report_expires_at)));
  }
  return el("section", { class: "section", "aria-label": "How this report was made" },
    el("div", { class: "card", style: "padding: 0 18px" },
      disclosure("How this report was made", "info", el("ul", { class: "made-list" }, rows), { style: "border-top: 0" })));
}

async function deleteReport(job) {
  if (!window.confirm("Delete this report now? It is removed from the server straight away and can’t be recovered.")) return;
  try {
    await api(`/api/jobs/${encodeURIComponent(job.job_id)}`, { method: "DELETE" });
  } catch (e) {
    if (e.status !== 404) { toast(`Could not delete the report: ${e.message}`, true); return; }
  }
  if (store.get(LS.job) === job.job_id) store.set(LS.job, null);
  if (savedMeta(job.video_id)) store.set(LS.meta, null);
  if (S.local && S.local.id === job.video_id) setLocal(null);
  S.job = null; S.reportVideo = null;
  go("#upload");
  toast("Report deleted");
}

function download(job) {
  const name = `streamclearance-${job.job_id}.json`;
  const blob = new Blob([JSON.stringify(job.report, null, 2)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  toast(`Downloaded ${name}`);
}

// ------------------------------------------------------------------ 5 · country detail

function disclosure(label, iconName, content, { open = false, style } = {}) {
  const chev = icon("chev");
  chev.classList.add("chev");
  const btn = el("button", { class: `disclosure ${open ? "open" : ""}`, type: "button", "aria-expanded": String(open), style },
    icon(iconName, "color: var(--text-3)"), el("span", { class: "grow" }, label), chev);
  const body = el("div", { hidden: !open }, content);
  btn.addEventListener("click", () => {
    const now = btn.getAttribute("aria-expanded") !== "true";
    btn.setAttribute("aria-expanded", String(now));
    btn.classList.toggle("open", now);
    body.hidden = !now;
  });
  return [btn, body];
}

// Where a law's text comes from: publisher, version and the official file it was taken from.
function lawSource(l) {
  if (!l || !l.source_url) return null;
  return el("span", { class: "law-source" },
    l.source_verified ? null : el("span", { class: "tag tag-draft" }, icon("info"), "Source not verified: not used in checks"),
    `Source: ${[l.publisher, l.version].filter(Boolean).join(", ")} · `,
    el("a", { href: l.source_url, target: "_blank", rel: "noopener noreferrer" }, "official file"),
    el("span", { class: "sr-only" }, ` of ${l.title}`));
}

const pdfLink = (href, text, srText) => el("a", { class: "pdf-link", href, target: "_blank", rel: "noopener" },
  icon("file"), text, srText ? el("span", { class: "sr-only" }, srText) : null);

function provisionCard(cc, p, law) {
  const [st, stCls] = lawStatus(p.law_status);
  const k = p.consequence || {};
  const href = `/api/laws/${encodeURIComponent(cc)}/${encodeURIComponent(p.law_id)}/pdf#page=${encodeURIComponent(p.page)}`;
  const ex = p.exceptions || [];
  return el("div", { class: "prov" },
    el("div", { class: "prov-head" },
      el("div", { class: "stack", style: "gap: 2px" },
        el("span", { class: "prov-title" }, p.law_title),
        el("span", { class: "prov-sec" }, `${p.section} · page ${p.page}`)),
      el("div", { class: "row wrap", style: "gap: 6px" },
        el("span", { class: `status-pill ${stCls}` }, st),
        p.review && p.review.status !== "reviewed" ? el("span", { class: "tag tag-draft" }, icon("draft"), "Draft, not reviewed") : null,
        pdfLink(href, `Open PDF at p. ${p.page}`, ` of ${p.law_title}`))),
    el("blockquote", { class: "vquote" }, p.verbatim_text),
    el("dl", { class: "kv", style: "margin: 0" },
      p.summary ? [el("dt", {}, "In short"), el("dd", {}, p.summary)] : null,
      specified(k.penalty) ? [el("dt", {}, "Penalty"), el("dd", {}, k.penalty)] : null,
      specified(k.kind) || specified(k.timeframe) ? [el("dt", {}, "Consequence"), el("dd", {}, [k.kind, k.timeframe].filter(specified).map((x) => x.replace(/_/g, " ")).join(", "))] : null,
      [el("dt", {}, "Applies to"), el("dd", {}, DUTY_TEXT[p.duty_holder] || p.duty_holder)],
      ex.length ? [el("dt", {}, "Exceptions"), el("dd", {}, ex.length === 1 ? ex[0] : el("ul", { style: "margin: 0; padding-left: 18px" }, ex.map((x) => el("li", {}, x))))] : null),
    lawSource(law));
}

function ruleCardCard(cc, card, law) {
  const href = `/api/laws/${encodeURIComponent(cc)}/pdf#page=${encodeURIComponent(card.page)}`;
  return el("div", { class: "prov" },
    el("div", { class: "prov-head" },
      el("div", { class: "stack", style: "gap: 2px" },
        el("span", { class: "prov-title" }, card.law),
        el("span", { class: "prov-sec" }, `${card.section} · page ${card.page}`)),
      el("div", { class: "row wrap", style: "gap: 6px" },
        el("span", { class: "status-pill" }, "Basic rule card"),
        card.needs_review ? el("span", { class: "tag tag-draft" }, icon("draft"), "Draft, not reviewed") : null,
        pdfLink(href, `Open PDF at p. ${card.page}`, ` of ${card.law}`))),
    card.quote ? el("blockquote", { class: "vquote" }, card.quote) : null,
    el("dl", { class: "kv", style: "margin: 0" },
      el("dt", {}, "In short"), el("dd", {}, card.conduct),
      specified(card.platform_duty) ? [el("dt", {}, "Platform duty"), el("dd", {}, card.platform_duty)] : null,
      specified(card.penalty) ? [el("dt", {}, "Penalty"), el("dd", {}, card.penalty)] : null,
      el("dt", {}, "Applies to"), el("dd", {}, DUTY_TEXT[card.applies_to] || card.applies_to)),
    lawSource(law));
}

function liableOf(f, provs, cards) {
  const who = new Set();
  for (const id of f.rule_card_ids) {
    if (provs[id]) (DUTY[provs[id].duty_holder] || []).forEach((w) => who.add(w));
    else if (cards[id]) (DUTY[cards[id].applies_to] || []).forEach((w) => who.add(w));
  }
  if (f.creator_liability) who.add("Creator");
  return WHO_ORDER.filter((w) => who.has(w));
}

function issueCard(job, c, f, p, provs, cards) {
  const m = riskMeta(f.risk);
  const label = leafLabel(f);
  const quote = quoteAt(job.report, f);
  const chev = icon("chev");
  chev.classList.add("chev");
  const toggleBtn = el("button", { class: "issue-toggle", type: "button", "aria-expanded": "false" },
    el("span", { class: `badge ${m.cls}` }, icon(m.icon), m.level),
    el("span", { class: "issue-title" }, el("b", {}, label), quote ? el("span", {}, `“${quote}”`) : null),
    chev);
  const preview = el("div", { class: "fix-preview" }, icon("wrench", "color: var(--accent)"), el("b", {}, "Fix:"), el("span", {}, f.suggested_fix));

  const laws = Object.fromEntries([...countryInfo(c.country).laws.map((l) => [l.id, l]), ...(c.laws || []).map((l) => [l.law_id, l])]);
  const legal = f.rule_card_ids.map((id) => (provs[id] ? provisionCard(c.country, provs[id], laws[provs[id].law_id])
    : cards[id] ? ruleCardCard(c.country, cards[id], (c.laws || [])[0]) : null)).filter(Boolean);
  const liable = liableOf(f, provs, cards);
  const ownDuties = (c.platform_obligations || []).filter((o) => (f.obligation_ids || []).includes(o.id));
  const body = el("div", { class: "issue-body", hidden: true },
    el("div", { class: "stack", style: "gap: 4px" }, el("span", { class: "eyebrow" }, "Why it’s a problem"), el("p", { style: "text-wrap: pretty" }, f.rationale)),
    el("div", { class: "fix" }, el("span", { class: "fix-icon" }, icon("wrench")),
      el("div", {}, el("div", { class: "fix-label" }, "How to fix"), el("p", { class: "fix-text" }, f.suggested_fix))),
    liable.length ? el("dl", { class: "kv", style: "margin: 0" },
      el("dt", {}, "Who’s liable"),
      el("dd", { class: "liable" }, liable.map((w) => el("span", { class: "tag on" }, icon("user"), w)))) : null,
    f.critic_note ? el("div", { class: "note" }, icon("bot"), el("span", {}, el("b", {}, "Reviewer AI note: "), f.critic_note)) : null,
    f.exception_applies ? el("div", { class: "note" }, icon("info"), el("span", {}, el("b", {}, "An exception may apply: "), f.exception_reason || "see the exceptions in the legal basis.")) : null,
    ownDuties.length ? el("div", { class: "note" }, icon("monitor"), el("span", {}, el("b", {}, "Platforms must also act: "),
      ownDuties.map((o) => `${o.section} (${o.law_id})`).join(", "), ". See Platform obligations below.")) : null,
    legal.length ? el("div", { class: "legal" }, disclosure(`Legal basis · ${plural(legal.length, "provision")}`, "scale", legal)) : null);

  const art = el("article", { class: "issue" },
    el("div", { class: "issue-head" },
      el("button", { class: "chip-time", type: "button", "aria-label": `Play ${range(f.start_s, f.end_s)}`, onclick: () => p.seek(f.start_s) },
        icon("play"), range(f.start_s, f.end_s)),
      toggleBtn),
    preview, body);
  toggleBtn.addEventListener("click", () => {
    const open = toggleBtn.getAttribute("aria-expanded") !== "true";
    toggleBtn.setAttribute("aria-expanded", String(open));
    art.classList.toggle("open", open);
    body.hidden = !open; preview.hidden = open;
  });
  return art;
}

function translationSection(c, info, p) {
  const langs = info.drift_languages || [];
  const lines = c.translation || [];
  const shown = [...new Set(lines.map((d) => d.language))];
  const lang = shown.length > 1 ? shown.join(", ") : shown[0] || langs.join(", ");
  const head = el("div", { class: "section-head" },
    el("h2", { class: "h2" }, el("span", { class: "row" }, icon("lang", "width: 18px; height: 18px; color: var(--text-3)"),
      `Translation check${lang ? ` · ${lang}` : ""}`)),
    el("span", { class: "small muted" }, "Does the meaning get riskier in the local language?"));
  const sec = (...kids) => el("section", { class: "section", "aria-label": "Translation check" }, head, kids);
  if (!langs.length) return sec(el("p", { class: "small muted" }, `Not run. Videos for ${info.name} are checked in English.`));
  if (!lines.length) {
    return sec(el("p", { class: "small muted" }, c.translation_note || "No spoken lines were checked."));
  }
  const riskier = lines.filter((d) => d.drift === "increased");
  let all = !riskier.length;
  const rowsBox = el("div");
  const note = el("span");
  const more = el("button", { class: "link-btn small", type: "button" });
  const draw = () => {
    rowsBox.replaceChildren(...(all ? lines : riskier).map((d) => el("div", { class: "tr-row" },
      el("div", {}, el("span", { class: "lbl" }, "Time"),
        el("button", { class: "chip-time", type: "button", "aria-label": `Play ${range(d.start_s, d.end_s)}`, onclick: () => p.seek(d.start_s) }, icon("play"), range(d.start_s, d.end_s))),
      el("div", {}, el("span", { class: "lbl" }, "Original"), d.original),
      el("div", {}, el("span", { class: "lbl" }, d.language), el("span", { class: "tr-native", lang: LANG_CODE[d.language] || null }, d.translated)),
      el("div", {}, el("span", { class: "lbl" }, "Back-translation"), el("span", { class: `tr-back ${d.drift === "increased" ? "changed" : ""}` }, d.back_translated)),
      el("div", {}, el("span", { class: "lbl" }, "Result"),
        d.drift === "increased" ? el("span", { class: "badge risk-amber" }, icon("warn"), "Riskier")
          : d.drift === "reduced" ? el("span", { class: "tag" }, icon("info"), "Less risky") : el("span", { class: "tag" }, icon("check"), "Same meaning"),
        d.explanation ? el("div", { class: "tr-why" }, d.explanation) : null))));
    note.textContent = all ? `Showing all ${plural(lines.length, "checked line")}.`
      : `${riskier.length} of ${plural(lines.length, "line")} read riskier. The other ${lines.length - riskier.length} kept their meaning.`;
    more.textContent = all ? "Show only riskier lines" : "Show all lines";
    more.hidden = !riskier.length || riskier.length === lines.length;
  };
  more.addEventListener("click", () => { all = !all; draw(); });
  draw();
  return sec(c.translation_note ? el("p", { class: "small muted" }, c.translation_note) : null, el("div", { class: "card tr-grid" },
    el("div", { class: "tr-row head" }, el("span", {}, "Time"), el("span", {}, "Original"), el("span", {}, shown.length > 1 ? "Translation" : lang), el("span", {}, "Back-translation"), el("span", {}, "Result")),
    rowsBox, el("div", { class: "tr-more" }, note, more)));
}

async function screenCountry(jobId, cc, stale) {
  const job = await loadReport(jobId, stale);
  if (!job || stale()) return;
  const c = job.report.countries.find((x) => x.country === cc);
  if (!c) { location.replace(`#results/${encodeURIComponent(jobId)}`); return; }
  const info = countryInfo(cc);
  const meta = riskMeta(badgeOf(c));
  const provs = Object.fromEntries((c.cited_provisions || []).map((x) => [x.id, x]));
  const cards = Object.fromEntries((c.cited_cards || []).map((x) => [x.id, x]));
  const laws = (c.laws || []).length ? c.laws : info.laws || [];
  const nf = c.findings.length;
  const unc = c.uncovered || [];
  const obl = c.platform_obligations || [];
  const p = makePlayer(job, [c]);
  const back = () => go(`#results/${jobId}`);

  const lawList = el("ul", { class: "law-list" }, laws.map((l) => {
    const [st, stCls] = lawStatus(l.status);
    return el("li", { class: "law" },
      el("span", { class: "law-title" }, l.title),
      el("span", { class: "law-meta" }, el("span", { class: `status-pill ${stCls}` }, st), pdfLink(l.pdf_url, "PDF", ` of ${l.title}`)),
      l.label ? el("span", { class: "law-note" }, l.label) : null,
      lawSource(l));
  }));

  const notes = [
    ...(c.notes || []),
    c.dropped_findings ? `${plural(c.dropped_findings, "model suggestion")} dropped by the citation ${c.brain === "adk" ? "verifier" : "validator"} (the cited text did not fit).` : null,
    c.critic_changes ? `The reviewer AI changed ${plural(c.critic_changes, "finding")}.` : null,
  ].filter(Boolean);

  const notChecked = c.overall_risk === "not_checked";
  const verdictSub = notChecked
    ? "No verified law is loaded for this country, so nothing was checked. This is not a clearance: check the video manually."
    : meta === RISK.grey
    ? "Sensitive content found, but no loaded law covers it. Check these moments manually."
    : nf ? `${reasonOf(c)}.` : "No moment matched a loaded law.";

  mount(el("main", { class: "main" },
    el("button", { class: "back", type: "button", onclick: back }, icon("left"), "All countries"),
    el("div", { class: "detail-top", style: "margin-top: 8px" },
      el("section", { class: "card detail-head", "aria-label": "Verdict" },
        el("div", { class: "dh-title" }, flag(cc), el("h1", { class: "h1" }, info.name),
          el("span", { class: covCls(info) }, covLabel(info)),
          c.rules_status !== "reviewed" ? el("span", { class: "tag tag-draft" }, icon("draft"), "Draft rules: not yet reviewed by a lawyer") : null),
        el("div", { class: "stack", style: "gap: 8px" },
          el("div", {}, el("span", { class: `badge badge-lg ${meta.cls}` }, icon(meta.icon), meta.verdict)),
          el("p", { class: "rcard-reason" }, verdictSub),
          c.risk_framing ? el("p", { class: "small muted" }, c.risk_framing) : null),
        el("div", { class: "stat-row" },
          el("div", { class: "stat" }, el("b", {}, String(nf)), el("span", {}, "Issues to fix")),
          el("div", { class: "stat" }, el("b", {}, String(unc.length)), el("span", {}, "To check manually")),
          el("div", { class: "stat" }, el("b", {}, String(laws.length)), el("span", {}, "Laws checked"))),
        el("div", {}, disclosure(`Laws checked (${laws.length})`, "scale", lawList))),
      el("section", { class: "card mini-player", "aria-label": "Video" }, p.player, p.timeline)),

    notes.length ? el("div", { class: "notes", style: "margin-top: 20px" }, notes.map((n) => alertBox(n, "alert-grey", "info"))) : null,

    el("section", { class: "section", "aria-label": "Issues" },
      el("div", { class: "section-head" }, el("h2", { class: "h2" }, `Issues to fix (${nf})`),
        nf ? el("span", { class: "small muted" }, "Open an issue for the legal basis") : null),
      nf ? c.findings.map((f) => issueCard(job, c, f, p, provs, cards))
        : el("div", { class: "neutral-card" }, icon("info", "width: 18px; height: 18px; color: var(--text-3); margin-top: 2px"),
          el("div", {}, el("div", { class: "h3" }, notChecked ? "Nothing was checked" : "No moment matched a loaded law"),
            el("p", { class: "small muted", style: "margin-top: 2px" }, notChecked
              ? `No verified law is loaded for ${info.name}. Check the video manually before publishing there.`
              : `This does not mean the video is safe here. Coverage for ${info.name} is ${info.coverage || "basic"}${unc.length ? ", so see the moments to check manually below." : "."}`)))),

    el("section", { class: "section", "aria-label": "Not covered by any loaded law" },
      el("div", { class: "section-head" }, el("h2", { class: "h2" }, `Not covered by any loaded law (${unc.length})`),
        el("span", { class: "small muted" }, "Sensitive, but nothing we loaded applies")),
      unc.length ? el("div", { class: "stack", style: "gap: 10px" }, unc.map((u) => el("div", { class: "neutral-card" },
        el("button", { class: "chip-time", type: "button", "aria-label": `Play ${range(u.start_s, u.end_s)}`, onclick: () => p.seek(u.start_s) }, icon("play"), range(u.start_s, u.end_s)),
        el("div", { class: "stack grow", style: "gap: 6px" },
          el("div", { class: "row wrap" }, el("span", { class: "badge risk-grey" }, icon("help"), "Check manually"), el("span", { class: "h3" }, leafLabel(u))),
          el("p", { class: "small", style: "color: var(--text-2)" }, u.note || "The loaded law library does not cover this moment. Check it manually.")))))
        : el("p", { class: "small muted" }, "Nothing to check manually.")),

    translationSection(c, info, p),

    el("section", { class: "section", "aria-label": "Platform obligations" },
      el("div", { class: "card", style: "padding: 0 18px" },
        disclosure([`Platform obligations (${obl.length}) `,
          el("span", { class: "small muted", style: "font-weight: 400" }, "· what YouTube, Instagram and others must do. Not your action items.")], "monitor",
        el("div", { style: "padding-bottom: 8px" }, obl.length ? obl.map((o) => el("div", { class: "oblig" }, icon("monitor"),
          el("div", { class: "stack", style: "gap: 2px" },
            el("span", { class: "small", style: "font-weight: 600" }, `${o.law_title} · `, el("span", { class: "mono" }, o.section)),
            el("span", { class: "small", style: "color: var(--text-2)" }, o.summary || o.verbatim_text),
            el("span", {}, pdfLink(`/api/laws/${encodeURIComponent(cc)}/${encodeURIComponent(o.law_id)}/pdf#page=${encodeURIComponent(o.page)}`, `PDF p. ${o.page}`, ` of ${o.law_title}`)))))
          : el("p", { class: "small muted", style: "padding: 4px 0 12px" }, c.brain === "adk"
            ? "No platform duties are triggered by these findings."
            : "Not available with basic coverage: platform duties are only listed for countries with deep coverage.")),
        { style: "border-top: 0" }))),

    el("div", { class: "disclaimer", style: "margin-top: 24px" }, icon("info"),
      el("span", {}, el("b", { style: "font-weight: 600" }, "Advisory only, not legal advice. "),
        "Quotes come from the loaded law PDFs. Confirm with a qualified lawyer before relying on them.")),
    el("div", { class: "footer-bar" },
      el("button", { class: "btn", type: "button", onclick: back }, icon("left"), "Back to overview"),
      el("button", { class: "btn", type: "button", onclick: () => download(job) }, icon("download"), "Download report (JSON)"))));
}

// ------------------------------------------------------------------ start

setTheme(document.documentElement.classList.contains("dark"));
route();
