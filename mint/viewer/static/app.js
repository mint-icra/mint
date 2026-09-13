const state = {
  samples: [],
  selected: null,
  job: null,
  pollTimer: null,
  sourceUrl: null,
  predictionUrl: null,
  trajectory: null,
  yaw: -0.55,
  pitch: 0.42,
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  status: $("#server-status"),
  sampleList: $("#sample-list"),
  sampleCount: $("#sample-count"),
  emptySamples: $("#empty-samples"),
  activeTitle: $("#active-title"),
  video: $("#video"),
  stagePlaceholder: $("#stage-placeholder"),
  stageTag: $("#stage-tag"),
  sourceView: $("#source-view"),
  predictionView: $("#prediction-view"),
  run: $("#run-button"),
  cancel: $("#cancel-button"),
  download: $("#download-link"),
  drawer: $("#job-drawer"),
  stage: $("#job-stage"),
  detail: $("#job-detail"),
  progress: $("#progress-bar"),
  progressValue: $("#progress-value"),
  frames: $("#metric-frames"),
  fps: $("#metric-fps"),
  canvas: $("#trajectory-canvas"),
  canvasPlaceholder: $("#canvas-placeholder"),
  toast: $("#toast"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { message = (await response.json()).error || message; } catch (_) { /* no JSON body */ }
    throw new Error(message);
  }
  return response.json();
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { elements.toast.hidden = true; }, 7000);
}

function renderSamples() {
  elements.sampleList.replaceChildren();
  elements.sampleCount.textContent = `${state.samples.length} clip${state.samples.length === 1 ? "" : "s"}`;
  elements.emptySamples.hidden = state.samples.length > 0;
  elements.sampleList.hidden = state.samples.length === 0;
  state.samples.forEach((sample, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `sample-card${state.selected?.id === sample.id ? " active" : ""}`;
    card.setAttribute("role", "listitem");
    card.innerHTML = `
      <span class="number">${String(index + 1).padStart(2, "0")}</span>
      <strong></strong>
      <span class="sample-meta"><span></span><span></span></span>`;
    card.querySelector("strong").textContent = sample.title;
    const meta = card.querySelectorAll(".sample-meta span");
    meta[0].textContent = sample.collection;
    meta[1].textContent = `${sample.size_mb.toFixed(1)} MB`;
    card.addEventListener("click", () => selectSample(sample));
    elements.sampleList.append(card);
  });
}

function selectSample(sample) {
  if (state.job && ["queued", "running"].includes(state.job.status)) return;
  state.selected = sample;
  state.sourceUrl = `/media/${sample.id}`;
  state.predictionUrl = null;
  state.trajectory = null;
  elements.activeTitle.textContent = sample.title;
  elements.video.src = state.sourceUrl;
  elements.video.load();
  elements.stagePlaceholder.hidden = true;
  elements.run.disabled = false;
  elements.predictionView.disabled = true;
  elements.download.hidden = true;
  elements.frames.textContent = "--";
  elements.fps.textContent = "--";
  elements.canvasPlaceholder.hidden = false;
  setVideoView("source");
  renderSamples();
  drawTrajectory();
}

function setVideoView(view) {
  const prediction = view === "prediction" && state.predictionUrl;
  elements.sourceView.classList.toggle("active", !prediction);
  elements.predictionView.classList.toggle("active", Boolean(prediction));
  elements.stageTag.textContent = prediction ? "PREDICTION" : "SOURCE";
  const time = elements.video.currentTime || 0;
  elements.video.src = prediction ? state.predictionUrl : state.sourceUrl;
  elements.video.load();
  elements.video.addEventListener("loadedmetadata", () => {
    if (time && time < elements.video.duration) elements.video.currentTime = time;
  }, { once: true });
}

async function startJob() {
  if (!state.selected) return;
  elements.run.disabled = true;
  elements.cancel.hidden = false;
  elements.drawer.hidden = false;
  elements.download.hidden = true;
  updateProgress({ stage: "Submitting", progress: 0, status: "queued" });
  try {
    state.job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ sample_id: state.selected.id }),
    });
    pollJob();
  } catch (error) {
    finishJob(false, error.message);
  }
}

function updateProgress(job) {
  const percent = Math.max(0, Math.min(100, Math.round((job.progress || 0) * 100)));
  elements.stage.textContent = job.stage || "Working";
  elements.detail.textContent = job.frames ? `${job.frames} frames / local processing` : "Private local processing";
  elements.progress.style.width = `${percent}%`;
  elements.progressValue.textContent = `${percent}%`;
}

async function pollJob() {
  if (!state.job) return;
  try {
    const job = await api(`/api/jobs/${state.job.id}`);
    state.job = job;
    updateProgress(job);
    if (["queued", "running"].includes(job.status)) {
      state.pollTimer = window.setTimeout(pollJob, 700);
      return;
    }
    if (job.status === "completed") {
      state.predictionUrl = job.artifacts.video;
      elements.predictionView.disabled = false;
      elements.download.href = job.artifacts.prediction;
      elements.download.hidden = false;
      elements.frames.textContent = String(job.frames ?? "--");
      elements.fps.textContent = job.fps ? `${Number(job.fps).toFixed(1)} FPS` : "--";
      state.trajectory = await api(job.artifacts.trajectory);
      elements.canvasPlaceholder.hidden = true;
      setVideoView("prediction");
      drawTrajectory();
      finishJob(true);
    } else {
      finishJob(false, job.error || (job.status === "cancelled" ? "Prediction cancelled." : "Prediction failed."));
    }
  } catch (error) {
    finishJob(false, error.message);
  }
}

function finishJob(success, error = null) {
  window.clearTimeout(state.pollTimer);
  elements.cancel.hidden = true;
  elements.run.disabled = !state.selected;
  window.setTimeout(() => { elements.drawer.hidden = true; }, success ? 900 : 0);
  if (error) showToast(error);
}

async function cancelJob() {
  if (!state.job) return;
  try {
    await api(`/api/jobs/${state.job.id}`, { method: "DELETE" });
    elements.stage.textContent = "Cancelling";
  } catch (error) {
    showToast(error.message);
  }
}

function normalizePoints(series) {
  const points = series.flatMap((item) => item.points);
  if (!points.length) return { series, scale: 1, center: [0, 0, 0] };
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  points.forEach((point) => point.forEach((value, axis) => {
    min[axis] = Math.min(min[axis], value);
    max[axis] = Math.max(max[axis], value);
  }));
  const center = min.map((value, axis) => (value + max[axis]) / 2);
  const extent = Math.max(...max.map((value, axis) => value - min[axis]), 0.01);
  return { series, scale: 1.65 / extent, center };
}

function projectPoint(point, width, height, normalized) {
  let [x, y, z] = point.map((value, axis) => (value - normalized.center[axis]) * normalized.scale);
  const cy = Math.cos(state.yaw); const sy = Math.sin(state.yaw);
  [x, z] = [x * cy - z * sy, x * sy + z * cy];
  const cp = Math.cos(state.pitch); const sp = Math.sin(state.pitch);
  [y, z] = [y * cp - z * sp, y * sp + z * cp];
  const perspective = 2.9 / Math.max(1.2, 3.1 + z);
  return [width * .5 + x * width * .34 * perspective, height * .53 - y * width * .34 * perspective, z];
}

function drawTrajectory() {
  const canvas = elements.canvas;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const width = rect.width; const height = rect.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#183f36";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "rgba(241,238,229,.08)";
  context.lineWidth = 1;
  for (let index = 1; index < 8; index += 1) {
    const y = height * index / 8;
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
    const x = width * index / 8;
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
  }
  if (!state.trajectory) return;

  const trajectory = state.trajectory;
  const series = [
    { points: trajectory.camera, color: "#d9f252", width: 2.4 },
    { points: trajectory.left_wrist.filter((_, i) => trajectory.left_visible[i]), color: "#55a7b5", width: 1.5 },
    { points: trajectory.right_wrist.filter((_, i) => trajectory.right_visible[i]), color: "#ed6a4c", width: 1.5 },
  ];
  const normalized = normalizePoints(series);
  normalized.series.forEach((item) => {
    const projected = item.points.map((point) => projectPoint(point, width, height, normalized));
    if (projected.length < 1) return;
    context.strokeStyle = item.color;
    context.lineWidth = item.width;
    context.lineJoin = "round";
    context.beginPath();
    projected.forEach((point, index) => index ? context.lineTo(point[0], point[1]) : context.moveTo(point[0], point[1]));
    context.stroke();
    projected.forEach((point, index) => {
      if (index % Math.max(1, Math.floor(projected.length / 26)) !== 0 && index !== projected.length - 1) return;
      context.fillStyle = item.color;
      context.globalAlpha = index === projected.length - 1 ? 1 : .45;
      context.beginPath(); context.arc(point[0], point[1], index === projected.length - 1 ? 3.5 : 1.8, 0, Math.PI * 2); context.fill();
    });
    context.globalAlpha = 1;
  });
}

let drag = null;
elements.canvas.addEventListener("pointerdown", (event) => {
  drag = { x: event.clientX, y: event.clientY, yaw: state.yaw, pitch: state.pitch };
  elements.canvas.setPointerCapture(event.pointerId);
});
elements.canvas.addEventListener("pointermove", (event) => {
  if (!drag) return;
  state.yaw = drag.yaw + (event.clientX - drag.x) * .009;
  state.pitch = Math.max(-1.15, Math.min(1.15, drag.pitch + (event.clientY - drag.y) * .007));
  drawTrajectory();
});
elements.canvas.addEventListener("pointerup", () => { drag = null; });
elements.canvas.addEventListener("pointercancel", () => { drag = null; });
new ResizeObserver(drawTrajectory).observe(elements.canvas);

elements.sourceView.addEventListener("click", () => setVideoView("source"));
elements.predictionView.addEventListener("click", () => setVideoView("prediction"));
elements.run.addEventListener("click", startJob);
elements.cancel.addEventListener("click", cancelJob);

async function initialize() {
  try {
    const [status, samples] = await Promise.all([api("/api/status"), api("/api/samples")]);
    elements.status.classList.add("online");
    elements.status.querySelector("span:last-child").textContent = status.model_loaded ? "Model ready" : "Local / idle";
    state.samples = samples;
    renderSamples();
  } catch (error) {
    elements.status.querySelector("span:last-child").textContent = "Offline";
    showToast(error.message);
  }
}

initialize();
