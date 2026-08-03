const state = {
  files: [],
  health: null,
  jobId: null,
  pollTimer: null,
  previewPassed: false,
  selectionError: "",
  uploadInProgress: false,
  uploadMode: "orchestrated",
  apiAuthorization: "",
  lastSubmitPayload: null,
  pollRetryCount: 0,
};

const remoteWebhookMode = window.SHARON_REMOTE_MODE === true;
const orchestratorBase = String(
  window.SHARON_ORCHESTRATOR_BASE || "/api/v1/orchestrator",
).replace(/\/$/, "");
const webhookBase = String(
  window.SHARON_WEBHOOK_BASE || "https://n8n.sharon-finefoods.com/webhook",
).replace(/\/$/, "");
const activeJobStorageKey = "sharon_inventory_active_job";
const r2UploadConcurrency = 4;
const defaultMaxImagesPerJob = 50;
const defaultMaxImageSizeMb = 50;
const defaultMaxJobUploadSizeMb = 160;
const jobPollIntervalMs = 1200;
const initialJobPollDelayMs = 1500;
const maxTransientPollRetries = 20;

const byId = (id) => document.getElementById(id);
const elements = {
  systemStatus: byId("system-status"),
  fileInput: byId("file-input"),
  folderInput: byId("folder-input"),
  dropZone: byId("drop-zone"),
  fileCounter: byId("file-counter"),
  selection: byId("selection"),
  selectionSummary: byId("selection-summary"),
  fileList: byId("file-list"),
  startJob: byId("start-job"),
  uploadProgress: byId("upload-progress"),
  progressValue: byId("progress-value"),
  progressBar: byId("progress-bar"),
  uploadError: byId("upload-error"),
  processingSection: byId("processing-section"),
  processingTitle: byId("processing-title"),
  processingCopy: byId("processing-copy"),
  resultsSection: byId("results-section"),
  kiotvietMessage: byId("kiotviet-message"),
  previewKiotviet: byId("preview-kiotviet"),
  createKiotviet: byId("create-kiotviet"),
  remoteLogin: byId("remote-login"),
  remoteLoginForm: byId("remote-login-form"),
  remoteUsername: byId("remote-username"),
  remotePassword: byId("remote-password"),
  remoteLoginSubmit: byId("remote-login-submit"),
  remoteLoginError: byId("remote-login-error"),
  remoteLogout: byId("remote-logout"),
};

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function showMessage(element, text, kind = "") {
  element.textContent = text;
  element.className = `message${kind ? ` message-${kind}` : ""}`;
}

function hideMessage(element) {
  element.classList.add("hidden");
}

function apiError(payload, fallback) {
  if (payload && typeof payload.detail === "string") return payload.detail;
  if (payload && typeof payload.error === "string") return payload.error;
  return fallback;
}

function lockRemoteSession(message = "") {
  if (!remoteWebhookMode) return;
  state.apiAuthorization = "";
  state.health = null;
  document.documentElement.classList.remove("authenticated");
  if (message) showMessage(elements.remoteLoginError, message, "error");
  else hideMessage(elements.remoteLoginError);
  window.setTimeout(() => elements.remoteUsername.focus(), 0);
}

function encodeBasicAuthorization(username, password) {
  const bytes = new TextEncoder().encode(`${username}:${password}`);
  let binary = "";
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return `Basic ${window.btoa(binary)}`;
}

function webhookUrl(path, query = {}) {
  const url = new URL(`${webhookBase}/${path}`);
  Object.entries(query).forEach(([key, value]) => url.searchParams.set(key, value));
  return url.toString();
}

async function checkHealth() {
  try {
    let data;
    if (remoteWebhookMode) {
      data = await requestJson(webhookUrl("bakery-health"), { cache: "no-store" });
      state.uploadMode = "orchestrated";
    } else {
      try {
        data = await requestJson(`${orchestratorBase}/health`, { cache: "no-store" });
        state.uploadMode = "orchestrated";
      } catch (orchestratorError) {
        data = await requestJson("/api/v1/bakery/health", { cache: "no-store" });
        if (!data.ready) throw orchestratorError;
        state.uploadMode = "direct";
      }
    }
    if (!data.ready || !data.r2_configured) {
      throw new Error(data.detail || data.error || "Pipeline n8n/R2 chưa sẵn sàng");
    }
    state.health = data;
    elements.systemStatus.classList.add("ready");
    elements.systemStatus.classList.remove("error");
    elements.systemStatus.querySelector("span:last-child").textContent = "Hệ thống sẵn sàng";
    byId("hero-status-label").textContent = state.uploadMode === "direct"
      ? "Sẵn sàng · chế độ trực tiếp"
      : "Sẵn sàng vận hành";
    byId("hero-panel").classList.remove("unavailable");
    const extensions = (data.allowed_image_extensions || []).map((item) => item.replace(".", "").toUpperCase()).join(", ");
    byId("upload-limits").textContent = `${extensions} · tối đa ${data.max_images_per_job} ảnh · ${data.max_image_size_mb} MB/ảnh · ${data.max_job_upload_size_mb} MB/lần`;
    validateSelection();
    renderSelection();
    return true;
  } catch (error) {
    elements.systemStatus.classList.add("error");
    elements.systemStatus.querySelector("span:last-child").textContent = "Hệ thống chưa sẵn sàng";
    byId("hero-status-label").textContent = "Cần kiểm tra hệ thống";
    byId("hero-panel").classList.add("unavailable");
    showMessage(elements.uploadError, `Không thể bắt đầu: ${error.message}`, "error");
    if (remoteWebhookMode) showMessage(elements.remoteLoginError, error.message, "error");
    return false;
  }
}

function fileKey(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function selectFiles(fileList) {
  const allowed = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp"]);
  const incoming = Array.from(fileList).filter((file) => allowed.has(file.type));
  const merged = new Map(state.files.map((file) => [fileKey(file), file]));
  incoming.forEach((file) => merged.set(fileKey(file), file));
  state.files = Array.from(merged.values());

  validateSelection();
  renderSelection();
}

function validateSelection() {
  const maxImages = state.health?.max_images_per_job ?? defaultMaxImagesPerJob;
  const maxImageSizeMb = state.health?.max_image_size_mb ?? defaultMaxImageSizeMb;
  const maxJobUploadSizeMb = state.health?.max_job_upload_size_mb ?? defaultMaxJobUploadSizeMb;
  const maxBytes = maxImageSizeMb * 1024 * 1024;
  const maxTotalBytes = maxJobUploadSizeMb * 1024 * 1024;
  const totalBytes = state.files.reduce((sum, file) => sum + file.size, 0);
  state.selectionError = "";
  if (state.files.length > maxImages) {
    state.files = state.files.slice(0, maxImages);
    state.selectionError = `Chỉ nhận tối đa ${maxImages} ảnh trong một lần.`;
  } else if (state.files.some((file) => file.size > maxBytes)) {
    state.selectionError = `Có ảnh vượt quá giới hạn ${maxImageSizeMb} MB.`;
  } else if (totalBytes > maxTotalBytes) {
    state.selectionError = `Tổng dung lượng vượt quá ${maxJobUploadSizeMb} MB. Hãy chia thành nhiều lần kiểm đếm.`;
  }
  if (state.selectionError) {
    showMessage(elements.uploadError, state.selectionError, "error");
  } else {
    hideMessage(elements.uploadError);
  }
}

function renderSelection() {
  const count = state.files.length;
  elements.fileCounter.textContent = `${count} ảnh`;
  elements.selection.classList.toggle("hidden", count === 0);
  elements.selectionSummary.textContent = count ? `${count} ảnh · ${formatBytes(state.files.reduce((sum, file) => sum + file.size, 0))}` : "";
  elements.fileList.replaceChildren();
  state.files.forEach((file) => {
    const item = document.createElement("li");
    const name = document.createElement("span");
    const size = document.createElement("span");
    name.textContent = file.webkitRelativePath || file.name;
    size.textContent = formatBytes(file.size);
    item.append(name, size);
    elements.fileList.appendChild(item);
  });
  elements.startJob.disabled = count === 0 || !state.health?.ready || Boolean(state.selectionError);
}

async function requestJson(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (remoteWebhookMode && state.apiAuthorization) {
    headers.set("Authorization", state.apiAuthorization);
  }
  const response = await fetch(url, {
    ...options,
    headers,
    credentials: remoteWebhookMode ? "omit" : "same-origin",
  });
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* ignore malformed response */ }
  if (remoteWebhookMode && response.status === 401) {
    lockRemoteSession("Tên đăng nhập hoặc mật khẩu không đúng.");
  }
  if (!response.ok) throw new Error(apiError(payload, `Yêu cầu thất bại (HTTP ${response.status}).`));
  return payload;
}

function throwWorkflowError(payload, fallback) {
  if (String(payload?.status || "").toUpperCase() === "ERROR") {
    throw new Error(apiError(payload, fallback));
  }
}

function isJobNotFound(value) {
  const message = typeof value === "string"
    ? value
    : apiError(value, value?.message || "");
  return /job\s+not\s+found/i.test(String(message));
}

function isTransientPollError(error) {
  const message = String(error?.message || error || "");
  return isJobNotFound(message)
    || /failed to fetch|networkerror|load failed|network request failed/i.test(message)
    || /HTTP 5\d\d/i.test(message);
}

function scheduleJobPoll(delayMs = jobPollIntervalMs) {
  clearTimeout(state.pollTimer);
  state.pollTimer = window.setTimeout(pollJob, delayMs);
}

async function prepareOrchestratedUploads(files) {
  const payload = { files };
  if (!remoteWebhookMode) {
    return requestJson(`${orchestratorBase}/uploads/presign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  const queued = await requestJson(webhookUrl("bakery-upload-init"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  throwWorkflowError(queued, "Không thể tạo phiên upload.");
  if (Array.isArray(queued.uploads)) return queued;
  const requestId = queued.request_id || queued.job_id;
  if (!requestId) throw new Error("n8n không trả về mã phiên upload.");

  for (let attempt = 0; attempt < 90; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 500));
    const prepared = await requestJson(
      webhookUrl("bakery-request-status", { request_id: requestId }),
      { cache: "no-store" },
    );
    throwWorkflowError(prepared, "Worker không thể chuẩn bị URL upload.");
    if (prepared.status === "READY" && Array.isArray(prepared.uploads)) return prepared;
  }
  throw new Error("Worker không phản hồi yêu cầu upload trong thời gian cho phép.");
}

async function submitOrchestratedJob(payload) {
  const accepted = await requestJson(
    remoteWebhookMode ? webhookUrl("bakery-submit") : `${orchestratorBase}/jobs`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
  throwWorkflowError(accepted, "n8n không thể tiếp nhận job.");
  return accepted;
}

async function readOrchestratedJob(jobId) {
  const job = await requestJson(
    remoteWebhookMode
      ? webhookUrl("bakery-job-status", { job_id: jobId })
      : `${orchestratorBase}/jobs/${encodeURIComponent(jobId)}`,
    { cache: "no-store" },
  );
  return job;
}

function updateUploadProgress(completed, total) {
  const percent = Math.round(10 + (completed / total) * 80);
  elements.progressValue.textContent = `${percent}%`;
  elements.progressBar.style.width = `${percent}%`;
  elements.uploadProgress.querySelector(".progress-copy span").textContent =
    `Đang tải ảnh lên R2 (${completed}/${total})…`;
}

async function uploadFilesToR2(files, uploads) {
  let nextIndex = 0;
  let completed = 0;
  const workerCount = Math.min(r2UploadConcurrency, files.length);

  async function worker() {
    while (nextIndex < files.length) {
      const index = nextIndex;
      nextIndex += 1;
      const target = uploads[index];
      const response = await fetch(target.upload_url, {
        method: target.method || "PUT",
        headers: target.headers || { "Content-Type": files[index].type },
        body: files[index],
      });
      if (!response.ok) {
        throw new Error(`Không thể tải ${files[index].name} lên R2.`);
      }
      completed += 1;
      updateUploadProgress(completed, files.length);
    }
  }

  await Promise.all(Array.from({ length: workerCount }, () => worker()));
}

function rememberActiveJob(jobId) {
  localStorage.setItem(
    activeJobStorageKey,
    JSON.stringify({
      job_id: jobId,
      upload_mode: state.uploadMode,
      submit_payload: state.lastSubmitPayload,
      saved_at: Date.now(),
    }),
  );
}

function forgetActiveJob() {
  localStorage.removeItem(activeJobStorageKey);
  state.lastSubmitPayload = null;
  state.pollRetryCount = 0;
}

async function uploadJobOrchestrated() {
  if (state.uploadInProgress) return;
  state.uploadInProgress = true;
  elements.startJob.disabled = true;
  elements.uploadProgress.classList.remove("hidden");
  elements.progressValue.textContent = "5%";
  elements.progressBar.style.width = "5%";
  elements.uploadProgress.querySelector(".progress-copy span").textContent =
    "Đang chuẩn bị phiên tải ảnh…";
  hideMessage(elements.uploadError);
  try {
    const prepared = await prepareOrchestratedUploads(
      state.files.map((file) => ({
        filename: file.name,
        content_type: file.type || "application/octet-stream",
        size_bytes: file.size,
      })),
    );
    if (!Array.isArray(prepared.uploads) || prepared.uploads.length !== state.files.length) {
      throw new Error("n8n không trả về đủ URL upload R2.");
    }

    await uploadFilesToR2(state.files, prepared.uploads);

    elements.uploadProgress.querySelector(".progress-copy span").textContent =
      "Đang giao job cho AI…";
    const submitPayload = {
      job_id: prepared.job_id,
      files: prepared.uploads.map((item) => ({ object_key: item.object_key })),
    };
    let accepted;
    try {
      accepted = await submitOrchestratedJob(submitPayload);
    } catch (submitError) {
      // A lost response after a successful submit is safe: the backend returns
      // the existing job on retry and never schedules a second receipt.
      try {
        accepted = await readOrchestratedJob(prepared.job_id);
        if (!accepted?.job_id) throw submitError;
      } catch (_) {
        throw submitError;
      }
    }
    state.jobId = accepted.job_id;
    state.lastSubmitPayload = submitPayload;
    state.pollRetryCount = 0;
    rememberActiveJob(state.jobId);
    state.previewPassed = false;
    elements.progressValue.textContent = "100%";
    elements.progressBar.style.width = "100%";
    elements.processingSection.classList.remove("hidden");
    elements.processingSection.querySelector(".spinner").classList.remove("hidden");
    elements.resultsSection.classList.add("hidden");
    elements.processingTitle.textContent = "AI đang kiểm đếm…";
    elements.processingCopy.textContent = `Đã tiếp nhận ${accepted.total_images} ảnh qua n8n/R2.`;
    elements.processingSection.scrollIntoView({ behavior: "smooth", block: "center" });
    scheduleJobPoll(initialJobPollDelayMs);
  } catch (error) {
    finishUploadError(error.message);
  } finally {
    state.uploadInProgress = false;
  }
}

function uploadJob() {
  if (!state.files.length) return;
  if (state.uploadMode === "direct") {
    uploadJobDirect();
  } else {
    uploadJobOrchestrated();
  }
}

async function uploadJobDirect() {
  if (state.uploadInProgress) return;
  state.uploadInProgress = true;
  elements.startJob.disabled = true;
  elements.uploadProgress.classList.remove("hidden");
  elements.progressValue.textContent = "10%";
  elements.progressBar.style.width = "10%";
  elements.uploadProgress.querySelector(".progress-copy span").textContent =
    "Đang tải ảnh trực tiếp vào hệ thống…";
  hideMessage(elements.uploadError);

  try {
    const formData = new FormData();
    state.files.forEach((file) => formData.append("files", file, file.name));
    const accepted = await requestJson("/api/v1/bakery/jobs", {
      method: "POST",
      body: formData,
    });
    state.jobId = accepted.job_id;
    state.lastSubmitPayload = null;
    state.pollRetryCount = 0;
    rememberActiveJob(state.jobId);
    state.previewPassed = false;
    elements.progressValue.textContent = "100%";
    elements.progressBar.style.width = "100%";
    elements.processingSection.classList.remove("hidden");
    elements.processingSection.querySelector(".spinner").classList.remove("hidden");
    elements.resultsSection.classList.add("hidden");
    elements.processingTitle.textContent = "AI đang kiểm đếm…";
    elements.processingCopy.textContent = `Đã tiếp nhận ${accepted.total_images} ảnh trực tiếp.`;
    elements.processingSection.scrollIntoView({ behavior: "smooth", block: "center" });
    scheduleJobPoll(initialJobPollDelayMs);
  } catch (error) {
    finishUploadError(error.message);
  } finally {
    state.uploadInProgress = false;
  }
}

function finishUploadError(message) {
  elements.startJob.disabled = false;
  elements.uploadProgress.classList.add("hidden");
  showMessage(elements.uploadError, message, "error");
}

async function pollJob() {
  clearTimeout(state.pollTimer);
  try {
    const job = state.uploadMode === "direct"
      ? await requestJson(`/api/v1/bakery/jobs/${encodeURIComponent(state.jobId)}`, { cache: "no-store" })
      : await readOrchestratedJob(state.jobId);
    if (
      remoteWebhookMode
      && String(job.status || "").toUpperCase() === "ERROR"
      && isJobNotFound(job)
      && state.pollRetryCount < maxTransientPollRetries
    ) {
      state.pollRetryCount += 1;
      elements.processingTitle.textContent = "Đang đồng bộ job với AI…";
      elements.processingCopy.textContent = `Đang xác nhận job (${state.pollRetryCount}/${maxTransientPollRetries})…`;

      // n8n persists workflow static data just after returning the webhook
      // response. If another webhook reads the old snapshot at that instant,
      // retrying the idempotent submit restores the same job without creating
      // a second local receipt.
      if (
        state.lastSubmitPayload
        && (state.pollRetryCount === 3 || state.pollRetryCount === 10)
      ) {
        await submitOrchestratedJob(state.lastSubmitPayload);
      }
      scheduleJobPoll();
      return;
    }
    elements.processingCopy.textContent = `Đã xử lý ${job.processed_images}/${job.total_images} ảnh.`;
    if (job.status === "COMPLETED") {
      forgetActiveJob();
      elements.processingSection.classList.add("hidden");
      elements.uploadProgress.classList.add("hidden");
      renderResults(job);
      return;
    }
    if (job.status === "ERROR") {
      if (remoteWebhookMode && isJobNotFound(job)) {
        forgetActiveJob();
        throw new Error("n8n chưa đồng bộ được job sau nhiều lần thử. Vui lòng kiểm tra worker rồi thực hiện lại.");
      }
      forgetActiveJob();
      throw new Error(job.error || "AI không thể xử lý bộ ảnh này.");
    }
    state.pollRetryCount = 0;
    scheduleJobPoll();
  } catch (error) {
    if (
      remoteWebhookMode
      && isTransientPollError(error)
      && state.pollRetryCount < maxTransientPollRetries
    ) {
      state.pollRetryCount += 1;
      elements.processingTitle.textContent = "Đang nối lại với hệ thống…";
      elements.processingCopy.textContent = `Kết nối tạm thời gián đoạn, đang thử lại (${state.pollRetryCount}/${maxTransientPollRetries})…`;
      scheduleJobPoll();
      return;
    }
    elements.processingTitle.textContent = "Xử lý không thành công";
    elements.processingCopy.textContent = error.message;
    elements.processingSection.querySelector(".spinner").classList.add("hidden");
    elements.startJob.disabled = false;
  }
}

function renderResults(job) {
  elements.resultsSection.classList.remove("hidden");
  byId("job-reference").textContent = `JOB ${job.job_id}`;
  byId("metric-products").textContent = job.product_count;
  byId("metric-quantity").textContent = job.total_quantity;
  byId("metric-images").textContent = job.processed_images;
  byId("metric-r2").textContent = job.r2_objects?.length ? "Đã lưu" : "Chưa lưu";
  byId("download-excel").href = job.excel_url;

  const rows = byId("product-rows");
  rows.replaceChildren();
  job.products.forEach((product) => {
    const row = document.createElement("tr");
    [product.product_code, product.product_name, product.quantity, `${Number(product.purchase_price || 0).toLocaleString("vi-VN")} đ`].forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index >= 2) cell.className = "number";
      row.appendChild(cell);
    });
    rows.appendChild(row);
  });

  const grid = byId("annotated-grid");
  grid.replaceChildren();
  job.images.forEach((image) => {
    const item = document.createElement("a");
    item.className = "annotated-item";
    item.href = image.annotated_url;
    item.target = "_blank";
    item.rel = "noopener";
    const picture = document.createElement("img");
    picture.src = image.annotated_url;
    picture.alt = `Ảnh nhận diện ${image.image_name}`;
    picture.loading = "lazy";
    const info = document.createElement("div");
    const name = document.createElement("span");
    const count = document.createElement("span");
    name.textContent = image.image_name;
    count.textContent = `${image.total_detections} SP`;
    info.append(name, count);
    item.append(picture, info);
    grid.appendChild(item);
  });

  const existingReceipt = job.kiotviet?.created === true;
  const automaticMode = state.health?.kiotviet_auto_create_draft === true;
  elements.previewKiotviet.disabled = true;
  elements.createKiotviet.disabled = true;
  if (existingReceipt) {
    const receipt = job.kiotviet.receipt || {};
    showMessage(elements.kiotvietMessage, `Đã tự động tạo phiếu nhập nháp ${receipt.code || "KiotViet"}. Hệ thống không cho tạo trùng từ job này.`, "success");
  } else if (automaticMode && job.kiotviet?.error) {
    const nextStep = remoteWebhookMode
      ? " Hãy kiểm tra trên máy vận hành trước khi thử lại."
      : " Bạn có thể kiểm tra lại và thử lại.";
    showMessage(elements.kiotvietMessage, `Không thể tự động tạo phiếu: ${job.kiotviet.error}${nextStep}`, "error");
  } else if (automaticMode) {
    showMessage(
      elements.kiotvietMessage,
      remoteWebhookMode
        ? "Job chưa có kết quả tạo phiếu tự động. Hãy kiểm tra trên máy vận hành."
        : "Job cũ chưa có kết quả tự động. Có thể kiểm tra lại dữ liệu và tạo phiếu thủ công.",
    );
  } else {
    hideMessage(elements.kiotvietMessage);
  }
  elements.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function previewKiotViet() {
  if (remoteWebhookMode) return;
  elements.previewKiotviet.disabled = true;
  elements.createKiotviet.disabled = true;
  showMessage(elements.kiotvietMessage, "Đang đối chiếu chi nhánh và hàng hóa với KiotViet…");
  try {
    const response = await fetch(`/api/v1/bakery/jobs/${state.jobId}/kiotviet-preview`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(apiError(payload, "Không thể kiểm tra dữ liệu KiotViet."));
    const validation = payload.validation || {};
    const productCount = (validation.products || []).length;
    state.previewPassed = true;
    elements.createKiotviet.disabled = false;
    showMessage(elements.kiotvietMessage, `Đã kiểm tra: chi nhánh ${validation.branch?.name || "Warehouse"}, ${productCount} mặt hàng hợp lệ. Có thể tạo phiếu nhập nháp.`, "success");
  } catch (error) {
    state.previewPassed = false;
    showMessage(elements.kiotvietMessage, error.message, "error");
  } finally {
    elements.previewKiotviet.disabled = false;
  }
}

async function createKiotVietReceipt() {
  if (remoteWebhookMode) return;
  if (!state.previewPassed) return;
  const confirmed = window.confirm("Tạo một phiếu nhập nháp thật trên KiotViet? Mỗi job chỉ được tạo một lần.");
  if (!confirmed) return;
  elements.previewKiotviet.disabled = true;
  elements.createKiotviet.disabled = true;
  showMessage(elements.kiotvietMessage, "Đang tạo phiếu nhập nháp trên KiotViet…");
  try {
    const response = await fetch(`/api/v1/bakery/jobs/${state.jobId}/kiotviet`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: true }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(apiError(payload, "Không thể tạo phiếu KiotViet."));
    const receipt = payload.receipt || {};
    showMessage(elements.kiotvietMessage, `Đã tạo phiếu nhập nháp ${receipt.code || "thành công"}. Không thể tạo lại từ job này.`, "success");
  } catch (error) {
    showMessage(elements.kiotvietMessage, error.message, "error");
    elements.previewKiotviet.disabled = false;
    elements.createKiotviet.disabled = false;
  }
}

byId("pick-files").addEventListener("click", (event) => { event.stopPropagation(); elements.fileInput.click(); });
byId("pick-folder").addEventListener("click", (event) => { event.stopPropagation(); elements.folderInput.click(); });
elements.fileInput.addEventListener("change", (event) => selectFiles(event.target.files));
elements.folderInput.addEventListener("change", (event) => selectFiles(event.target.files));
elements.fileInput.addEventListener("click", (event) => event.stopPropagation());
elements.folderInput.addEventListener("click", (event) => event.stopPropagation());
elements.dropZone.addEventListener("click", (event) => {
  if (event.target.closest("button, input")) return;
  elements.fileInput.click();
});
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); elements.fileInput.click(); }
});
["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault(); elements.dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault(); elements.dropZone.classList.remove("dragging");
}));
elements.dropZone.addEventListener("drop", (event) => selectFiles(event.dataTransfer.files));
byId("clear-files").addEventListener("click", () => {
  state.files = [];
  state.selectionError = "";
  elements.fileInput.value = "";
  elements.folderInput.value = "";
  renderSelection();
});
elements.startJob.addEventListener("click", uploadJob);
elements.previewKiotviet.addEventListener("click", previewKiotViet);
elements.createKiotviet.addEventListener("click", createKiotVietReceipt);

elements.remoteLoginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!remoteWebhookMode) return;
  const username = elements.remoteUsername.value.trim();
  const password = elements.remotePassword.value;
  if (!username || !password) return;
  elements.remoteLoginSubmit.disabled = true;
  elements.remoteLoginSubmit.textContent = "Đang kiểm tra…";
  hideMessage(elements.remoteLoginError);
  state.apiAuthorization = encodeBasicAuthorization(username, password);
  const ready = await checkHealth();
  if (ready) {
    document.documentElement.classList.add("authenticated");
    elements.remotePassword.value = "";
    hideMessage(elements.remoteLoginError);
    await resumeActiveJob();
  } else {
    state.apiAuthorization = "";
  }
  elements.remoteLoginSubmit.disabled = false;
  elements.remoteLoginSubmit.textContent = "Đăng nhập";
});

elements.remoteLogout.addEventListener("click", () => {
  clearTimeout(state.pollTimer);
  lockRemoteSession();
  elements.remotePassword.value = "";
});

async function resumeActiveJob() {
  let active = null;
  try {
    active = JSON.parse(localStorage.getItem(activeJobStorageKey) || "null");
  } catch (_) {
    forgetActiveJob();
  }
  if (!active?.job_id || !/^[0-9a-f]{32}$/.test(active.job_id)) return;
  if (active.upload_mode === "direct" || active.upload_mode === "orchestrated") {
    state.uploadMode = active.upload_mode;
  }
  state.jobId = active.job_id;
  state.lastSubmitPayload = active.submit_payload || null;
  state.pollRetryCount = 0;
  elements.processingSection.classList.remove("hidden");
  elements.processingSection.querySelector(".spinner").classList.remove("hidden");
  elements.processingTitle.textContent = "Đang tiếp tục job trước…";
  elements.processingCopy.textContent = `Job ${active.job_id}`;
  scheduleJobPoll(initialJobPollDelayMs);
}

(async () => {
  if (remoteWebhookMode) {
    lockRemoteSession();
    return;
  }
  await checkHealth();
  await resumeActiveJob();
})();
