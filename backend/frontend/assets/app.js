javascript
const state = {
  files: [],
  health: null,
  jobId: null,
  pollTimer: null,
  currentJob: null,
  confirmationInProgress: false,
  selectionError: "",
  uploadInProgress: false,
  uploadMode: "orchestrated",
  inferenceMode: "AUTO",
  productCatalog: [],
  selectedProductCode: "",
  apiAuthorization: "",
  lastSubmitPayload: null,
  pollRetryCount: 0,
  developerKey: "",
  developerSettings: null,
  developerBusy: false,
};

const remoteWebhookMode = window.SHARON_REMOTE_MODE === true;
const orchestratorBase = String(
  window.SHARON_ORCHESTRATOR_BASE || "/api/v1/orchestrator",
).replace(/\/$/, "");
const webhookBase = String(
  window.SHARON_WEBHOOK_BASE || "https://n8n.sharon-finefoods.com/webhook",
).replace(/\/$/, "");
const activeJobStorageKey = "sharon_inventory_active_job";
const remoteAuthStorageKey = "sharon_inventory_remote_authorization";
const remoteUsernameStorageKey = "sharon_inventory_remote_username";

if (remoteWebhookMode) {
  document.documentElement.classList.add("remote-mode");
}

/*
 * n8n serves webhook HTML inside a sandboxed iframe.
 * In that environment localStorage may be blocked because
 * the iframe does not have allow-same-origin.
 *
 * Use localStorage when available. n8n's sandbox can either throw on access
 * or expose storage that is discarded on the next navigation, so also mirror
 * the values into window.name. window.name survives reloads in the same tab
 * even when the document has an opaque/sandboxed origin.
 */
const fallbackStorage = new Map();
const windowNameStoragePrefix = "sharon_inventory_storage:";

function readWindowNameStorage() {
  const raw = String(window.name || "");
  if (!raw.startsWith(windowNameStoragePrefix)) return {};
  try {
    const parsed = JSON.parse(
      decodeURIComponent(raw.slice(windowNameStoragePrefix.length)),
    );
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed
      : {};
  } catch (_) {
    return {};
  }
}

function writeWindowNameStorage(values) {
  try {
    window.name = `${windowNameStoragePrefix}${encodeURIComponent(JSON.stringify(values))}`;
  } catch (_) {
    // Keep the in-memory Map as a last-resort fallback for unusual browsers.
  }
}

function getFallbackStorageItem(key) {
  const persisted = readWindowNameStorage();
  if (Object.prototype.hasOwnProperty.call(persisted, key)) {
    return String(persisted[key]);
  }
  return fallbackStorage.get(key) ?? null;
}

function setFallbackStorageItem(key, value) {
  const stringValue = String(value);
  fallbackStorage.set(key, stringValue);
  const persisted = readWindowNameStorage();
  persisted[key] = stringValue;
  writeWindowNameStorage(persisted);
}

function removeFallbackStorageItem(key) {
  fallbackStorage.delete(key);
  const persisted = readWindowNameStorage();
  if (!Object.prototype.hasOwnProperty.call(persisted, key)) return;
  delete persisted[key];
  if (Object.keys(persisted).length) writeWindowNameStorage(persisted);
  else window.name = "";
}

const safeStorage = {
  getItem(key) {
    try {
      const value = window.localStorage.getItem(key);
      return value ?? getFallbackStorageItem(key);
    } catch (_) {
      return getFallbackStorageItem(key);
    }
  },

  setItem(key, value) {
    // Always mirror to window.name because sandboxed localStorage can appear
    // writable yet be cleared when the webhook document reloads.
    setFallbackStorageItem(key, value);
    try {
      window.localStorage.setItem(key, value);
    } catch (_) {}
  },

  removeItem(key) {
    removeFallbackStorageItem(key);
    try {
      window.localStorage.removeItem(key);
    } catch (_) {}
  },
};

const r2UploadConcurrency = 4;
const uploadMaxLongSide = 1600;
const uploadJpegQuality = 0.86;
const optimiseMinBytes = 1.5 * 1024 * 1024;
const imageOptimiseConcurrency = (() => {
  const logicalCores = Number(navigator.hardwareConcurrency || 4);
  const deviceMemoryGb = Number(navigator.deviceMemory || 4);
  // Three simultaneous decodes are safe only on stronger phones/desktops.
  // Lower-memory devices stay at two to avoid browser tab eviction.
  return logicalCores >= 6 && deviceMemoryGb >= 6 ? 3 : 2;
})();
const defaultMaxImagesPerJob = 50;
const defaultMaxImageSizeMb = 50;
const defaultMaxJobUploadSizeMb = 200;
const defaultInferenceMode = "AUTO";
const jobPollIntervalMs = 1200;
const initialJobPollDelayMs = 1500;
const maxTransientPollRetries = 20;

const byId = (id) => document.getElementById(id);

// Never leave credentials in the address bar/history if an older broken page
// submitted the login form with the browser's default GET behaviour.
const currentUrl = new URL(window.location.href);
if (currentUrl.searchParams.has("username") || currentUrl.searchParams.has("password")) {
  currentUrl.searchParams.delete("username");
  currentUrl.searchParams.delete("password");
  window.history.replaceState({}, document.title, currentUrl.toString());
}

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
  inferenceMode: byId("inference-mode"),
  uploadProgress: byId("upload-progress"),
  progressValue: byId("progress-value"),
  progressBar: byId("progress-bar"),
  uploadError: byId("upload-error"),
  processingSection: byId("processing-section"),
  processingTitle: byId("processing-title"),
  processingCopy: byId("processing-copy"),
  decisionSection: byId("decision-section"),
  decisionJobReference: byId("decision-job-reference"),
  decisionClass: byId("decision-class"),
  decisionCount: byId("decision-count"),
  decisionImages: byId("decision-images"),
  decisionPurity: byId("decision-purity"),
  decisionBadge: byId("decision-badge"),
  decisionTitle: byId("decision-title"),
  decisionMessage: byId("decision-message"),
  decisionConfidence: byId("decision-confidence"),
  familySelection: byId("family-selection"),
  familySelectionLabel: byId("family-selection-label"),
  familySelectionHelp: byId("family-selection-help") || byId("family-selection")?.querySelector("small"),
  familySkuSelect: byId("family-sku-select"),
  confirmedQuantity: byId("confirmed-quantity"),
  documentType: byId("document-type"),
  productOverride: byId("product-override"),
  productSearchInput: byId("product-search-input"),
  productSuggestions: byId("product-suggestions"),
  selectedProductSummary: byId("selected-product-summary"),
  confirmationError: byId("confirmation-error"),
  confirmJob: byId("confirm-job"),
  retakeJob: byId("retake-job"),
  decisionImageRows: byId("decision-image-rows"),
  decisionAnnotatedGrid: byId("decision-annotated-grid"),
  resultsSection: byId("results-section"),
  kiotvietMessage: byId("kiotviet-message"),
  remoteLogin: byId("remote-login"),
  remoteLoginForm: byId("remote-login-form"),
  remoteUsername: byId("remote-username"),
  remotePassword: byId("remote-password"),
  remoteLoginSubmit: byId("remote-login-submit"),
  remoteLoginError: byId("remote-login-error"),
  remoteLogout: byId("remote-logout"),
  developerOverlay: byId("developer-settings"),
  developerTrigger: byId("developer-trigger"),
  developerClose: byId("developer-close"),
  developerUnlockForm: byId("developer-unlock-form"),
  developerKey: byId("developer-key"),
  developerLoad: byId("developer-load"),
  developerConfigForm: byId("developer-config-form"),
  developerModel: byId("developer-model"),
  developerThresholdRows: byId("developer-threshold-rows"),
  developerSave: byId("developer-save"),
  developerUpdatedAt: byId("developer-updated-at"),
  developerMessage: byId("developer-message"),
};

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function imageContentType(file) {
  const declared = String(file?.type || "").toLowerCase();
  if (["image/jpeg", "image/png", "image/webp", "image/bmp"].includes(declared)) {
    return declared;
  }
  const extension = String(file?.name || "").split(".").pop()?.toLowerCase();
  return {
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    png: "image/png",
    webp: "image/webp",
    bmp: "image/bmp",
  }[extension] || "";
}


function normaliseSearch(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
}

function productLabel(product) {
  const name = product?.display_name || product?.product_name || product?.product_code || "";
  return `${name} · ${product?.product_code || ""}`;
}

function catalogProduct(productCode) {
  const target = String(productCode || "").trim().toLowerCase();
  return state.productCatalog.find(
    (item) => String(item?.product_code || "").trim().toLowerCase() === target,
  ) || null;
}

function updateConfirmEnabled() {
  const quantity = Number(elements.confirmedQuantity.value);
  const quantityValid = Number.isInteger(quantity) && quantity >= 1 && quantity <= 5000;
  const decisionType = String(state.currentJob?.decision?.decision || "").toUpperCase();
  const confirmable = ["DIRECT", "FAMILY", "REVIEW"].includes(decisionType);
  elements.confirmJob.disabled = !confirmable
    || !quantityValid
    || !state.selectedProductCode
    || state.confirmationInProgress;
}

function selectConfirmedProduct(productCode, { updateInput = true } = {}) {
  const product = catalogProduct(productCode);
  state.selectedProductCode = product?.product_code || "";
  if (updateInput) {
    elements.productSearchInput.value = product ? productLabel(product) : "";
  }
  elements.selectedProductSummary.textContent = product
    ? `Sẽ xác nhận: ${productLabel(product)}`
    : "Chưa chọn sản phẩm xác nhận.";
  elements.selectedProductSummary.classList.toggle("selected", Boolean(product));
  elements.productSuggestions.replaceChildren();
  elements.productSuggestions.classList.add("hidden");
  updateConfirmEnabled();
}

function renderProductSuggestions(query = "") {
  const needle = normaliseSearch(query);
  const matches = state.productCatalog
    .filter((item) => {
      if (!needle) return true;
      const haystack = normaliseSearch([
        item.product_code,
        item.product_name,
        item.display_name,
        item.visual_class,
        item.family_name,
      ].filter(Boolean).join(" "));
      return haystack.includes(needle);
    })
    .slice(0, 12);

  elements.productSuggestions.replaceChildren();
  matches.forEach((product) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "product-suggestion";
    const main = document.createElement("strong");
    main.textContent = product.display_name || product.product_name || product.product_code;
    const meta = document.createElement("span");
    meta.textContent = [product.product_code, product.family_name].filter(Boolean).join(" · ");
    button.append(main, meta);
    button.addEventListener("click", () => selectConfirmedProduct(product.product_code));
    elements.productSuggestions.appendChild(button);
  });
  elements.productSuggestions.classList.toggle("hidden", matches.length === 0);
}

function openProductOverride({ clear = false } = {}) {
  elements.productOverride.classList.remove("hidden");
  if (clear) {
    state.selectedProductCode = "";
    elements.productSearchInput.value = "";
    elements.selectedProductSummary.textContent = "Nhập tên hoặc mã của một sản phẩm đã train.";
    elements.selectedProductSummary.classList.remove("selected");
  }
  renderProductSuggestions(elements.productSearchInput.value);
  window.setTimeout(() => elements.productSearchInput.focus(), 0);
  updateConfirmEnabled();
}

async function optimiseImageFile(file) {
  if (!file || file.size < optimiseMinBytes || typeof createImageBitmap !== "function") {
    return file;
  }
  try {
    const bitmap = await createImageBitmap(file);
    const longest = Math.max(bitmap.width, bitmap.height);
    const scale = Math.min(1, uploadMaxLongSide / Math.max(1, longest));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = typeof OffscreenCanvas === "function"
      ? new OffscreenCanvas(width, height)
      : document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: false });
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    context.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    const blob = typeof canvas.convertToBlob === "function"
      ? await canvas.convertToBlob({ type: "image/jpeg", quality: uploadJpegQuality })
      : await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", uploadJpegQuality));
    // Release the large pixel buffer promptly before starting the next image.
    canvas.width = 1;
    canvas.height = 1;
    if (!blob || blob.size >= file.size * 0.95) return file;
    const stem = String(file.name || "image").replace(/\.[^.]+$/, "") || "image";
    return new File([blob], `${stem}.jpg`, { type: "image/jpeg", lastModified: file.lastModified });
  } catch (_) {
    return file;
  }
}

async function optimiseFilesForUpload(files) {
  let nextIndex = 0;
  let completed = 0;
  const output = new Array(files.length);
  const originalBytes = files.reduce((sum, file) => sum + file.size, 0);
  const workerCount = Math.min(imageOptimiseConcurrency, files.length);

  elements.uploadProgress.querySelector(".progress-copy span").textContent =
    `Đang chuẩn bị tối ưu ${files.length} ảnh…`;
  elements.progressValue.textContent = "3%";
  elements.progressBar.style.width = "3%";

  async function worker() {
    while (nextIndex < files.length) {
      const index = nextIndex++;
      output[index] = await optimiseImageFile(files[index]);
      completed += 1;
      elements.uploadProgress.querySelector(".progress-copy span").textContent =
        `Đang tối ưu ảnh ${completed}/${files.length}…`;
      const progress = Math.round(3 + (completed / files.length) * 7);
      elements.progressValue.textContent = `${progress}%`;
      elements.progressBar.style.width = `${progress}%`;
    }
  }
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  const optimisedBytes = output.reduce((sum, file) => sum + file.size, 0);
  const savedPercent = originalBytes > 0
    ? Math.max(0, Math.round((1 - optimisedBytes / originalBytes) * 100))
    : 0;
  elements.uploadProgress.querySelector(".progress-copy span").textContent =
    `Đã tối ưu ${files.length} ảnh: ${formatBytes(originalBytes)} → ${formatBytes(optimisedBytes)}${savedPercent ? ` (giảm ${savedPercent}%)` : ""}.`;
  return output;
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

function lockRemoteSession(message = "", forgetCredentials = true) {
  if (!remoteWebhookMode) return;
  state.apiAuthorization = "";
  state.health = null;
  if (forgetCredentials) {
    safeStorage.removeItem(remoteAuthStorageKey);
    safeStorage.removeItem(remoteUsernameStorageKey);
  }
  document.documentElement.classList.remove("authenticated");
  if (message) showMessage(elements.remoteLoginError, message, "error");
  else hideMessage(elements.remoteLoginError);
  window.setTimeout(() => elements.remoteUsername.focus(), 0);
}

function rememberRemoteSession(username, authorization) {
  safeStorage.setItem(remoteAuthStorageKey, authorization);
  safeStorage.setItem(remoteUsernameStorageKey, username);
}

function restoreRemoteSession() {
  const authorization = String(
    safeStorage.getItem(remoteAuthStorageKey) || "",
  ).trim();
  const username = String(
    safeStorage.getItem(remoteUsernameStorageKey) || "",
  ).trim();
  if (!/^Basic\s+[A-Za-z0-9+/]+={0,2}$/.test(authorization)) {
    safeStorage.removeItem(remoteAuthStorageKey);
    safeStorage.removeItem(remoteUsernameStorageKey);
    return false;
  }
  state.apiAuthorization = authorization;
  elements.remoteUsername.value = username;
  return true;
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
    const manufacturingOption = elements.documentType.querySelector(
      'option[value="MANUFACTURING"]',
    );
    if (manufacturingOption) {
      manufacturingOption.disabled = data.manufacturing_configured === false;
      if (manufacturingOption.disabled && elements.documentType.value === "MANUFACTURING") {
        elements.documentType.value = "PURCHASE_RECEIPT";
      }
    }
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

function selectFiles(fileList) {
  const incoming = Array.from(fileList).filter((file) => imageContentType(file));
  state.files = incoming;
  state.selectionError = incoming.length
    ? ""
    : "Không tìm thấy ảnh JPG, PNG, WEBP hoặc BMP hợp lệ.";

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
  const inputError = state.selectionError;
  state.selectionError = inputError;
  if (state.files.length > maxImages) {
    state.selectionError = `Mỗi job nhận tối đa ${maxImages} ảnh của cùng một SKU.`;
  } else if (state.files.some((file) => file.size > maxBytes)) {
    state.selectionError = `Có ảnh vượt quá giới hạn ${maxImageSizeMb} MB.`;
  } else if (totalBytes > maxTotalBytes) {
    // The browser compresses/resizes before presign, so validate the actual
    // optimized bytes later instead of rejecting high-resolution originals.
    state.selectionError = "";
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
  elements.selectionSummary.textContent = count ? `${count} ảnh · ${formatBytes(state.files.reduce((sum, file) => sum + file.size, 0))} · sẽ tự resize/compress trước upload` : "";
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

async function runRemoteDeveloperRequest(action, payload = {}) {
  const queued = await requestJson(webhookUrl("bakery-developer-settings"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action,
      developer_key: state.developerKey,
      ...payload,
    }),
  });
  throwWorkflowError(queued, "Không thể gửi yêu cầu cấu hình developer.");
  const requestId = queued.request_id;
  if (!requestId) throw new Error("n8n không trả về mã yêu cầu developer.");

  for (let attempt = 0; attempt < 180; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 500));
    const result = await requestJson(
      webhookUrl("bakery-developer-settings-status", { request_id: requestId }),
      { cache: "no-store" },
    );
    const status = String(result.status || "").toUpperCase();
    if (status === "COMPLETED") return result;
    if (status === "ERROR") throw new Error(apiError(result, "Cấu hình developer thất bại."));
  }
  throw new Error("Worker không phản hồi cấu hình developer trong thời gian cho phép.");
}

async function developerRequest(action, payload = {}) {
  if (remoteWebhookMode) return runRemoteDeveloperRequest(action, payload);
  const isUpdate = action === "UPDATE";
  return requestJson(
    isUpdate
      ? "/api/v1/bakery/developer/settings"
      : "/api/v1/bakery/developer/settings/query",
    {
      method: isUpdate ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ developer_key: state.developerKey, ...payload }),
    },
  );
}

function renderDeveloperSettings(settings) {
  state.developerSettings = settings;
  elements.developerModel.replaceChildren();
  (settings.available_models || []).forEach((modelName) => {
    const option = document.createElement("option");
    option.value = modelName;
    option.textContent = modelName;
    option.selected = modelName === settings.active_model;
    elements.developerModel.appendChild(option);
  });

  elements.developerThresholdRows.replaceChildren();
  (settings.classes || []).forEach((item) => {
    const row = document.createElement("tr");
    const rawClass = document.createElement("td");
    const displayName = document.createElement("td");
    const thresholdCell = document.createElement("td");
    const input = document.createElement("input");
    rawClass.textContent = item.raw_class || "";
    displayName.textContent = item.display_name || "";
    input.className = "developer-threshold-input";
    input.type = "number";
    input.min = "0.01";
    input.max = "1";
    input.step = "0.01";
    input.value = Number(item.confidence_threshold || 0.25).toFixed(2);
    input.dataset.rawClass = item.raw_class || "";
    input.setAttribute("aria-label", `Confidence ${item.raw_class || "class"}`);
    thresholdCell.appendChild(input);
    row.append(rawClass, displayName, thresholdCell);
    elements.developerThresholdRows.appendChild(row);
  });
  elements.developerUpdatedAt.textContent = settings.updated_at
    ? `Lưu gần nhất: ${new Date(settings.updated_at).toLocaleString("vi-VN")}`
    : "Đang dùng cấu hình mặc định.";
  elements.developerUnlockForm.classList.add("hidden");
  elements.developerConfigForm.classList.remove("hidden");
}

function setDeveloperBusy(busy, label = "") {
  state.developerBusy = busy;
  elements.developerLoad.disabled = busy;
  elements.developerSave.disabled = busy;
  if (label) showMessage(elements.developerMessage, label);
}

function openDeveloperSettings() {
  if (
    remoteWebhookMode
    && !document.documentElement.classList.contains("authenticated")
  ) return;
  elements.developerOverlay.classList.remove("hidden");
  elements.developerKey.focus();
}

function closeDeveloperSettings() {
  if (state.developerBusy) return;
  elements.developerOverlay.classList.add("hidden");
  elements.developerConfigForm.classList.add("hidden");
  elements.developerUnlockForm.classList.remove("hidden");
  elements.developerKey.value = "";
  state.developerKey = "";
  state.developerSettings = null;
  hideMessage(elements.developerMessage);
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
  const payload = { files, inference_mode: state.inferenceMode };
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


async function confirmOrchestratedJob(jobId, payload) {
  const confirmed = await requestJson(
    remoteWebhookMode
      ? webhookUrl("bakery-confirm")
      : `${orchestratorBase}/jobs/${encodeURIComponent(jobId)}/confirm`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, ...payload }),
    },
  );
  throwWorkflowError(confirmed, "Không thể gửi xác nhận.");
  return confirmed;
}

async function confirmDirectJob(jobId, payload) {
  return requestJson(
    `/api/v1/bakery/jobs/${encodeURIComponent(jobId)}/confirm`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
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
  safeStorage.setItem(
    activeJobStorageKey,
    JSON.stringify({
      job_id: jobId,
      upload_mode: state.uploadMode,
      submit_payload: state.lastSubmitPayload,
      inference_mode: state.inferenceMode,
      saved_at: Date.now(),
    }),
  );
}

function forgetActiveJob() {
  safeStorage.removeItem(
    activeJobStorageKey,
  );

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
    state.inferenceMode = elements.inferenceMode.value;
    const uploadFiles = await optimiseFilesForUpload(state.files);
    const prepared = await prepareOrchestratedUploads(
      uploadFiles.map((file) => ({
        filename: file.name,
        content_type: imageContentType(file),
        size_bytes: file.size,
      })),
    );
    if (!Array.isArray(prepared.uploads) || prepared.uploads.length !== uploadFiles.length) {
      throw new Error("n8n không trả về đủ URL upload R2.");
    }

    await uploadFilesToR2(uploadFiles, prepared.uploads);

    elements.uploadProgress.querySelector(".progress-copy span").textContent =
      "Đang giao job cho AI…";
    const submitPayload = {
      job_id: prepared.job_id,
      inference_mode: state.inferenceMode,
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
    elements.progressValue.textContent = "100%";
    elements.progressBar.style.width = "100%";
    elements.processingSection.classList.remove("hidden");
    elements.processingSection.querySelector(".spinner").classList.remove("hidden");
    elements.decisionSection.classList.add("hidden");
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
    state.inferenceMode = elements.inferenceMode.value;
    const uploadFiles = await optimiseFilesForUpload(state.files);
    const formData = new FormData();
    uploadFiles.forEach((file) => formData.append("files", file, file.name));
    formData.append("inference_mode", state.inferenceMode);
    const accepted = await requestJson("/api/v1/bakery/jobs", {
      method: "POST",
      body: formData,
    });
    state.jobId = accepted.job_id;
    state.lastSubmitPayload = null;
    state.pollRetryCount = 0;
    rememberActiveJob(state.jobId);
    elements.progressValue.textContent = "100%";
    elements.progressBar.style.width = "100%";
    elements.processingSection.classList.remove("hidden");
    elements.processingSection.querySelector(".spinner").classList.remove("hidden");
    elements.decisionSection.classList.add("hidden");
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
      ? await requestJson(
          `/api/v1/bakery/jobs/${encodeURIComponent(state.jobId)}`,
          { cache: "no-store" },
        )
      : await readOrchestratedJob(state.jobId);

    const status = String(job.status || "").toUpperCase();

    if (
      remoteWebhookMode
      && status === "ERROR"
      && isJobNotFound(job)
      && state.pollRetryCount < maxTransientPollRetries
    ) {
      state.pollRetryCount += 1;
      elements.processingTitle.textContent = "Đang đồng bộ job với AI…";
      elements.processingCopy.textContent =
        `Đang xác nhận job (${state.pollRetryCount}/${maxTransientPollRetries})…`;

      if (
        state.lastSubmitPayload
        && (state.pollRetryCount === 3 || state.pollRetryCount === 10)
      ) {
        await submitOrchestratedJob(state.lastSubmitPayload);
      }

      scheduleJobPoll();
      return;
    }

    state.currentJob = job;
    const progressText = `Đã xử lý ${job.processed_images || 0}/${job.total_images || 0} ảnh.`;
    elements.processingCopy.textContent = job.message
      ? `${job.message} ${progressText}`
      : progressText;

    if (["AWAITING_CONFIRMATION", "NEEDS_RETAKE"].includes(status)) {
      // n8n receives worker progress and the final linked job state in two HTTP
      // callbacks. A very fast browser poll can see the terminal status from the
      // progress callback before `decision`, `images` and `product_catalog` are
      // merged by the final callback. Never render that partial state as
      // AMBIGUOUS; keep polling until the decision payload is present.
      const hasDecisionPayload = Boolean(
        job.decision
        && typeof job.decision === "object"
        && String(job.decision.decision || "").trim(),
      );
      if (!hasDecisionPayload) {
        state.pollRetryCount = 0;
        elements.processingSection.classList.remove("hidden");
        elements.processingSection.querySelector(".spinner").classList.remove("hidden");
        elements.processingTitle.textContent = "Đang đồng bộ kết quả AI…";
        elements.processingCopy.textContent =
          "AI đã xử lý xong. Đang nhận class, count và ảnh kết quả đầy đủ từ worker.";
        scheduleJobPoll(400);
        return;
      }

      state.pollRetryCount = 0;
      elements.processingSection.classList.add("hidden");
      elements.uploadProgress.classList.add("hidden");
      if (status === "NEEDS_RETAKE") forgetActiveJob();
      renderDecision(job);
      return;
    }

    if (status === "CONFIRMING") {
      state.pollRetryCount = 0;
      if (job.confirmation_error) {
        elements.processingSection.classList.add("hidden");
        renderDecision(job);
        return;
      }
      elements.processingSection.classList.remove("hidden");
      elements.processingSection.querySelector(".spinner").classList.remove("hidden");
      const manufacturing = String(job.document_type || "").toUpperCase() === "MANUFACTURING";
      elements.processingTitle.textContent = manufacturing
        ? "Đang tạo phiếu sản xuất…"
        : "Đang tạo phiếu nhập hàng…";
      elements.processingCopy.textContent = manufacturing
        ? "Đã xác nhận kết quả AI. Worker đang điều khiển KiotViet để tạo phiếu sản xuất."
        : "Đã xác nhận kết quả AI. Worker đang đối chiếu productCode và tạo phiếu nhập hàng.";
      scheduleJobPoll();
      return;
    }

    if (status === "COMPLETED") {
      state.pollRetryCount = 0;
      forgetActiveJob();
      elements.processingSection.classList.add("hidden");
      elements.uploadProgress.classList.add("hidden");
      elements.decisionSection.classList.add("hidden");
      renderResults(job);
      return;
    }

    if (status === "ERROR") {
      if (remoteWebhookMode && isJobNotFound(job)) {
        forgetActiveJob();
        throw new Error(
          "n8n chưa đồng bộ được job sau nhiều lần thử. Vui lòng kiểm tra worker rồi thực hiện lại.",
        );
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
      elements.processingCopy.textContent =
        `Kết nối tạm thời gián đoạn, đang thử lại (${state.pollRetryCount}/${maxTransientPollRetries})…`;
      scheduleJobPoll();
      return;
    }

    elements.processingTitle.textContent = "Xử lý không thành công";
    elements.processingCopy.textContent = error.message;
    elements.processingSection.querySelector(".spinner").classList.add("hidden");
    elements.startJob.disabled = false;
  }
}

function percent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${(number * 100).toFixed(1)}%`;
}

function renderAnnotatedGrid(container, images = []) {
  container.replaceChildren();

  images.forEach((image) => {
    const wrapper = image.annotated_url
      ? document.createElement("a")
      : document.createElement("div");

    wrapper.className = "annotated-item";

    if (image.annotated_url) {
      wrapper.href = image.annotated_url;
      wrapper.target = "_blank";
      wrapper.rel = "noopener";
    }

    if (image.annotated_url) {
      const picture = document.createElement("img");
      picture.src = image.annotated_url;
      picture.alt = `Ảnh nhận diện ${image.image_name || ""}`;
      picture.loading = "lazy";
      wrapper.appendChild(picture);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "annotated-placeholder";
      placeholder.textContent = "Ảnh annotated chưa có link tải";
      wrapper.appendChild(placeholder);
    }

    const info = document.createElement("div");
    const name = document.createElement("span");
    const count = document.createElement("span");
    name.textContent = image.image_name || "Ảnh";
    count.textContent = `${Number(image.total_detections || 0)} box`;
    info.append(name, count);
    wrapper.appendChild(info);
    container.appendChild(wrapper);
  });
}

function renderDecision(job) {
  state.currentJob = job;
  state.productCatalog = Array.isArray(job.product_catalog) ? job.product_catalog : [];
  state.selectedProductCode = "";
  elements.resultsSection.classList.add("hidden");
  elements.decisionSection.classList.remove("hidden");
  elements.documentType.value = String(job.document_type || "").toUpperCase() === "MANUFACTURING"
    ? "MANUFACTURING"
    : "PURCHASE_RECEIPT";

  const decision = job.decision || {};
  const type = String(decision.decision || "AMBIGUOUS").toUpperCase();
  const isConfirmable = ["DIRECT", "FAMILY", "REVIEW"].includes(type);

  // Old jobs may not contain product_catalog; keep their known candidates usable.
  if (!state.productCatalog.length) {
    const fallback = [];
    if (decision.product_code) fallback.push({
      product_code: decision.product_code,
      product_name: decision.product_name || decision.display_name || decision.product_code,
      display_name: decision.display_name || decision.product_name || decision.product_code,
    });
    [...(decision.members || []), ...(decision.candidates || [])].forEach((item) => {
      if (item?.product_code) fallback.push(item);
    });
    const seen = new Set();
    state.productCatalog = fallback.filter((item) => {
      const key = String(item.product_code || "").toLowerCase();
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  elements.decisionJobReference.textContent = `JOB ${job.job_id}`;
  elements.decisionClass.textContent = decision.display_name || decision.dominant_class || "Cần review";
  elements.decisionCount.textContent = Number(decision.count || 0);
  elements.decisionImages.textContent = `${Number(job.processed_images || 0)}/${Number(job.total_images || 0)}`;
  elements.decisionPurity.textContent = percent(decision.purity);
  elements.decisionConfidence.textContent = percent(decision.avg_confidence);
  elements.decisionBadge.textContent = type.replaceAll("_", " ");
  elements.decisionBadge.dataset.kind = type.toLowerCase();

  if (type === "DIRECT") {
    elements.decisionTitle.textContent = `${decision.display_name || decision.product_name || "SKU"} · ${Number(decision.count || 0)} sản phẩm`;
  } else if (type === "FAMILY") {
    elements.decisionTitle.textContent = `${decision.display_name || "Family"} · ${Number(decision.count || 0)} sản phẩm`;
  } else if (type === "REVIEW") {
    elements.decisionTitle.textContent = "AI chưa đồng thuận hoàn toàn — hãy chọn sản phẩm đúng";
  } else if (type === "NO_DETECTION") {
    elements.decisionTitle.textContent = "Không phát hiện sản phẩm";
  } else {
    elements.decisionTitle.textContent = "Cần chụp lại hoặc review";
  }
  elements.decisionMessage.textContent = decision.message || job.message || (isConfirmable
    ? "Kiểm tra class và count trước khi xác nhận."
    : "Job chưa đủ điều kiện tạo phiếu.");

  elements.familySkuSelect.replaceChildren();
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "— Chọn sản phẩm —";
  elements.familySkuSelect.appendChild(defaultOption);

  elements.productOverride.classList.add("hidden");
  elements.productSuggestions.classList.add("hidden");
  elements.productSearchInput.value = "";
  elements.selectedProductSummary.textContent = "Chưa chọn sản phẩm xác nhận.";
  elements.selectedProductSummary.classList.remove("selected");

  if (type === "DIRECT") {
    elements.familySelection.classList.add("hidden");
    elements.productOverride.classList.remove("hidden");
    const detected = catalogProduct(decision.product_code);
    if (detected) selectConfirmedProduct(detected.product_code);
    else openProductOverride({ clear: true });
  } else if (["FAMILY", "REVIEW"].includes(type)) {
    const choices = type === "FAMILY" ? (decision.members || []) : (decision.candidates || []);
    choices.forEach((member) => {
      const option = document.createElement("option");
      option.value = member.product_code || "";
      const source = member.source ? `${member.source} · ` : "";
      const candidateCount = Number(member.count || 0) > 0 ? ` · count ${Number(member.count)}` : "";
      option.textContent = `${source}${member.display_name || member.product_name || member.product_code} · ${member.product_code}${candidateCount}`;
      elements.familySkuSelect.appendChild(option);
    });
    const other = document.createElement("option");
    other.value = "__OTHER__";
    other.textContent = "Khác — tìm trong toàn bộ sản phẩm đã train";
    elements.familySkuSelect.appendChild(other);
    elements.familySelectionLabel.textContent = type === "REVIEW"
      ? "Chọn gợi ý đúng hoặc chọn Khác"
      : "Chọn SKU con trong family hoặc Khác";
    if (elements.familySelectionHelp) {
      elements.familySelectionHelp.textContent = type === "REVIEW"
        ? "AI chưa đồng thuận hoàn toàn về class/SKU. Hãy chọn sản phẩm đúng trước khi xác nhận KiotViet."
        : "AI chỉ xác định được family hình ảnh; hãy chọn đúng SKU con để gửi productCode sang KiotViet.";
    }
    elements.familySelection.classList.remove("hidden");
    if (!choices.length) {
      elements.familySkuSelect.value = "__OTHER__";
      openProductOverride({ clear: true });
    }
  } else {
    elements.familySelection.classList.add("hidden");
    elements.productOverride.classList.add("hidden");
  }

  elements.confirmedQuantity.value = Number(decision.count || 0) > 0 ? String(Number(decision.count)) : "";

  const perImage = Array.isArray(decision.per_image)
    ? decision.per_image
    : (job.images || []).map((image) => ({
        image_name: image.image_name,
        count: image.decision?.count ?? image.total_detections ?? 0,
        purity: image.decision?.purity,
        avg_confidence: image.decision?.avg_confidence ?? image.avg_confidence,
      }));
  elements.decisionImageRows.replaceChildren();
  perImage.forEach((image) => {
    const row = document.createElement("tr");
    const values = [image.image_name || "Ảnh", Number(image.count || 0), percent(image.purity), percent(image.avg_confidence)];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index > 0) cell.className = "number";
      row.appendChild(cell);
    });
    elements.decisionImageRows.appendChild(row);
  });
  renderAnnotatedGrid(elements.decisionAnnotatedGrid, job.images || []);

  elements.confirmJob.classList.toggle("hidden", !isConfirmable);
  elements.confirmJob.textContent = elements.documentType.value === "MANUFACTURING"
    ? "Xác nhận sản phẩm, count và tạo phiếu sản xuất"
    : "Xác nhận sản phẩm, count và tạo phiếu nhập hàng";
  if (job.confirmation_error) showMessage(elements.confirmationError, job.confirmation_error, "error");
  else hideMessage(elements.confirmationError);
  updateConfirmEnabled();

  elements.startJob.disabled = false;
  elements.decisionSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderResults(job) {
  state.currentJob = job;
  elements.resultsSection.classList.remove("hidden");
  byId("job-reference").textContent = `JOB ${job.job_id}`;
  byId("metric-products").textContent = Number(job.product_count || 0);
  byId("metric-quantity").textContent = Number(job.total_quantity || 0);
  byId("metric-images").textContent = Number(job.processed_images || 0);
  byId("metric-r2").textContent = job.r2_objects?.length ? "Đã lưu" : "Chưa lưu";

  const excel = byId("download-excel");
  if (job.excel_url) {
    excel.href = job.excel_url;
    excel.classList.remove("hidden");
  } else {
    excel.removeAttribute("href");
    excel.classList.add("hidden");
  }

  const rows = byId("product-rows");
  rows.replaceChildren();
  (job.products || []).forEach((product) => {
    const row = document.createElement("tr");
    [
      product.product_code,
      product.product_name,
      product.quantity,
      `${Number(product.purchase_price || 0).toLocaleString("vi-VN")} đ`,
    ].forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value ?? "";
      if (index >= 2) cell.className = "number";
      row.appendChild(cell);
    });
    rows.appendChild(row);
  });

  renderAnnotatedGrid(byId("annotated-grid"), job.images || []);

  const existingReceipt = job.kiotviet?.created === true;
  if (existingReceipt) {
    const receipt = job.kiotviet.receipt || {};
    const manufacturing = String(
      job.document_type || job.kiotviet.document_type || "",
    ).toUpperCase() === "MANUFACTURING";
    const productId = job.kiotviet.resolved_product_id
      || job.confirmed_product?.product_id;
    showMessage(
      elements.kiotvietMessage,
      `Đã tạo ${manufacturing ? "phiếu sản xuất" : "phiếu nhập hàng"} ${receipt.code || "KiotViet"}`
      + `${productId ? ` · productId ${productId}` : ""}.`,
      "success",
    );
  } else {
    showMessage(
      elements.kiotvietMessage,
      "Job đã hoàn tất nhưng chưa ghi nhận phiếu KiotViet. Vui lòng kiểm tra trạng thái backend.",
      "error",
    );
  }

  elements.startJob.disabled = false;
  elements.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function submitConfirmation() {
  if (state.confirmationInProgress || !state.currentJob?.job_id) return;
  const decision = state.currentJob.decision || {};
  const type = String(decision.decision || "").toUpperCase();
  if (!["DIRECT", "FAMILY", "REVIEW"].includes(type)) return;

  const confirmedQuantity = Number(elements.confirmedQuantity.value);
  if (!Number.isInteger(confirmedQuantity) || confirmedQuantity < 1 || confirmedQuantity > 5000) {
    showMessage(elements.confirmationError, "Số lượng xác nhận phải là số nguyên từ 1 đến 5000.", "error");
    return;
  }
  if (!state.selectedProductCode || !catalogProduct(state.selectedProductCode)) {
    showMessage(elements.confirmationError, "Hãy chọn một sản phẩm trong danh sách đã train trước khi xác nhận.", "error");
    return;
  }

  const selected = catalogProduct(state.selectedProductCode);
  const documentType = elements.documentType.value === "MANUFACTURING"
    ? "MANUFACTURING"
    : "PURCHASE_RECEIPT";
  const documentLabel = documentType === "MANUFACTURING"
    ? "phiếu sản xuất"
    : "phiếu nhập hàng";
  const payload = {
    confirm: true,
    quantity: confirmedQuantity,
    product_code: selected.product_code,
    document_type: documentType,
  };
  const confirmed = window.confirm(
    `Xác nhận ${productLabel(selected)} · số lượng ${confirmedQuantity} và tạo ${documentLabel} thật trên KiotViet?`,
  );
  if (!confirmed) return;

  state.confirmationInProgress = true;
  updateConfirmEnabled();
  hideMessage(elements.confirmationError);
  try {
    let result;
    if (state.uploadMode === "direct") result = await confirmDirectJob(state.currentJob.job_id, payload);
    else result = await confirmOrchestratedJob(state.currentJob.job_id, payload);
    state.currentJob = result;
    if (String(result.status || "").toUpperCase() === "COMPLETED") {
      forgetActiveJob();
      elements.decisionSection.classList.add("hidden");
      renderResults(result);
      return;
    }
    rememberActiveJob(state.currentJob.job_id);
    elements.decisionSection.classList.add("hidden");
    elements.processingSection.classList.remove("hidden");
    elements.processingSection.querySelector(".spinner").classList.remove("hidden");
    elements.processingTitle.textContent = `Đang tạo ${documentLabel}…`;
    elements.processingCopy.textContent = documentType === "MANUFACTURING"
      ? "Worker đang điều khiển KiotViet để tạo phiếu sản xuất; vui lòng không tắt máy AI."
      : "Đang xác thực SKU với KiotViet và chờ worker hoàn tất.";
    scheduleJobPoll(400);
  } catch (error) {
    showMessage(elements.confirmationError, error.message, "error");
  } finally {
    state.confirmationInProgress = false;
    updateConfirmEnabled();
  }
}

function resetForRetake() {
  clearTimeout(state.pollTimer);
  forgetActiveJob();
  state.jobId = null;
  state.currentJob = null;
  state.productCatalog = [];
  state.selectedProductCode = "";
  state.files = [];
  state.selectionError = "";
  // Every fresh counting session must start in AUTO.  A resumed active job
  // intentionally retains its recorded mode so its status stays truthful.
  state.inferenceMode = defaultInferenceMode;
  elements.inferenceMode.value = defaultInferenceMode;
  elements.documentType.value = "PURCHASE_RECEIPT";
  elements.fileInput.value = "";
  elements.folderInput.value = "";
  elements.decisionSection.classList.add("hidden");
  elements.resultsSection.classList.add("hidden");
  elements.processingSection.classList.add("hidden");
  elements.uploadProgress.classList.add("hidden");
  renderSelection();
  byId("upload-panel").scrollIntoView({ behavior: "smooth", block: "start" });
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
elements.confirmJob.addEventListener("click", submitConfirmation);
elements.retakeJob.addEventListener("click", resetForRetake);
elements.familySkuSelect.addEventListener("change", () => {
  const value = elements.familySkuSelect.value.trim();
  hideMessage(elements.confirmationError);
  if (value === "__OTHER__") {
    openProductOverride({ clear: true });
  } else if (value) {
    elements.productOverride.classList.add("hidden");
    selectConfirmedProduct(value);
  } else {
    state.selectedProductCode = "";
    elements.productOverride.classList.add("hidden");
    updateConfirmEnabled();
  }
});
elements.confirmedQuantity.addEventListener("input", updateConfirmEnabled);
elements.documentType.addEventListener("change", () => {
  elements.confirmJob.textContent = elements.documentType.value === "MANUFACTURING"
    ? "Xác nhận và tạo phiếu sản xuất"
    : "Xác nhận và tạo phiếu nhập hàng";
  updateConfirmEnabled();
});
elements.productSearchInput.addEventListener("input", () => {
  state.selectedProductCode = "";
  elements.selectedProductSummary.textContent = "Chọn một gợi ý bên dưới để xác nhận class.";
  elements.selectedProductSummary.classList.remove("selected");
  renderProductSuggestions(elements.productSearchInput.value);
  updateConfirmEnabled();
});
elements.productSearchInput.addEventListener("focus", () => {
  renderProductSuggestions(elements.productSearchInput.value);
});
document.addEventListener("click", (event) => {
  if (!event.target.closest("#product-override")) {
    elements.productSuggestions.classList.add("hidden");
  }
});

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
    rememberRemoteSession(username, state.apiAuthorization);
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

let developerHoldTimer = null;
let developerClickTimes = [];
elements.developerTrigger.addEventListener("pointerdown", () => {
  window.clearTimeout(developerHoldTimer);
  developerHoldTimer = window.setTimeout(openDeveloperSettings, 1200);
});
["pointerup", "pointercancel", "pointerleave"].forEach((eventName) => {
  elements.developerTrigger.addEventListener(eventName, () => {
    window.clearTimeout(developerHoldTimer);
  });
});
elements.developerTrigger.addEventListener("click", (event) => {
  event.preventDefault();
  const now = Date.now();
  developerClickTimes = developerClickTimes.filter((value) => now - value < 4000);
  developerClickTimes.push(now);
  if (developerClickTimes.length >= 7) {
    developerClickTimes = [];
    openDeveloperSettings();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "d") {
    event.preventDefault();
    openDeveloperSettings();
  } else if (event.key === "Escape" && !elements.developerOverlay.classList.contains("hidden")) {
    closeDeveloperSettings();
  }
});
elements.developerClose.addEventListener("click", closeDeveloperSettings);
elements.developerOverlay.addEventListener("click", (event) => {
  if (event.target === elements.developerOverlay) closeDeveloperSettings();
});
elements.developerUnlockForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.developerBusy) return;
  state.developerKey = elements.developerKey.value;
  if (!state.developerKey) return;
  setDeveloperBusy(true, "Đang đọc cấu hình từ máy AI…");
  try {
    const settings = await developerRequest("GET");
    renderDeveloperSettings(settings);
    showMessage(elements.developerMessage, "Đã xác thực. Chỉ lưu khi bạn chủ động bấm áp dụng.", "success");
  } catch (error) {
    state.developerKey = "";
    showMessage(elements.developerMessage, error.message, "error");
  } finally {
    setDeveloperBusy(false);
  }
});
elements.developerConfigForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.developerBusy || !state.developerSettings) return;
  const thresholds = {};
  for (const input of elements.developerThresholdRows.querySelectorAll("input[data-raw-class]")) {
    const value = Number(input.value);
    if (!Number.isFinite(value) || value <= 0 || value > 1) {
      showMessage(elements.developerMessage, `Confidence của ${input.dataset.rawClass} phải trong khoảng 0.01–1.00.`, "error");
      input.focus();
      return;
    }
    thresholds[input.dataset.rawClass] = value;
  }
  const activeModel = elements.developerModel.value;
  if (!activeModel) {
    showMessage(elements.developerMessage, "Không tìm thấy model hợp lệ trong backend/models.", "error");
    return;
  }
  if (!window.confirm(`Nạp model ${activeModel} và áp dụng threshold mới cho các job tiếp theo?`)) return;
  setDeveloperBusy(true, "Đang nạp và kiểm tra model. Vui lòng không tắt máy AI…");
  try {
    const settings = await developerRequest("UPDATE", {
      active_model: activeModel,
      thresholds,
    });
    renderDeveloperSettings(settings);
    if (state.health?.model?.yolo) state.health.model.yolo.model_path = settings.active_model;
    showMessage(elements.developerMessage, "Đã áp dụng model và confidence mới cho các job tiếp theo.", "success");
  } catch (error) {
    showMessage(elements.developerMessage, error.message, "error");
  } finally {
    setDeveloperBusy(false);
  }
});

async function resumeActiveJob() {
  let active = null;

  try {
    active = JSON.parse(
      safeStorage.getItem(
        activeJobStorageKey,
      ) || "null"
    );
  } catch (_) {
    safeStorage.removeItem(
      activeJobStorageKey,
    );
  }

  if (
    !active?.job_id
    || !/^[0-9a-f]{32}$/.test(
      active.job_id
    )
  ) {
    return;
  }

  if (
    active.upload_mode === "direct"
    || active.upload_mode === "orchestrated"
  ) {
    state.uploadMode =
      active.upload_mode;
  }

  state.jobId =
    active.job_id;

  state.lastSubmitPayload =
    active.submit_payload || null;

  // The job already stores its own mode server-side.  Keep the picker at
  // AUTO so every newly opened/reused web session starts from the operating
  // default rather than inheriting a previous job's manual YOLO/Foundation mode.
  state.inferenceMode = defaultInferenceMode;
  elements.inferenceMode.value = defaultInferenceMode;

  state.pollRetryCount = 0;

  elements.processingSection
    .classList.remove("hidden");

  elements.processingSection
    .querySelector(".spinner")
    .classList.remove("hidden");

  elements.processingTitle.textContent =
    "Đang tiếp tục job trước…";

  elements.processingCopy.textContent =
    `Job ${active.job_id}`;

  scheduleJobPoll(
    initialJobPollDelayMs
  );
}

(async () => {
  if (remoteWebhookMode) {
    if (!restoreRemoteSession()) {
      lockRemoteSession("", false);
      return;
    }
    document.documentElement.classList.add("authenticated");
    const ready = await checkHealth();
    if (ready) {
      await resumeActiveJob();
    } else {
      document.documentElement.classList.remove("authenticated");
    }
    return;
  }
  await checkHealth();
  await resumeActiveJob();
})();