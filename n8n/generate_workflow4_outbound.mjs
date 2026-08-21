import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const directory = dirname(fileURLToPath(import.meta.url));
const outputPath = join(directory, "Workflow 4_ Sharon Bakery Outbound Worker.json");
const frontendDirectory = join(directory, "..", "backend", "frontend");
const backendCredentialId = process.env.N8N_BACKEND_CREDENTIAL_ID
  || "gVwKQzabSgtnyWS7";

function buildEmbeddedWebUi() {
  const html = readFileSync(join(frontendDirectory, "index.html"), "utf8");
  const css = readFileSync(join(frontendDirectory, "assets", "styles.css"), "utf8");
  const app = readFileSync(join(frontendDirectory, "assets", "app.js"), "utf8");
  const remoteConfig = `<script>
window.SHARON_REMOTE_MODE = true;
window.SHARON_WEBHOOK_BASE = window.location.origin + "/webhook";
document.documentElement.classList.add("remote-mode");
</script>`;

  return html
    .replace(/<link rel="stylesheet" href="\/assets\/styles\.css\?v=\d+" \/>/, `<style>\n${css}\n</style>`)
    .replace(/\s*<script src="\/assets\/config\.js\?v=\d+"><\/script>/, `\n${remoteConfig}`)
    // An inline script in <head> does not inherit the original external
    // script's `defer` behaviour. Remove it from <head> and inject it after all
    // DOM elements so event listeners never bind to null elements.
    .replace(/\s*<script src="\/assets\/app\.js\?v=\d+" defer><\/script>/, "")
    .replace(/\s*<\/body>/, `\n<script>\n${app}\n</script>\n</body>`);
}

let idCounter = 1;
const nodeId = () => `outbound-worker-${String(idCounter++).padStart(3, "0")}`;

function webhook(name, path, method, position) {
  return {
    parameters: {
      httpMethod: method,
      path,
      authentication: "basicAuth",
      responseMode: "responseNode",
      options: { allowedOrigins: "*" },
    },
    type: "n8n-nodes-base.webhook",
    typeVersion: 2.1,
    position,
    id: nodeId(),
    name,
    webhookId: nodeId(),
    credentials: {
      httpBasicAuth: {
        id: backendCredentialId,
        name: "Sharon Bakery Backend API",
      },
    },
  };
}

function publicWebhook(name, path, position) {
  return {
    parameters: {
      path,
      responseMode: "responseNode",
      options: { allowedOrigins: "*" },
    },
    type: "n8n-nodes-base.webhook",
    typeVersion: 2.1,
    position,
    id: nodeId(),
    name,
    webhookId: nodeId(),
  };
}

function code(name, jsCode, position) {
  return {
    parameters: { jsCode },
    type: "n8n-nodes-base.code",
    typeVersion: 2,
    position,
    id: nodeId(),
    name,
  };
}

function respond(name, position) {
  return {
    parameters: {
      respondWith: "json",
      responseBody: "={{ $json }}",
      options: {},
    },
    type: "n8n-nodes-base.respondToWebhook",
    typeVersion: 1.5,
    position,
    id: nodeId(),
    name,
  };
}

function respondHtml(name, position) {
  return {
    parameters: {
      respondWith: "text",
      responseBody: "={{ $json.html }}",
      options: {
        responseHeaders: {
          entries: [{ name: "Content-Type", value: "text/html; charset=utf-8" }],
        },
      },
    },
    type: "n8n-nodes-base.respondToWebhook",
    typeVersion: 1.5,
    position,
    id: nodeId(),
    name,
  };
}

const initUploadCode = String.raw`
const store = $getWorkflowStaticData('global');
store.tasks = store.tasks || {};
store.queue = store.queue || [];
store.requests = store.requests || {};
store.jobs = store.jobs || {};
store.worker = store.worker || {};
const body = $json.body || {};
const files = Array.isArray(body.files) ? body.files : [];
const inferenceMode = String(body.inference_mode || 'AUTO').toUpperCase();
const validModes = new Set(['AUTO', 'YOLO', 'FOUNDATION', 'COMPARE']);
const allowed = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp']);
const maxImages = Math.max(1, Number(store.worker?.max_images_per_job || 50));
const maxFile = 50 * 1024 * 1024;
const maxTotalMb = Math.max(1, Number(store.worker?.max_job_upload_size_mb || 200));
const maxTotal = maxTotalMb * 1024 * 1024;
let error = '';
if (!files.length) error = 'Upload at least one image.';
else if (!validModes.has(inferenceMode)) error = 'Invalid inference mode.';
else if (files.length > maxImages) error = 'A job accepts at most ' + maxImages + ' images of the same SKU.';
else if (files.some(f => !allowed.has(String(f.content_type || '')))) error = 'Unsupported image format.';
else if (files.some(f => Number(f.size_bytes || 0) <= 0 || Number(f.size_bytes) > maxFile)) error = 'An image exceeds the 50 MB limit.';
else if (files.reduce((sum, f) => sum + Number(f.size_bytes || 0), 0) > maxTotal) error = 'Total upload exceeds the ' + maxTotalMb + ' MB limit.';
if (error) return [{ json: { status: 'ERROR', error, detail: error } }];
const jobId = Array.from(
  { length: 32 },
  () => Math.floor(Math.random() * 16).toString(16),
).join('');
const now = Date.now();
const task = { task_id: jobId, job_id: jobId, task_type: 'PRESIGN', payload: { files, inference_mode: inferenceMode }, status: 'QUEUED', attempts: 0, created_at: now, updated_at: now };
store.tasks[jobId] = task;
store.queue.push(jobId);
store.requests[jobId] = { request_id: jobId, job_id: jobId, inference_mode: inferenceMode, status: 'WAITING_FOR_WORKER', created_at: new Date(now).toISOString() };
return [{ json: store.requests[jobId] }];`;

const requestStatusCode = String.raw`
const store = $getWorkflowStaticData('global');
store.requests = store.requests || {};
const requestId = String(($json.query || {}).request_id || '');
const request = store.requests[requestId];
if (!request) return [{ json: { status: 'ERROR', error: 'Upload request not found.', detail: 'Upload request not found.' } }];
return [{ json: request }];`;

const submitJobCode = String.raw`
const store = $getWorkflowStaticData('global');
store.tasks = store.tasks || {};
store.queue = store.queue || [];
store.jobs = store.jobs || {};
const body = $json.body || {};
const jobId = String(body.job_id || '');
const files = Array.isArray(body.files) ? body.files : [];
const inferenceMode = String(body.inference_mode || 'AUTO').toUpperCase();
if (!/^[0-9a-f]{32}$/.test(jobId) || !files.length || !['AUTO', 'YOLO', 'FOUNDATION', 'COMPARE'].includes(inferenceMode)) {
  const detail = 'Invalid job ID or empty R2 file list.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}
if (store.jobs[jobId]) return [{ json: store.jobs[jobId] }];
const uploadRequest = store.requests?.[jobId];
const expectedFiles = Array.isArray(uploadRequest?.uploads)
  ? uploadRequest.uploads.map(item => ({ object_key: String(item.object_key || '') }))
  : [];
const suppliedKeys = files.map(item => String(item.object_key || '')).sort();
const expectedKeys = expectedFiles.map(item => item.object_key).sort();
if (
  uploadRequest?.status !== 'READY'
  || String(uploadRequest?.inference_mode || 'AUTO').toUpperCase() !== inferenceMode
  || !expectedFiles.length
  || JSON.stringify(suppliedKeys) !== JSON.stringify(expectedKeys)
) {
  const detail = 'R2 files do not match a ready upload session.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}
const now = Date.now();
const taskId = 'process-' + jobId;
store.tasks[taskId] = { task_id: taskId, job_id: jobId, task_type: 'PROCESS', payload: { job_id: jobId, files: expectedFiles, inference_mode: inferenceMode }, status: 'QUEUED', attempts: 0, created_at: now, updated_at: now };
store.queue.push(taskId);
store.jobs[jobId] = { job_id: jobId, inference_mode: inferenceMode, status: 'QUEUED', total_images: files.length, processed_images: 0, created_at: new Date(now).toISOString(), updated_at: new Date(now).toISOString(), products: [], images: [], r2_objects: [], error: '' };
return [{ json: { ...store.jobs[jobId], status_url: '/webhook/bakery-job-status?job_id=' + jobId, message: 'Job queued for the outbound AI worker.' } }];`;

const jobStatusCode = String.raw`
const store = $getWorkflowStaticData('global');
store.jobs = store.jobs || {};
const jobId = String(($json.query || {}).job_id || '');
const job = store.jobs[jobId];
if (!job) return [{ json: { status: 'ERROR', error: 'Job not found.', detail: 'Job not found.' } }];
return [{ json: job }];`;

const confirmJobCode = String.raw`
const store = $getWorkflowStaticData('global');
store.tasks = store.tasks || {};
store.queue = store.queue || [];
store.jobs = store.jobs || {};

const body = $json.body || {};
const jobId = String(body.job_id || '');
const selectedProductCode = String(body.product_code || '').trim();
const selectedQuantity = body.quantity == null ? null : Number(body.quantity);
const selectedDocumentType = String(body.document_type || 'PURCHASE_RECEIPT').toUpperCase();

if (body.confirm !== true) {
  const detail = 'Explicit confirmation is required.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}

if (!/^[0-9a-f]{32}$/.test(jobId)) {
  const detail = 'Invalid job ID.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}
if (selectedQuantity !== null && (!Number.isInteger(selectedQuantity) || selectedQuantity < 1 || selectedQuantity > 5000)) {
  const detail = 'Confirmed quantity must be an integer between 1 and 5000.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}
if (!['PURCHASE_RECEIPT', 'MANUFACTURING'].includes(selectedDocumentType)) {
  const detail = 'Invalid KiotViet document type.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}

const job = store.jobs[jobId];
if (!job) {
  const detail = 'Job not found.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}

const currentStatus = String(job.status || '').toUpperCase();

if (currentStatus === 'COMPLETED' && job.kiotviet?.created === true) {
  return [{ json: job }];
}

if (currentStatus === 'CONFIRMING') {
  const taskId = 'confirm-' + jobId;
  const existingTask = store.tasks[taskId];
  if (
    existingTask
    && ['QUEUED', 'LEASED'].includes(String(existingTask.status || '').toUpperCase())
  ) {
    return [{ json: job }];
  }

  // The local backend persisted CONFIRMING before calling KiotViet. A retry
  // must reach that backend so it can reconcile by job_id. Re-queueing the
  // same task ID is safe; it does not ask the backend for an unguarded POST.
  const retryPayload = { confirm: true };
  retryPayload.document_type = String(
    job.confirmation?.document_type || job.document_type || 'PURCHASE_RECEIPT',
  ).toUpperCase();
  const confirmedCode = String(job.confirmation?.product_code || '').trim();
  if (confirmedCode) retryPayload.product_code = confirmedCode;
  const confirmedQuantity = Number(job.confirmation?.quantity || 0);
  if (confirmedQuantity > 0) retryPayload.quantity = confirmedQuantity;
  const now = Date.now();
  store.tasks[taskId] = {
    task_id: taskId,
    job_id: jobId,
    task_type: 'CONFIRM',
    payload: retryPayload,
    status: 'QUEUED',
    attempts: 0,
    created_at: now,
    updated_at: now,
  };
  store.queue = store.queue.filter(id => id !== taskId);
  store.queue.push(taskId);
  store.jobs[jobId] = {
    ...job,
    confirmation_error: '',
    updated_at: new Date(now).toISOString(),
  };
  return [{ json: {
    ...store.jobs[jobId],
    message: 'Confirmation reconciliation queued for the outbound AI worker.',
  } }];
}

if (currentStatus !== 'AWAITING_CONFIRMATION') {
  const detail = 'Job is not waiting for confirmation. Current status: ' + currentStatus + '.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}

const decision = job.decision && typeof job.decision === 'object' ? job.decision : {};
const decisionType = String(decision.decision || '').toUpperCase();

if (!['DIRECT', 'FAMILY', 'REVIEW'].includes(decisionType)) {
  const detail = 'Only DIRECT, FAMILY or REVIEW decisions can be confirmed.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}

const taskPayload = { confirm: true, document_type: selectedDocumentType };
if (selectedQuantity !== null) taskPayload.quantity = selectedQuantity;

if (decisionType === 'DIRECT') {
  const expectedCode = String(decision.product_code || '').trim();
  if (!expectedCode) {
    const detail = 'Direct decision has no product code.';
    return [{ json: { status: 'ERROR', error: detail, detail } }];
  }
  if (selectedProductCode && selectedProductCode.toLowerCase() !== expectedCode.toLowerCase()) {
    const detail = 'Selected product does not match the detected direct SKU.';
    return [{ json: { status: 'ERROR', error: detail, detail } }];
  }
} else if (decisionType === 'FAMILY') {
  if (!selectedProductCode) {
    const detail = 'Select one SKU from the detected family.';
    return [{ json: { status: 'ERROR', error: detail, detail } }];
  }

  const members = Array.isArray(decision.members) ? decision.members : [];
  const valid = members.some(
    member => String(member?.product_code || '').toLowerCase() === selectedProductCode.toLowerCase(),
  );

  if (!valid) {
    const detail = 'Selected product is not a member of the detected family.';
    return [{ json: { status: 'ERROR', error: detail, detail } }];
  }

  taskPayload.product_code = selectedProductCode;
} else {
  if (!selectedProductCode) {
    const detail = 'Select the correct SKU from the hybrid review candidates.';
    return [{ json: { status: 'ERROR', error: detail, detail } }];
  }
  const candidates = Array.isArray(decision.candidates) ? decision.candidates : [];
  const valid = candidates.some(
    candidate => String(candidate?.product_code || '').toLowerCase() === selectedProductCode.toLowerCase(),
  );
  if (!valid) {
    const detail = 'Selected product is not a hybrid review candidate.';
    return [{ json: { status: 'ERROR', error: detail, detail } }];
  }
  taskPayload.product_code = selectedProductCode;
}

const taskId = 'confirm-' + jobId;
const existingTask = store.tasks[taskId];

if (
  existingTask
  && ['QUEUED', 'LEASED'].includes(String(existingTask.status || '').toUpperCase())
) {
  return [{ json: store.jobs[jobId] }];
}

const now = Date.now();
store.tasks[taskId] = {
  task_id: taskId,
  job_id: jobId,
  task_type: 'CONFIRM',
  payload: taskPayload,
  status: 'QUEUED',
  attempts: 0,
  created_at: now,
  updated_at: now,
};
store.queue.push(taskId);

store.jobs[jobId] = {
  ...job,
  status: 'CONFIRMING',
  confirmation: {
    requested_at: new Date(now).toISOString(),
    product_code: taskPayload.product_code || decision.product_code || null,
    quantity: taskPayload.quantity || decision.count || null,
    document_type: taskPayload.document_type,
  },
  document_type: taskPayload.document_type,
  confirmation_error: '',
  updated_at: new Date(now).toISOString(),
};

return [{
  json: {
    ...store.jobs[jobId],
    message: 'Confirmation queued for the outbound AI worker.',
  },
}];`;

const workerNextCode = String.raw`
const store = $getWorkflowStaticData('global');
store.tasks = store.tasks || {};
store.queue = store.queue || [];
const workerId = String(($json.query || {}).worker_id || 'unknown-worker');
const now = Date.now();
const leaseMs = 60 * 60 * 1000;
let selected = null;
for (const taskId of store.queue) {
  const task = store.tasks[taskId];
  if (!task) continue;
  const expired = task.status === 'LEASED' && Number(task.leased_at || 0) + leaseMs < now;
  if (task.status === 'QUEUED' || expired) {
    task.status = 'LEASED';
    task.leased_at = now;
    task.leased_by = workerId;
    task.attempts = Number(task.attempts || 0) + 1;
    task.updated_at = now;
    selected = { ...task };
    break;
  }
}
return [{ json: { task: selected } }];`;

const developerSettingsRequestCode = String.raw`
const store = $getWorkflowStaticData('global');
store.tasks = store.tasks || {};
store.queue = store.queue || [];
store.developerRequests = store.developerRequests || {};
const body = $json.body || {};
const action = String(body.action || 'GET').trim().toUpperCase();
const developerKey = String(body.developer_key || '');
if (!['GET', 'UPDATE'].includes(action) || !developerKey) {
  return [{ json: { status: 'ERROR', detail: 'Invalid developer settings request.' } }];
}
const requestId = Array.from(
  { length: 32 },
  () => Math.floor(Math.random() * 16).toString(16),
).join('');
const taskId = 'developer-' + requestId;
const now = Date.now();
const payload = { action, developer_key: developerKey };
if (action === 'UPDATE') {
  payload.active_model = String(body.active_model || '');
  payload.thresholds = body.thresholds && typeof body.thresholds === 'object'
    ? body.thresholds
    : {};
}
store.tasks[taskId] = {
  task_id: taskId,
  job_id: requestId,
  task_type: 'DEVELOPER_SETTINGS',
  payload,
  status: 'QUEUED',
  attempts: 0,
  created_at: now,
  updated_at: now,
};
store.queue.push(taskId);
store.developerRequests[requestId] = {
  request_id: requestId,
  status: 'QUEUED',
  action,
  created_at: new Date(now).toISOString(),
};
return [{ json: store.developerRequests[requestId] }];`;

const developerSettingsStatusCode = String.raw`
const store = $getWorkflowStaticData('global');
store.developerRequests = store.developerRequests || {};
const requestId = String(($json.query || {}).request_id || '');
const request = store.developerRequests[requestId];
if (!request) {
  return [{ json: { status: 'ERROR', detail: 'Developer request not found.' } }];
}
return [{ json: request }];`;

const workerResultCode = String.raw`
const store = $getWorkflowStaticData('global');
store.tasks = store.tasks || {};
store.queue = store.queue || [];
store.requests = store.requests || {};
store.jobs = store.jobs || {};
store.worker = store.worker || {};
store.developerRequests = store.developerRequests || {};
const body = $json.body || {};
const taskId = String(body.task_id || '');
const task = store.tasks[taskId];
if (!task) return [{ json: { ok: false, detail: 'Task not found.' } }];
if (task.status === 'COMPLETED' || task.status === 'ERROR') {
  return [{ json: { ok: true, task_id: taskId, final: true, duplicate: true } }];
}
const reportingWorker = String(body.worker_id || '');
const registeredWorker = String(store.worker.worker_id || '');
const ownsLease = String(task.leased_by || '') === reportingWorker;
const canRecoverStaleLease = Boolean(reportingWorker)
  && reportingWorker === registeredWorker
  && ['QUEUED', 'LEASED'].includes(String(task.status || '').toUpperCase());
if (!ownsLease && !canRecoverStaleLease) {
  return [{ json: { ok: false, detail: 'Task lease does not belong to this worker.' } }];
}
const now = Date.now();
if (canRecoverStaleLease) {
  task.status = 'LEASED';
  task.leased_by = reportingWorker;
}
const result = body.result && typeof body.result === 'object' ? body.result : {};
const ok = body.ok === true;
const final = body.final === true;
const error = String(body.error || result.error || '');
task.updated_at = now;
task.leased_at = now;
if (task.task_type === 'PRESIGN') {
  if (ok && final) store.requests[task.job_id] = { ...(store.requests[task.job_id] || {}), ...result, request_id: task.job_id, job_id: task.job_id, inference_mode: String(task.payload?.inference_mode || 'AUTO').toUpperCase(), status: 'READY', updated_at: new Date(now).toISOString() };
  if (!ok && final) store.requests[task.job_id] = { request_id: task.job_id, job_id: task.job_id, status: 'ERROR', error, detail: error, updated_at: new Date(now).toISOString() };
}
if (task.task_type === 'PROCESS') {
  if (ok) store.jobs[task.job_id] = { ...(store.jobs[task.job_id] || {}), ...result, job_id: task.job_id, updated_at: new Date(now).toISOString() };
  if (!ok && final) store.jobs[task.job_id] = { ...(store.jobs[task.job_id] || {}), ...result, job_id: task.job_id, status: 'ERROR', error, updated_at: new Date(now).toISOString() };
}
if (task.task_type === 'CONFIRM') {
  if (ok) {
    store.jobs[task.job_id] = {
      ...(store.jobs[task.job_id] || {}),
      ...result,
      job_id: task.job_id,
      confirmation_error: '',
      updated_at: new Date(now).toISOString(),
    };
  }
  if (!ok && final) {
    store.jobs[task.job_id] = {
      ...(store.jobs[task.job_id] || {}),
      job_id: task.job_id,
      status: 'CONFIRMING',
      confirmation_error: error || 'Confirmation failed.',
      error: '',
      updated_at: new Date(now).toISOString(),
    };
  }
}
if (task.task_type === 'DEVELOPER_SETTINGS') {
  store.developerRequests[task.job_id] = ok && final
    ? {
        ...(store.developerRequests[task.job_id] || {}),
        ...result,
        request_id: task.job_id,
        status: 'COMPLETED',
        updated_at: new Date(now).toISOString(),
      }
    : {
        ...(store.developerRequests[task.job_id] || {}),
        request_id: task.job_id,
        status: final ? 'ERROR' : 'PROCESSING',
        error,
        detail: error,
        updated_at: new Date(now).toISOString(),
      };
  if (final && task.payload) delete task.payload.developer_key;
}
if (final) {
  task.status = ok ? 'COMPLETED' : 'ERROR';
  store.queue = store.queue.filter(id => id !== taskId);
}
return [{ json: { ok: true, task_id: taskId, final } }];`;

const heartbeatCode = String.raw`
const store = $getWorkflowStaticData('global');
const body = $json.body || {};
const now = Date.now();
const previousWorkerId = String(store.worker?.worker_id || '');
const currentWorkerId = String(body.worker_id || '');
if (previousWorkerId && currentWorkerId && previousWorkerId !== currentWorkerId) {
  store.tasks = store.tasks || {};
  for (const task of Object.values(store.tasks)) {
    if (
      String(task.status || '').toUpperCase() === 'LEASED'
      && String(task.leased_by || '') === previousWorkerId
    ) {
      task.status = 'QUEUED';
      task.leased_by = '';
      task.leased_at = 0;
      task.updated_at = now;
    }
  }
}
store.worker = { ...body, last_seen_ms: now, last_seen: new Date(now).toISOString() };
if (Number(store.last_cleanup_ms || 0) + 60 * 60 * 1000 < now) {
  store.tasks = store.tasks || {};
  store.requests = store.requests || {};
  store.jobs = store.jobs || {};
  store.developerRequests = store.developerRequests || {};
  const taskCutoff = now - 24 * 60 * 60 * 1000;
  const jobCutoff = now - 30 * 24 * 60 * 60 * 1000;
  for (const [id, task] of Object.entries(store.tasks)) {
    if (['COMPLETED', 'ERROR'].includes(String(task.status || '')) && Number(task.updated_at || 0) < taskCutoff) delete store.tasks[id];
  }
  for (const [id, request] of Object.entries(store.requests)) {
    const updated = Date.parse(request.updated_at || request.created_at || 0);
    if (Number.isFinite(updated) && updated < taskCutoff) delete store.requests[id];
  }
  for (const [id, job] of Object.entries(store.jobs)) {
    const updated = Date.parse(job.updated_at || job.created_at || 0);
    if (['COMPLETED', 'ERROR'].includes(String(job.status || '')) && Number.isFinite(updated) && updated < jobCutoff) delete store.jobs[id];
  }
  for (const [id, request] of Object.entries(store.developerRequests)) {
    const updated = Date.parse(request.updated_at || request.created_at || 0);
    if (Number.isFinite(updated) && updated < taskCutoff) delete store.developerRequests[id];
  }
  store.queue = (store.queue || []).filter(id => store.tasks[id] && !['COMPLETED', 'ERROR'].includes(String(store.tasks[id].status || '')));
  store.last_cleanup_ms = now;
}
return [{ json: { ok: true, worker_id: String(body.worker_id || '') } }];`;

const healthCode = String.raw`
const store = $getWorkflowStaticData('global');
const worker = store.worker || {};
const online = Number(worker.last_seen_ms || 0) + 30000 > Date.now();
return [{ json: {
  ready: online && worker.ready === true && worker.r2_configured === true,
  orchestration: 'n8n-outbound-worker',
  worker_online: online,
  worker_id: worker.worker_id || null,
  worker_last_seen: worker.last_seen || null,
  r2_configured: worker.r2_configured === true,
  kiotviet_configured: worker.kiotviet_configured === true,
  manufacturing_configured: worker.manufacturing_configured === true,
  kiotviet_auto_create_draft: worker.kiotviet_auto_create_draft === true,
  model: worker.model || null,
  max_images_per_job: Math.max(1, Number(worker.max_images_per_job || 50)),
  max_image_size_mb: worker.max_image_size_mb || 50,
  max_job_upload_size_mb: worker.max_job_upload_size_mb || 200,
  allowed_image_extensions: worker.allowed_image_extensions || ['.jpg', '.jpeg', '.png', '.webp', '.bmp'],
  error: online ? '' : 'Outbound AI worker is offline.'
} }];`;

const definitions = [
  ["01 - Init Upload", "bakery-upload-init", "POST", initUploadCode, -540],
  ["01b - Upload Status", "bakery-request-status", "GET", requestStatusCode, -360],
  ["02 - Submit Job", "bakery-submit", "POST", submitJobCode, -180],
  ["03 - Job Status", "bakery-job-status", "GET", jobStatusCode, 0],
  ["03b - Confirm Job", "bakery-confirm", "POST", confirmJobCode, 180],
  ["04 - Worker Next", "bakery-worker-next", "GET", workerNextCode, 360],
  ["05 - Worker Result", "bakery-worker-result", "POST", workerResultCode, 540],
  ["06 - Worker Heartbeat", "bakery-worker-heartbeat", "POST", heartbeatCode, 720],
  ["00 - Health", "bakery-health", "GET", healthCode, 900],
  ["07 - Developer Settings", "bakery-developer-settings", "POST", developerSettingsRequestCode, 1080],
  ["07b - Developer Settings Status", "bakery-developer-settings-status", "GET", developerSettingsStatusCode, 1260],
];

const nodes = [];
const connections = {};
const webUiName = "00a - Web UI";
const webUiCodeName = "00a - Web UI - State";
const webUiResponseName = "00a - Web UI - Respond";
const embeddedHtml = buildEmbeddedWebUi();
nodes.push(publicWebhook(webUiName, "sharon-bakery-inventory", [-720, -720]));
nodes.push(
  code(
    webUiCodeName,
    `return [{ json: { html: ${JSON.stringify(embeddedHtml)} } }];`,
    [-420, -720],
  ),
);
nodes.push(respondHtml(webUiResponseName, [-120, -720]));
connections[webUiName] = {
  main: [[{ node: webUiCodeName, type: "main", index: 0 }]],
};
connections[webUiCodeName] = {
  main: [[{ node: webUiResponseName, type: "main", index: 0 }]],
};

for (const [name, path, method, jsCode, y] of definitions) {
  const hookName = name;
  const codeName = `${name} - State`;
  const responseName = `${name} - Respond`;
  nodes.push(webhook(hookName, path, method, [-720, y]));
  nodes.push(code(codeName, jsCode, [-420, y]));
  nodes.push(respond(responseName, [-120, y]));
  connections[hookName] = {
    main: [[{ node: codeName, type: "main", index: 0 }]],
  };
  connections[codeName] = {
    main: [[{ node: responseName, type: "main", index: 0 }]],
  };
}

const workflow = {
  name: "Sharon Bakery - Outbound AI Worker Queue",
  nodes,
  pinData: {},
  connections,
  active: false,
  settings: { executionOrder: "v1", availableInMCP: false },
  meta: { templateCredsSetupCompleted: false },
  tags: [],
};

writeFileSync(outputPath, `${JSON.stringify(workflow, null, 2)}\n`, "utf8");
console.log(outputPath);
