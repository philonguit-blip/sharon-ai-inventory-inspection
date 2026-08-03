import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const directory = dirname(fileURLToPath(import.meta.url));
const outputPath = join(directory, "Workflow 4_ Sharon Bakery Outbound Worker.json");

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
        id: "REPLACE_AFTER_IMPORT",
        name: "Sharon Bakery Backend API",
      },
    },
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

const initUploadCode = String.raw`
const store = $getWorkflowStaticData('global');
store.tasks = store.tasks || {};
store.queue = store.queue || [];
store.requests = store.requests || {};
store.jobs = store.jobs || {};
const body = $json.body || {};
const files = Array.isArray(body.files) ? body.files : [];
const allowed = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp']);
const maxImages = 50;
const maxFile = 50 * 1024 * 1024;
const maxTotalMb = Math.max(1, Number(store.worker?.max_job_upload_size_mb || 160));
const maxTotal = maxTotalMb * 1024 * 1024;
let error = '';
if (!files.length) error = 'Upload at least one image.';
else if (files.length > maxImages) error = 'A job accepts at most 50 images.';
else if (files.some(f => !allowed.has(String(f.content_type || '')))) error = 'Unsupported image format.';
else if (files.some(f => Number(f.size_bytes || 0) <= 0 || Number(f.size_bytes) > maxFile)) error = 'An image exceeds the 50 MB limit.';
else if (files.reduce((sum, f) => sum + Number(f.size_bytes || 0), 0) > maxTotal) error = 'Total upload exceeds the ' + maxTotalMb + ' MB limit.';
if (error) return [{ json: { status: 'ERROR', error, detail: error } }];
const jobId = Array.from(
  { length: 32 },
  () => Math.floor(Math.random() * 16).toString(16),
).join('');
const now = Date.now();
const task = { task_id: jobId, job_id: jobId, task_type: 'PRESIGN', payload: { files }, status: 'QUEUED', attempts: 0, created_at: now, updated_at: now };
store.tasks[jobId] = task;
store.queue.push(jobId);
store.requests[jobId] = { request_id: jobId, job_id: jobId, status: 'WAITING_FOR_WORKER', created_at: new Date(now).toISOString() };
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
if (!/^[0-9a-f]{32}$/.test(jobId) || !files.length) {
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
  || !expectedFiles.length
  || JSON.stringify(suppliedKeys) !== JSON.stringify(expectedKeys)
) {
  const detail = 'R2 files do not match a ready upload session.';
  return [{ json: { status: 'ERROR', error: detail, detail } }];
}
const now = Date.now();
const taskId = 'process-' + jobId;
store.tasks[taskId] = { task_id: taskId, job_id: jobId, task_type: 'PROCESS', payload: { job_id: jobId, files: expectedFiles }, status: 'QUEUED', attempts: 0, created_at: now, updated_at: now };
store.queue.push(taskId);
store.jobs[jobId] = { job_id: jobId, status: 'QUEUED', total_images: files.length, processed_images: 0, created_at: new Date(now).toISOString(), updated_at: new Date(now).toISOString(), products: [], images: [], r2_objects: [], error: '' };
return [{ json: { ...store.jobs[jobId], status_url: '/webhook/bakery-job-status?job_id=' + jobId, message: 'Job queued for the outbound AI worker.' } }];`;

const jobStatusCode = String.raw`
const store = $getWorkflowStaticData('global');
store.jobs = store.jobs || {};
const jobId = String(($json.query || {}).job_id || '');
const job = store.jobs[jobId];
if (!job) return [{ json: { status: 'ERROR', error: 'Job not found.', detail: 'Job not found.' } }];
return [{ json: job }];`;

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

const workerResultCode = String.raw`
const store = $getWorkflowStaticData('global');
store.tasks = store.tasks || {};
store.queue = store.queue || [];
store.requests = store.requests || {};
store.jobs = store.jobs || {};
const body = $json.body || {};
const taskId = String(body.task_id || '');
const task = store.tasks[taskId];
if (!task) return [{ json: { ok: false, detail: 'Task not found.' } }];
if (task.status === 'COMPLETED' || task.status === 'ERROR') {
  return [{ json: { ok: true, task_id: taskId, final: true, duplicate: true } }];
}
if (String(task.leased_by || '') !== String(body.worker_id || '')) {
  return [{ json: { ok: false, detail: 'Task lease does not belong to this worker.' } }];
}
const now = Date.now();
const result = body.result && typeof body.result === 'object' ? body.result : {};
const ok = body.ok === true;
const final = body.final === true;
const error = String(body.error || result.error || '');
task.updated_at = now;
task.leased_at = now;
if (task.task_type === 'PRESIGN') {
  if (ok && final) store.requests[task.job_id] = { ...result, request_id: task.job_id, job_id: task.job_id, status: 'READY', updated_at: new Date(now).toISOString() };
  if (!ok && final) store.requests[task.job_id] = { request_id: task.job_id, job_id: task.job_id, status: 'ERROR', error, detail: error, updated_at: new Date(now).toISOString() };
}
if (task.task_type === 'PROCESS') {
  if (ok) store.jobs[task.job_id] = { ...(store.jobs[task.job_id] || {}), ...result, job_id: task.job_id, updated_at: new Date(now).toISOString() };
  if (!ok && final) store.jobs[task.job_id] = { ...(store.jobs[task.job_id] || {}), ...result, job_id: task.job_id, status: 'ERROR', error, updated_at: new Date(now).toISOString() };
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
store.worker = { ...body, last_seen_ms: now, last_seen: new Date(now).toISOString() };
if (Number(store.last_cleanup_ms || 0) + 60 * 60 * 1000 < now) {
  store.tasks = store.tasks || {};
  store.requests = store.requests || {};
  store.jobs = store.jobs || {};
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
  kiotviet_auto_create_draft: worker.kiotviet_auto_create_draft === true,
  model: worker.model || null,
  max_images_per_job: worker.max_images_per_job || 50,
  max_image_size_mb: worker.max_image_size_mb || 50,
  max_job_upload_size_mb: worker.max_job_upload_size_mb || 160,
  allowed_image_extensions: worker.allowed_image_extensions || ['.jpg', '.jpeg', '.png', '.webp', '.bmp'],
  error: online ? '' : 'Outbound AI worker is offline.'
} }];`;

const definitions = [
  ["01 - Init Upload", "bakery-upload-init", "POST", initUploadCode, -540],
  ["01b - Upload Status", "bakery-request-status", "GET", requestStatusCode, -360],
  ["02 - Submit Job", "bakery-submit", "POST", submitJobCode, -180],
  ["03 - Job Status", "bakery-job-status", "GET", jobStatusCode, 0],
  ["04 - Worker Next", "bakery-worker-next", "GET", workerNextCode, 180],
  ["05 - Worker Result", "bakery-worker-result", "POST", workerResultCode, 360],
  ["06 - Worker Heartbeat", "bakery-worker-heartbeat", "POST", heartbeatCode, 540],
  ["00 - Health", "bakery-health", "GET", healthCode, 720],
];

const nodes = [];
const connections = {};
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
