import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";

const directory = dirname(fileURLToPath(import.meta.url));
const workflow = JSON.parse(
  readFileSync(
    join(directory, "Workflow 4_ Sharon Bakery Outbound Worker.json"),
    "utf8",
  ),
);
const store = {};

function execute(nodeName, json) {
  const node = workflow.nodes.find((item) => item.name === nodeName);
  assert.ok(node, `Missing workflow node: ${nodeName}`);
  const run = new Function(
    "$json",
    "$getWorkflowStaticData",
    node.parameters.jsCode,
  );
  return run(json, () => store)[0].json;
}

let health = execute("00 - Health - State", {});
assert.equal(health.ready, false);

execute("06 - Worker Heartbeat - State", {
  body: {
    worker_id: "test-worker",
    ready: true,
    r2_configured: true,
    max_images_per_job: 50,
  },
});
health = execute("00 - Health - State", {});
assert.equal(health.ready, true);

const request = execute("01 - Init Upload - State", {
  body: {
    files: [
      { filename: "tray.jpg", content_type: "image/jpeg", size_bytes: 123 },
    ],
  },
});
assert.match(request.job_id, /^[0-9a-f]{32}$/);
assert.equal(request.status, "WAITING_FOR_WORKER");

const presignTask = execute("04 - Worker Next - State", {
  query: { worker_id: "test-worker" },
}).task;
assert.equal(presignTask.task_type, "PRESIGN");

const uploads = [
  {
    filename: "tray.jpg",
    content_type: "image/jpeg",
    size_bytes: 123,
    object_key: `purchase-intake/${request.job_id}/incoming/001_tray.jpg`,
    upload_url: "https://upload.example/signed",
    method: "PUT",
    headers: { "Content-Type": "image/jpeg" },
  },
];
execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: presignTask.task_id,
    ok: true,
    final: true,
    result: { job_id: request.job_id, uploads, expires_in: 900 },
  },
});
const ready = execute("01b - Upload Status - State", {
  query: { request_id: request.job_id },
});
assert.equal(ready.status, "READY");
assert.deepEqual(ready.uploads, uploads);

const accepted = execute("02 - Submit Job - State", {
  body: {
    job_id: request.job_id,
    files: [{ object_key: uploads[0].object_key }],
  },
});
assert.equal(accepted.status, "QUEUED");

const processTask = execute("04 - Worker Next - State", {
  query: { worker_id: "test-worker" },
}).task;
assert.equal(processTask.task_type, "PROCESS");
execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: processTask.task_id,
    ok: true,
    final: false,
    result: {
      job_id: request.job_id,
      status: "PROCESSING",
      total_images: 1,
      processed_images: 1,
    },
  },
});
const processing = execute("03 - Job Status - State", {
  query: { job_id: request.job_id },
});
assert.equal(processing.status, "PROCESSING");
assert.equal(processing.processed_images, 1);
execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: processTask.task_id,
    ok: true,
    final: true,
    result: {
      job_id: request.job_id,
      status: "COMPLETED",
      total_images: 1,
      processed_images: 1,
      products: [],
      images: [],
      r2_objects: [],
    },
  },
});
const completed = execute("03 - Job Status - State", {
  query: { job_id: request.job_id },
});
assert.equal(completed.status, "COMPLETED");
assert.equal(store.queue.length, 0);

const duplicate = execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: processTask.task_id,
    ok: false,
    final: true,
    error: "late retry",
  },
});
assert.equal(duplicate.duplicate, true);
assert.equal(
  execute("03 - Job Status - State", {
    query: { job_id: request.job_id },
  }).status,
  "COMPLETED",
);

const largeRequest = execute("01 - Init Upload - State", {
  body: {
    files: [
      { filename: "tray-1.jpg", content_type: "image/jpeg", size_bytes: 50 * 1024 * 1024 },
      { filename: "tray-2.jpg", content_type: "image/jpeg", size_bytes: 50 * 1024 * 1024 },
      { filename: "tray-3.jpg", content_type: "image/jpeg", size_bytes: 50 * 1024 * 1024 },
    ],
  },
});
assert.equal(largeRequest.status, "WAITING_FOR_WORKER");

const oversizedRequest = execute("01 - Init Upload - State", {
  body: {
    files: [
      { filename: "tray-1.jpg", content_type: "image/jpeg", size_bytes: 50 * 1024 * 1024 },
      { filename: "tray-2.jpg", content_type: "image/jpeg", size_bytes: 50 * 1024 * 1024 },
      { filename: "tray-3.jpg", content_type: "image/jpeg", size_bytes: 50 * 1024 * 1024 },
      { filename: "tray-4.jpg", content_type: "image/jpeg", size_bytes: 11 * 1024 * 1024 },
    ],
  },
});
assert.equal(oversizedRequest.status, "ERROR");

console.log("Workflow 4 outbound queue simulation passed.");
