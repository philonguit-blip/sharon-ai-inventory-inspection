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

// Embedded UI JavaScript must run after the body has been parsed. The original
// external script used `defer`; converting it to an inline head script caused
// every DOM lookup to return null on the public n8n page.
const webUi = execute("00a - Web UI - State", {}).html;
assert.ok(webUi.includes('onsubmit="return false;"'));
assert.equal(webUi.includes('src="/assets/app.js'), false);
assert.ok(webUi.indexOf("const fallbackStorage") > webUi.indexOf("</main>"));
assert.ok(webUi.indexOf("const fallbackStorage") < webUi.indexOf("</body>"));
assert.ok(webUi.includes('<option value="AUTO" selected>'));
assert.ok(webUi.includes('inferenceMode: "AUTO"'));
assert.ok(webUi.includes('id="developer-settings"'));
assert.ok(webUi.includes('bakery-developer-settings-status'));

// Worker health.
let health = execute("00 - Health - State", {});
assert.equal(health.ready, false);

execute("06 - Worker Heartbeat - State", {
  body: {
    worker_id: "test-worker",
    ready: true,
    r2_configured: true,
    kiotviet_configured: true,
    kiotviet_auto_create_draft: false,
    max_images_per_job: 50,
    max_job_upload_size_mb: 200,
  },
});

health = execute("00 - Health - State", {});
assert.equal(health.ready, true);
assert.equal(health.kiotviet_auto_create_draft, false);

store.tasks["stale-task"] = {
  task_id: "stale-task",
  job_id: "stale-job",
  task_type: "PROCESS",
  status: "LEASED",
  leased_by: "retired-worker",
  leased_at: Date.now(),
};
store.queue.push("stale-task");
store.worker.worker_id = "retired-worker";
execute("06 - Worker Heartbeat - State", {
  body: {
    worker_id: "test-worker",
    ready: true,
    r2_configured: true,
    max_images_per_job: 50,
  },
});
assert.equal(store.tasks["stale-task"].status, "QUEUED");
delete store.tasks["stale-task"];
store.queue = store.queue.filter((id) => id !== "stale-task");

const rejectedTooMany = execute("01 - Init Upload - State", {
  body: {
    files: Array.from({ length: 51 }, (_, index) => ({
      filename: `tray-${index + 1}.jpg`, content_type: "image/jpeg", size_bytes: 123,
    })),
  },
});
assert.equal(rejectedTooMany.status, "ERROR");

// One job may contain many images, but they must represent the same SKU.
const request = execute("01 - Init Upload - State", {
  body: {
    inference_mode: "COMPARE",
    files: [
      { filename: "tray-1.jpg", content_type: "image/jpeg", size_bytes: 123 },
      { filename: "tray-2.jpg", content_type: "image/jpeg", size_bytes: 456 },
    ],
  },
});
assert.match(request.job_id, /^[0-9a-f]{32}$/);
assert.equal(request.status, "WAITING_FOR_WORKER");

const presignTask = execute("04 - Worker Next - State", {
  query: { worker_id: "test-worker" },
}).task;
assert.equal(presignTask.task_type, "PRESIGN");
assert.equal(presignTask.payload.inference_mode, "COMPARE");

const uploads = [
  {
    filename: "tray-1.jpg",
    content_type: "image/jpeg",
    size_bytes: 123,
    object_key: `purchase-intake/${request.job_id}/incoming/001_tray-1.jpg`,
    upload_url: "https://upload.example/signed-1",
    method: "PUT",
    headers: { "Content-Type": "image/jpeg" },
  },
  {
    filename: "tray-2.jpg",
    content_type: "image/jpeg",
    size_bytes: 456,
    object_key: `purchase-intake/${request.job_id}/incoming/002_tray-2.jpg`,
    upload_url: "https://upload.example/signed-2",
    method: "PUT",
    headers: { "Content-Type": "image/jpeg" },
  },
];

// A restarted/previous worker may leave a stale lease. The worker currently
// registered by heartbeat is allowed to recover it instead of looping forever.
store.tasks[presignTask.task_id].leased_by = "stale-worker";

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

// Submit and AI process.
const accepted = execute("02 - Submit Job - State", {
  body: {
    job_id: request.job_id,
    inference_mode: "COMPARE",
    files: uploads.map((item) => ({ object_key: item.object_key })),
  },
});
assert.equal(accepted.status, "QUEUED");
assert.equal(accepted.total_images, 2);

const processTask = execute("04 - Worker Next - State", {
  query: { worker_id: "test-worker" },
}).task;
assert.equal(processTask.task_type, "PROCESS");
assert.equal(processTask.payload.inference_mode, "COMPARE");

execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: processTask.task_id,
    ok: true,
    final: false,
    result: {
      job_id: request.job_id,
      status: "PROCESSING",
      total_images: 2,
      processed_images: 1,
    },
  },
});

let processing = execute("03 - Job Status - State", {
  query: { job_id: request.job_id },
});
assert.equal(processing.status, "PROCESSING");
assert.equal(processing.processed_images, 1);

// Backend now stops at AWAITING_CONFIRMATION instead of auto-creating KiotViet.
execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: processTask.task_id,
    ok: true,
    final: true,
    result: {
      job_id: request.job_id,
      status: "AWAITING_CONFIRMATION",
      total_images: 2,
      processed_images: 2,
      total_quantity: 10,
      product_count: 1,
      decision: {
        decision: "DIRECT",
        dominant_class: "PA-CRO-0000054_MiniCroissant_Baked",
        display_name: "Mini Croissant (Baked)",
        product_code: "PA-CRO-0000054",
        product_name: "Mini Croissant (Baked)",
        count: 10,
        purity: 0.98,
        avg_confidence: 0.42,
        requires_confirmation: true,
        requires_user_selection: false,
        per_image: [
          { image_name: "tray-1.jpg", count: 4, purity: 0.98, avg_confidence: 0.42 },
          { image_name: "tray-2.jpg", count: 6, purity: 0.98, avg_confidence: 0.42 },
        ],
      },
      products: [],
      images: [],
      r2_objects: [],
    },
  },
});

let awaiting = execute("03 - Job Status - State", {
  query: { job_id: request.job_id },
});
assert.equal(awaiting.status, "AWAITING_CONFIRMATION");
assert.equal(awaiting.decision.decision, "DIRECT");
assert.equal(store.queue.length, 0);

// Explicit confirmation creates a CONFIRM task.
const confirming = execute("03b - Confirm Job - State", {
  body: {
    job_id: request.job_id,
    confirm: true,
  },
});
assert.equal(confirming.status, "CONFIRMING");

const confirmTask = execute("04 - Worker Next - State", {
  query: { worker_id: "test-worker" },
}).task;
assert.equal(confirmTask.task_type, "CONFIRM");
assert.deepEqual(confirmTask.payload, {
  confirm: true,
  document_type: "PURCHASE_RECEIPT",
});

// Worker executes local /confirm and returns the final KiotViet result.
execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: confirmTask.task_id,
    ok: true,
    final: true,
    result: {
      job_id: request.job_id,
      status: "COMPLETED",
      total_images: 2,
      processed_images: 2,
      total_quantity: 10,
      product_count: 1,
      confirmed_product: {
        product_code: "PA-CRO-0000054",
        product_name: "Mini Croissant (Baked)",
        product_id: 12345,
        quantity: 10,
      },
      products: [
        {
          product_code: "PA-CRO-0000054",
          product_name: "Mini Croissant (Baked)",
          product_id: 12345,
          quantity: 10,
          purchase_price: 0,
        },
      ],
      kiotviet: {
        created: true,
        resolved_product_id: 12345,
        receipt: { code: "PN000001" },
      },
      images: [],
      r2_objects: [],
      excel_url: "https://download.example/report.xlsx",
    },
  },
});

const completed = execute("03 - Job Status - State", {
  query: { job_id: request.job_id },
});
assert.equal(completed.status, "COMPLETED");
assert.equal(completed.kiotviet.created, true);
assert.equal(store.queue.length, 0);

// Operator can explicitly route a confirmation to manufacturing RPA.
const manufacturingJobId = "e".repeat(32);
store.jobs[manufacturingJobId] = {
  job_id: manufacturingJobId,
  status: "AWAITING_CONFIRMATION",
  decision: {
    decision: "DIRECT",
    product_code: "PA-CRO-0000054",
    product_name: "Mini Croissant (Baked)",
    count: 6,
  },
};
const manufacturingConfirm = execute("03b - Confirm Job - State", {
  body: {
    job_id: manufacturingJobId,
    confirm: true,
    document_type: "MANUFACTURING",
  },
});
assert.equal(manufacturingConfirm.document_type, "MANUFACTURING");
const manufacturingTask = execute("04 - Worker Next - State", {
  query: { worker_id: "test-worker" },
}).task;
assert.deepEqual(manufacturingTask.payload, {
  confirm: true,
  document_type: "MANUFACTURING",
});
execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: manufacturingTask.task_id,
    ok: true,
    final: true,
    result: {
      job_id: manufacturingJobId,
      status: "COMPLETED",
      document_type: "MANUFACTURING",
      kiotviet: { created: true, document_type: "MANUFACTURING" },
    },
  },
});

// Late retry remains idempotent.
const duplicate = execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: confirmTask.task_id,
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

// Family confirmation requires one valid family member.
const familyJobId = "f".repeat(32);
store.jobs[familyJobId] = {
  job_id: familyJobId,
  status: "AWAITING_CONFIRMATION",
  decision: {
    decision: "FAMILY",
    dominant_class: "BR-SD-COMMON-RUSTIC_SourdoughRusticLoaf",
    display_name: "Rustic Sourdough Loaf",
    count: 8,
    members: [
      { product_code: "BR-SD-0000135", display_name: "Wholemeal Walnuts" },
      { product_code: "BR-SD-0000131", display_name: "Sharon White Bread" },
      { product_code: "BR-SD-0000167", display_name: "Dark Rye" },
    ],
  },
};

const invalidFamily = execute("03b - Confirm Job - State", {
  body: { job_id: familyJobId, confirm: true, product_code: "INVALID" },
});
assert.equal(invalidFamily.status, "ERROR");

const validFamily = execute("03b - Confirm Job - State", {
  body: { job_id: familyJobId, confirm: true, product_code: "BR-SD-0000167" },
});
assert.equal(validFamily.status, "CONFIRMING");

const familyConfirmTask = execute("04 - Worker Next - State", {
  query: { worker_id: "test-worker" },
}).task;
assert.equal(familyConfirmTask.task_type, "CONFIRM");
assert.deepEqual(familyConfirmTask.payload, {
  confirm: true,
  document_type: "PURCHASE_RECEIPT",
  product_code: "BR-SD-0000167",
});

// A failed confirmation keeps the job in reconciliation mode. Retrying cannot
// issue an unguarded second KiotViet POST.
execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: familyConfirmTask.task_id,
    ok: false,
    final: true,
    error: "KiotViet temporarily unavailable",
  },
});
const familyRetry = execute("03 - Job Status - State", {
  query: { job_id: familyJobId },
});
assert.equal(familyRetry.status, "CONFIRMING");
assert.match(familyRetry.confirmation_error, /temporarily unavailable/);

// Retrying queues the same stable task ID so the backend can reconcile by
// job_id. It does not create a second independent confirmation task.
const reconciliation = execute("03b - Confirm Job - State", {
  body: { job_id: familyJobId, confirm: true, product_code: "BR-SD-0000167" },
});
assert.equal(reconciliation.status, "CONFIRMING");
assert.equal(reconciliation.confirmation_error, "");
const reconciliationTask = execute("04 - Worker Next - State", {
  query: { worker_id: "test-worker" },
}).task;
assert.equal(reconciliationTask.task_id, familyConfirmTask.task_id);
assert.equal(reconciliationTask.task_type, "CONFIRM");
assert.deepEqual(reconciliationTask.payload, {
  confirm: true,
  document_type: "PURCHASE_RECEIPT",
  product_code: "BR-SD-0000167",
  quantity: 8,
});

// Per-image size limits remain enforced.
const oversizedRequest = execute("01 - Init Upload - State", {
  body: {
    files: [
      { filename: "tray-1.jpg", content_type: "image/jpeg", size_bytes: 50 * 1024 * 1024 + 1 },
    ],
  },
});
assert.equal(oversizedRequest.status, "ERROR");

// Hidden developer settings travel through the same outbound worker queue.
const developerRequest = execute("07 - Developer Settings - State", {
  body: { action: "GET", developer_key: "test-secret" },
});
assert.match(developerRequest.request_id, /^[0-9a-f]{32}$/);
const developerTask = execute("04 - Worker Next - State", {
  query: { worker_id: "test-worker" },
}).task;
assert.equal(developerTask.task_type, "DEVELOPER_SETTINGS");
assert.equal(developerTask.payload.developer_key, "test-secret");
execute("05 - Worker Result - State", {
  body: {
    worker_id: "test-worker",
    task_id: developerTask.task_id,
    ok: true,
    final: true,
    result: {
      ready: true,
      active_model: "best_next.pt",
      available_models: ["best_next.pt"],
      classes: [],
    },
  },
});
const developerResult = execute("07b - Developer Settings Status - State", {
  query: { request_id: developerRequest.request_id },
});
assert.equal(developerResult.status, "COMPLETED");
assert.equal(developerResult.active_model, "best_next.pt");
assert.equal(store.tasks[developerTask.task_id].payload.developer_key, undefined);

console.log("Workflow 4 outbound queue + confirmation simulation passed.");
