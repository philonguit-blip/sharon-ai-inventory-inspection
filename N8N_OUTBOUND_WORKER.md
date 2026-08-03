# n8n outbound worker deployment

## Current architecture

The production n8n workflow is **Sharon Bakery - Outbound AI Worker Queue v1**.
It is published at `https://n8n.sharon-finefoods.com` and uses authenticated
webhooks as a small command queue.

The public operator interface is deployed at:

```text
https://sharon-bakery-inventory.pages.dev/
```

The AI computer does not expose FastAPI to the internet:

1. The local web app asks n8n to create an upload session.
2. n8n queues a `PRESIGN` command.
3. `app.queue_worker` polls n8n over outbound HTTPS and asks local FastAPI to
   create short-lived R2 upload URLs.
4. The browser uploads image bytes directly to R2.
5. n8n queues a `PROCESS` command.
6. The worker downloads the approved R2 objects, runs the local AI pipeline,
   uploads artifacts to R2, and reports the final state to n8n.

No Cloudflare Tunnel or inbound port-forward is required for this flow.

## Start the system

Open one terminal for FastAPI:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Open a second terminal for the outbound worker:

```powershell
.\start_worker.bat
```

Then open either the local web interface:

```text
http://127.0.0.1:8080
```

or the public interface from another computer or phone:

```text
https://sharon-bakery-inventory.pages.dev/
```

The public interface uses the same `APP_AUTH_USERNAME` and
`APP_AUTH_PASSWORD` configured in `backend/.env`. The password is sent only
over HTTPS to the authenticated n8n webhooks and is kept in browser memory for
the current tab; it is not embedded in the deployed static files or written to
browser storage.

Do not browse to `http://0.0.0.0:8080`; `0.0.0.0` is a bind address, not a
browser destination.

## Health checks

The local UI calls `/api/v1/orchestrator/health`. A ready response requires:

- the n8n workflow to be published;
- the outbound worker heartbeat to be newer than 30 seconds;
- the local bakery model to be ready;
- R2 to be configured.

If the worker is offline, the local UI can still fall back to direct local
processing. The n8n webhook route will intentionally report `ready: false`.

## Workflow source and verification

- Import source: `n8n/Workflow 4_ Sharon Bakery Outbound Worker.json`
- Generator: `n8n/generate_workflow4_outbound.mjs`
- Queue simulation: `node n8n/test_workflow4_outbound.mjs`
- Worker tests: `backend/tests/test_queue_worker.py`

After importing into another n8n instance, select the existing Basic Auth
credential named **Sharon Bakery Backend API** for every Webhook trigger before
publishing. The JSON intentionally contains no credential secret.

## Operational notes

- The n8n static-data queue is suitable for the current single-worker,
  low-volume bakery workflow. Move the queue to Postgres/Redis before using
  multiple concurrent workers or high-volume traffic.
- Presign and process commands are idempotent. Retrying a lost response reuses
  the same job and does not schedule a second local receipt.
- The remote frontend waits before its first status request and treats a brief
  `Job not found` response as an n8n persistence delay. It retries status and,
  when needed, safely resubmits the same idempotent job instead of showing an
  immediate failure on mobile networks.
- With `KIOTVIET_AUTO_CREATE_DRAFT=true`, a successfully completed real image
  job can create a KiotViet draft automatically. Smoke tests must stop before
  submitting a processing job unless that behavior is intended.
- n8n webhooks are the public API gateway. Cloudflare Pages hosts only the
  static graphical interface, so no application secret is included in the
  public deployment.
- Remote users can open the Pages URL at any time, but processing is ready only
  while both local FastAPI and `start_worker.bat` are running on the AI
  computer and that computer has internet access.
- The worker refreshes its heartbeat while a multi-image AI job is running, so
  remote clients do not see a false `Outbound AI worker is offline` message
  during longer batches. If FastAPI restarts, unfinished local jobs are marked
  `ERROR` because background processing cannot survive the restart.
- A job accepts up to 50 images, 50 MB per image and 160 MB total. The browser
  uploads up to four files concurrently to R2; the backend downloads and later
  uploads R2 artifacts concurrently while keeping model inference sequential.
- During processing the worker reports every `processed_images` change to n8n,
  allowing the remote page to display `1/N`, `2/N`, and so on before the final
  result is ready.
- After changing files under `backend/frontend`, redeploy with:

  ```powershell
  npx --yes wrangler@4.30.0 pages deploy backend\frontend --project-name sharon-bakery-inventory --branch main
  ```
