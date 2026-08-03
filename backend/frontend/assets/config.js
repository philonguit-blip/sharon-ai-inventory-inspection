// Localhost keeps using the same-origin FastAPI gateway. A deployed static
// build talks to the authenticated n8n webhooks directly; image bytes still
// upload straight to R2 and never pass through n8n.
const localHosts = new Set(["127.0.0.1", "localhost", "::1"]);
const forceRemote = new URLSearchParams(window.location.search).get("remote") === "1";

window.SHARON_REMOTE_MODE =
  window.SHARON_REMOTE_MODE ?? (forceRemote || !localHosts.has(window.location.hostname));
window.SHARON_ORCHESTRATOR_BASE =
  window.SHARON_ORCHESTRATOR_BASE || "/api/v1/orchestrator";
window.SHARON_WEBHOOK_BASE =
  window.SHARON_WEBHOOK_BASE || "https://n8n.sharon-finefoods.com/webhook";

if (window.SHARON_REMOTE_MODE) {
  document.documentElement.classList.add("remote-mode");
}
