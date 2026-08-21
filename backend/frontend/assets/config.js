// Runtime routing for the same frontend bundle.
// Local FastAPI uses same-origin /api routes. Static/public hosts use the n8n
// outbound-worker webhooks and ask the operator for Basic Auth credentials.
(() => {
  const hostname = window.location.hostname.toLowerCase();
  const isLocal = hostname === "127.0.0.1" || hostname === "localhost";

  if (!isLocal) {
    window.SHARON_REMOTE_MODE = true;
    window.SHARON_WEBHOOK_BASE = "https://n8n.sharon-finefoods.com/webhook";
    document.documentElement.classList.add("remote-mode");
  }
})();
