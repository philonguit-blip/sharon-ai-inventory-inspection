# Security policy

## Secrets

Never commit `backend/.env`, Cloudflare R2 keys, KiotViet credentials, Supabase
keys, n8n credentials, access tokens, or signed upload URLs. Use
`backend/.env.example` as the configuration template and keep production values
only in the local `.env` file and the relevant provider secret stores.

The hidden developer panel is protected by `DEVELOPER_SETTINGS_KEY`. Configure
it separately from the operator password for production use. The key is kept in
browser memory only; n8n removes it from a completed relay task immediately and
the local runtime settings file never stores it.

If a credential is committed or shared accidentally:

1. Revoke or rotate it at the provider immediately.
2. Update the local `.env` and the corresponding n8n credential.
3. Restart the backend and outbound worker.
4. Verify `/api/v1/bakery/health` and the public n8n health webhook.

## Reporting

This is a private internal system. Report suspected vulnerabilities or leaked
credentials directly to the Sharon Fine Foods AI/IT administrator. Do not open
a public issue containing sensitive details.
