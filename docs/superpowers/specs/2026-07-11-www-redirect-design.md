# www.macmlx.app Permanent Redirect Design

## Goal

Make every HTTPS request to `www.macmlx.app` permanently redirect to the equivalent `https://macmlx.app` URL. Preserve the request path and query string. Keep the existing apex Workers Static Assets deployment unchanged.

## Architecture

Create a separate, script-only Cloudflare Worker named `macmlx-www-redirect`. Its only production trigger is the `www.macmlx.app` Custom Domain. The Worker constructs a destination URL from the incoming request, replaces the origin with `https://macmlx.app`, and returns a permanent redirect.

The existing `macmlx-site` Worker, its assets, cache policy, and `macmlx.app` Custom Domain remain untouched. R2, Pages, KV, and other bindings are not introduced.

## Redirect behavior

- `https://www.macmlx.app/` redirects to `https://macmlx.app/`.
- Paths are preserved: `/zh/` remains `/zh/`.
- Query strings are preserved exactly.
- URL fragments are not available to servers and therefore cannot be preserved by any HTTP redirect.
- The destination scheme is always HTTPS and the destination host is always `macmlx.app`.
- The response uses a permanent redirect status and includes no cache or application content.

## DNS and deployment

The current DNS-only `www.macmlx.app` CNAME conflicts with a Workers Custom Domain. Delete only that exact CNAME after confirming its name, type, and target in Cloudflare. Deploy the redirect Worker with Bun-managed Wrangler and an explicit config path. Wrangler then creates and manages the `www.macmlx.app` Custom Domain and certificate.

Deployment order:

1. Run unit tests and Wrangler dry-run.
2. Record the current CNAME as the DNS rollback value.
3. Delete the exact conflicting CNAME.
4. Deploy `macmlx-www-redirect` with Wrangler.
5. Verify DNS, TLS, redirect status, destination, path, and query preservation.

If deployment fails after DNS deletion, restore `www.macmlx.app CNAME macmlx.app` as DNS-only. If a later Worker release fails, roll back to its last known-good Worker version.

## Tests

Unit tests call the Worker directly and verify root, nested path, Unicode-encoded path, and query preservation. They also verify that spoofed incoming hosts cannot influence the apex destination. Configuration tests assert the exact Worker name, entrypoint, compatibility date, disabled `workers.dev` exposure, and single `www.macmlx.app` Custom Domain.

Online verification uses manual redirect handling so it can assert the exact permanent status and `Location` value without following the response. It also confirms the apex production verifier still passes after deployment.

## Non-goals

- No changes to apex site content or visual design.
- No redirect from arbitrary subdomains.
- No Cloudflare Pages, R2, Bulk Redirect, or dashboard-only rule.
- No modification of the existing apex Worker configuration.
