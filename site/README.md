# macMLX static site maintenance

The site build is network-free. It validates the checked-in registries and assets, then replaces the ignored `public/` directory with deterministic output. Do not add network fetches to the build or crawler.

## Build and verify

The reviewed 1200×630 social cards are tracked build inputs. Their SVG source is registry-driven and network-free. Refresh them only in an environment where Sharp is already available, either through normal Node resolution or an explicit `MACMLX_NODE_MODULES` directory; the normal build and CI do not install or require Sharp.

Run the complete local sequence from the repository root:

```sh
MACMLX_NODE_MODULES=/path/to/node_modules node scripts/render-social-cards.mjs
node scripts/validate-social-cards.mjs
node scripts/build-public-site.mjs
node --test site/tests/*.test.mjs
node scripts/crawl-public-site.mjs
node scripts/test-public-site.mjs
node --check scripts/build-public-site.mjs
node --check scripts/crawl-public-site.mjs
node --check scripts/render-social-cards.mjs
node --check scripts/validate-social-cards.mjs
node --check scripts/verify-cloudflare-deploy.mjs
node --check public/assets/js/main.js
xmllint --noout public/sitemap.xml public/assets/og-image.svg
git diff --check
```

These commands do not verify external URLs over the network. The crawler validates their syntax and checks all local HTML, Markdown, text, XML, links, assets, canonicals, locale counterparts, social images, and generated-file hygiene.

## Cloudflare deployment runbook

The site uses Workers Static Assets without a Worker script or storage binding. Production attaches only the `macmlx.app` apex custom domain; staging stays on `workers.dev`. Wrangler must be installed for Bun in a separate package directory. Run every `bun wrangler` command from that directory while pointing `--config` at the site checkout.

Set these required environment variables for the current machine. The values below are generic examples, not repository-specific defaults:

```sh
export SITE_ROOT=/absolute/path/to/macmlx-site-checkout
export WRANGLER_PACKAGE_DIR=/absolute/path/to/directory-containing-wrangler-package
```

Validate and publish staging first:

```sh
cd "$WRANGLER_PACKAGE_DIR"
bun wrangler deploy --dry-run --config "$SITE_ROOT/wrangler.jsonc" --env staging
bun wrangler deploy --config "$SITE_ROOT/wrangler.jsonc" --env staging
node "$SITE_ROOT/scripts/verify-cloudflare-deploy.mjs" "https://macmlx-site-staging.<your-workers-subdomain>.workers.dev/"
```

Replace `<your-workers-subdomain>` with the hostname printed by the staging deployment. The staging and version-preview host pattern receives `X-Robots-Tag: noindex, nofollow` from `_headers`.

### Mandatory production preflight

The root `custom_domain` deployment mutates apex routing. Cloudflare can provision or use Cloudflare DNS and a certificate for `macmlx.app` and attach the Worker to the exact hostname. That attachment captures all paths on that exact host. This can conflict with an existing CNAME, Worker route or custom domain, Pages custom domain, or origin deployment.

Before any production command, inventory the existing `macmlx.app` DNS records, proxied origin, Worker routes and custom domains, Pages projects and custom domains, and certificate state. Record an explicitly approved migration and rollback owner for every existing apex dependency. Do not deploy the root configuration until that preflight inventory and approved migration are complete.

Wrangler does not manage every zone control used by this site. WAF policy, zone TLS mode, and HSTS remain separate and out of scope for this deployment runbook.

After staging verification and production preflight succeed, list versions and record the previous known-good version ID in the change record. Then run a fresh root-config dry run immediately before production deployment:

```sh
cd "$WRANGLER_PACKAGE_DIR"
bun wrangler versions list --config "$SITE_ROOT/wrangler.jsonc"
PREVIOUS_VERSION_ID="<recorded-known-good-version-id>"
bun wrangler deploy --dry-run --config "$SITE_ROOT/wrangler.jsonc"
bun wrangler deploy --config "$SITE_ROOT/wrangler.jsonc"
node "$SITE_ROOT/scripts/verify-cloudflare-deploy.mjs" https://macmlx.app/
```

If verification fails, roll back to the recorded version and immediately verify the apex again:

```sh
cd "$WRANGLER_PACKAGE_DIR"
bun wrangler rollback "$PREVIOUS_VERSION_ID" --config "$SITE_ROOT/wrangler.jsonc"
node "$SITE_ROOT/scripts/verify-cloudflare-deploy.mjs" https://macmlx.app/
```

The `_redirects` policy canonicalizes only `/index.html` and `/zh/index.html`. Query-string language migration remains in the existing client compatibility code because Static Assets redirects do not match query parameters. CSS and JavaScript use short browser caching, images use at most seven days, and stable filenames are not assigned year-long caching. CSP is intentionally deferred until it is browser-tested against the theme script and JSON-LD on both locales.

## Release refresh

1. Update the project and release registries with the released version, date, download, and official release sources.
2. Reclassify facts as released, development, or planned. Keep limitations explicit and preserve English/Chinese parity.
3. Reverify only affected competitors against official sources. Do not refresh unrelated snapshots.
4. Update `lastVerified` only after those checks have actually been completed.
5. Refresh and review the two social cards when their registry-driven copy changes, then rebuild, run the full tests and crawler, and complete browser QA for both locales, responsive layouts, and structured data.

Search Console, Bing Webmaster Tools, and IndexNow submissions are separately authorized post-deployment operations. The local build does not perform them.
