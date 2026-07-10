---
name: cloudflare-static-site-release
description: Use when publishing or updating a static website on Cloudflare with Bun-managed Wrangler, especially when the release includes a custom domain, staging verification, cache/security headers, or rollback requirements.
---

# Cloudflare Static Site Release

Release static sites through Workers Static Assets with a staged, verifiable, and reversible workflow. Use the installed Cloudflare and Wrangler skills for current platform syntax.

## Choose the delivery surface

- Prefer Workers Static Assets for a new static deployment. Do not add a Worker script when assets-only routing is sufficient.
- Use R2 only when object storage materially helps, such as independently managed large media or application uploads. Do not create a second asset origin merely for ordinary site files.
- Preserve an existing Pages deployment unless migration is explicitly in scope.

## Release workflow

1. Discover the repository root, generated output directory, Wrangler package directory, config, build, crawler, and existing verifier. Never assume the current directory is the deployable site.
2. Build from tracked source and run tests, crawler, syntax checks, and deterministic-output checks. Refuse to deploy stale generated output.
3. Configure explicit production and staging environments. Keep staging on `workers.dev` with `X-Robots-Tag: noindex, nofollow`.
4. Use bounded caching: documents and discovery files revalidate; stable CSS/JS receive short caching; ordinary images receive measured caching. Never add `immutable` to stable filenames.
5. Run every Wrangler command through `bun wrangler` with an explicit `--config`. Set `WRANGLER_LOG_PATH` to a writable temporary path when the environment restricts the default log directory.
6. Run staging and production dry-runs. Deploy staging, then verify status, canonical URLs, same-origin redirects, 404 behavior, security headers, MIME types, cache policy, discovery files, social images, and primary assets.
7. Before a custom-domain production deploy, inventory DNS, certificates, proxied origins, Worker routes/custom domains, and Pages custom domains. Record the approved migration, previous known-good version ID, and rollback owner.
8. Deploy production only after staging and preflight pass. Verify the public domain immediately, list the new version, and perform browser QA for locale, theme, and responsive behavior.
9. On verification failure, roll back to the recorded version and run the same verifier again.

## Safety rules

- Allow online verification only against the exact production origin or an expected staging hostname; reject arbitrary hosts, credentials, ports, paths, queries, and fragments.
- Require same-origin redirect destinations and bounded response reads with timeouts.
- Treat custom-domain attachment as a DNS/routing/certificate mutation. Do not describe it as an assets-only change.
- Keep WAF, zone TLS mode, HSTS, Search Console, and indexing submissions out of scope unless explicitly requested.

Use [references/release-checklist.md](references/release-checklist.md) as the final evidence checklist.
