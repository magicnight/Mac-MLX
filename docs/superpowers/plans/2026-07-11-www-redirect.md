# www.macmlx.app Permanent Redirect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permanently redirect every `www.macmlx.app` HTTPS request to the equivalent `https://macmlx.app` URL without changing the apex static site Worker.

**Architecture:** Add a separate script-only Worker named `macmlx-www-redirect`, with one Custom Domain trigger for `www.macmlx.app`. A dedicated verifier checks the exact status and `Location` header without following redirects; the existing apex verifier remains unchanged.

**Tech Stack:** JavaScript modules, Node test runner, Bun-managed Wrangler 4, Cloudflare Workers Custom Domains.

---

## File structure

- Create `site/cloudflare/www-redirect.mjs`: pure destination builder and Worker fetch handler.
- Create `wrangler.www.jsonc`: isolated production configuration for the redirect Worker.
- Create `site/tests/www-redirect.test.mjs`: Worker behavior and configuration contract tests.
- Create `scripts/verify-www-redirect.mjs`: bounded online redirect verifier.
- Modify `site/README.md`: deployment, verification, version, and DNS rollback procedure.
- Modify `.github/workflows/ci.yml`: syntax-check the Worker and verifier.

### Task 1: Lock redirect behavior with tests

**Files:**
- Create: `site/tests/www-redirect.test.mjs`
- Create: `site/cloudflare/www-redirect.mjs`

- [ ] **Step 1: Write the failing Worker tests**

Create `site/tests/www-redirect.test.mjs` with imports for `node:test` and `node:assert/strict`, then import `redirectDestination` and the default Worker from `../cloudflare/www-redirect.mjs`. Add assertions equivalent to:

```js
test("www requests permanently redirect to the equivalent apex URL", async () => {
  const response = await worker.fetch(new Request("https://www.macmlx.app/zh/?q=mlx%20swift&lang=zh"));
  assert.equal(response.status, 308);
  assert.equal(response.headers.get("location"), "https://macmlx.app/zh/?q=mlx%20swift&lang=zh");
  assert.equal(await response.text(), "");
});

test("the incoming host can never change the apex destination", () => {
  assert.equal(
    redirectDestination("https://attacker.example/models/?next=https%3A%2F%2Fevil.example"),
    "https://macmlx.app/models/?next=https%3A%2F%2Fevil.example",
  );
});
```

Also cover `/`, an encoded Unicode path, and a POST request receiving 308.

- [ ] **Step 2: Run the test and verify RED**

Run:

```sh
node --test site/tests/www-redirect.test.mjs
```

Expected: FAIL because `site/cloudflare/www-redirect.mjs` does not exist.

- [ ] **Step 3: Implement the minimal Worker**

Create `site/cloudflare/www-redirect.mjs`:

```js
const apexOrigin = "https://macmlx.app";

export function redirectDestination(requestURL) {
  const source = new URL(requestURL);
  return new URL(`${source.pathname}${source.search}`, apexOrigin).toString();
}

export default {
  fetch(request) {
    return new Response(null, {
      status: 308,
      headers: {
        Location: redirectDestination(request.url),
        "Cache-Control": "public, max-age=3600",
      },
    });
  },
};
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run `node --test site/tests/www-redirect.test.mjs`.

Expected: all Worker behavior tests PASS.

- [ ] **Step 5: Commit the behavior**

Stage only the Worker and its test, then commit with a Lore-formatted message beginning `feat(site): make the www entry permanently canonical`.

### Task 2: Lock deployment configuration and verification

**Files:**
- Modify: `site/tests/www-redirect.test.mjs`
- Create: `wrangler.www.jsonc`
- Create: `scripts/verify-www-redirect.mjs`

- [ ] **Step 1: Add failing configuration tests**

Read and parse `wrangler.www.jsonc` in the existing test. Assert this exact contract:

```js
assert.equal(config.name, "macmlx-www-redirect");
assert.equal(config.main, "site/cloudflare/www-redirect.mjs");
assert.equal(config.compatibility_date, "2026-07-11");
assert.equal(config.workers_dev, false);
assert.equal(config.preview_urls, false);
assert.deepEqual(config.routes, [{ pattern: "www.macmlx.app", custom_domain: true }]);
assert.equal(config.assets, undefined);
```

- [ ] **Step 2: Run the test and verify RED**

Run `node --test site/tests/www-redirect.test.mjs`.

Expected: FAIL because `wrangler.www.jsonc` does not exist.

- [ ] **Step 3: Add the minimal Wrangler configuration**

Create `wrangler.www.jsonc` with exactly the asserted fields. Do not add assets, bindings, environments, or a `workers.dev` route.

- [ ] **Step 4: Add verifier tests before implementation**

Extend the test to import `verifyWWWRedirect` from `../../scripts/verify-www-redirect.mjs`. Inject a fetch implementation that returns manual 308 responses and assert that root, nested path, and encoded-query checks pass. Add negative cases for status 200, a cross-origin `Location`, a changed path, and a dropped query.

- [ ] **Step 5: Run the test and verify verifier RED**

Run `node --test site/tests/www-redirect.test.mjs`.

Expected: FAIL because `scripts/verify-www-redirect.mjs` does not exist.

- [ ] **Step 6: Implement the bounded verifier**

Create `scripts/verify-www-redirect.mjs` exporting:

```js
export async function verifyWWWRedirect({ fetchImpl = globalThis.fetch } = {})
```

The function must request these URLs with `redirect: "manual"` and an AbortSignal timeout:

```js
[
  ["https://www.macmlx.app/", "https://macmlx.app/"],
  ["https://www.macmlx.app/zh/", "https://macmlx.app/zh/"],
  ["https://www.macmlx.app/models/choosing-a-model/?q=mlx%20swift", "https://macmlx.app/models/choosing-a-model/?q=mlx%20swift"],
]
```

Require status 308 and an exact `Location` match. Throw one aggregate error listing every failed case. When run directly, print the successful check count.

- [ ] **Step 7: Verify GREEN and dry-run syntax**

Run:

```sh
node --test site/tests/www-redirect.test.mjs
node --check site/cloudflare/www-redirect.mjs
node --check scripts/verify-www-redirect.mjs
WRANGLER_LOG_PATH=/tmp/macmlx-www-dry-run.log bun wrangler deploy --dry-run --config /Users/kevin/Projects/macmlx/.worktrees/engine-scroll-story/wrangler.www.jsonc
```

Expected: tests PASS, syntax checks produce no output, and Wrangler reports a dry-run for `macmlx-www-redirect` with no bindings.

- [ ] **Step 8: Commit config and verifier**

Stage only the test, `wrangler.www.jsonc`, and verifier, then commit with a Lore-formatted message beginning `test(site): make the www cutover independently verifiable`.

### Task 3: Document, deploy, and verify the cutover

**Files:**
- Modify: `site/README.md`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add failing maintenance assertions**

Extend `site/tests/www-redirect.test.mjs` to assert that the README includes the exact Bun Wrangler dry-run/deploy/version/rollback commands for `wrangler.www.jsonc`, records the DNS rollback value `www.macmlx.app CNAME macmlx.app` as DNS-only, and runs `scripts/verify-www-redirect.mjs`. Assert CI syntax-checks both new JavaScript files.

- [ ] **Step 2: Run the test and verify RED**

Run `node --test site/tests/www-redirect.test.mjs`.

Expected: FAIL because the runbook and CI do not yet reference the redirect Worker.

- [ ] **Step 3: Update the runbook and CI**

Document this sequence in `site/README.md`:

```sh
WRANGLER_LOG_PATH=/tmp/macmlx-www.log bun wrangler deploy --dry-run --config "$SITE_ROOT/wrangler.www.jsonc"
WRANGLER_LOG_PATH=/tmp/macmlx-www.log bun wrangler deploy --config "$SITE_ROOT/wrangler.www.jsonc"
node "$SITE_ROOT/scripts/verify-www-redirect.mjs"
WRANGLER_LOG_PATH=/tmp/macmlx-www.log bun wrangler versions list --config "$SITE_ROOT/wrangler.www.jsonc"
```

Include DNS restore and Worker rollback instructions. Add `node --check` commands for both new modules to CI.

- [ ] **Step 4: Run the complete local verification**

Run:

```sh
MACMLX_NODE_MODULES=/Users/kevin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node --test site/tests/*.test.mjs
node scripts/build-public-site.mjs
node scripts/crawl-public-site.mjs
node scripts/test-public-site.mjs
node --check site/cloudflare/www-redirect.mjs
node --check scripts/verify-www-redirect.mjs
git diff --check
```

Expected: every test passes, the crawler reports 77 files and 28 HTML documents, the bilingual regression reports 97 nodes, and the diff check is clean.

- [ ] **Step 5: Record and delete the conflicting DNS record**

In Cloudflare, verify the exact record is `www.macmlx.app`, type `CNAME`, target `macmlx.app`, DNS-only. Delete only that record. The recorded rollback is recreating the same DNS-only CNAME.

- [ ] **Step 6: Deploy the redirect Worker**

From the Bun package directory, run:

```sh
WRANGLER_LOG_PATH=/tmp/macmlx-www.log bun wrangler deploy --config /Users/kevin/Projects/macmlx/.worktrees/engine-scroll-story/wrangler.www.jsonc
```

Expected: Custom Domain `www.macmlx.app` and a current version ID.

- [ ] **Step 7: Verify production and apex regression**

Run:

```sh
node /Users/kevin/Projects/macmlx/.worktrees/engine-scroll-story/scripts/verify-www-redirect.mjs
node /Users/kevin/Projects/macmlx/.worktrees/engine-scroll-story/scripts/verify-cloudflare-deploy.mjs https://macmlx.app/
```

Expected: all www checks pass and all 58 apex checks still pass. Also confirm the public DNS no longer resolves to `68.64.178.113`, HTTPS succeeds, and the browser lands on the apex URL.

- [ ] **Step 8: Commit the runbook and deployed state**

Stage only the README, CI, and any final redirect test update. Commit with a Lore-formatted message beginning `docs(site): make www releases reversible` and record the deployed version plus all verification evidence in the commit trailers.
