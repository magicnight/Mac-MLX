# Signal M Website Logo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current generic website mark with the approved Signal M vector system, deterministic raster icons, metadata, and social-card integration without changing the site layout or copy.

**Architecture:** Keep one canonical 128×128 SVG and one optically simplified favicon SVG as tracked sources. Render required PNG icons deterministically with the existing optional Sharp runtime, copy every tracked brand asset through the deterministic site build, and expose them through centralized templates and metadata.

**Tech Stack:** SVG, JavaScript ES modules, Node test runner, Sharp, deterministic static-site builder, Bun-managed Wrangler.

---

## File structure

- Create `site/assets/brand/macmlx-mark.svg`: canonical Signal M source.
- Create `site/assets/brand/favicon.svg`: 16-pixel optical variant.
- Create `site/assets/brand/apple-touch-icon.png`, `icon-192.png`, `icon-512.png`: tracked deterministic raster outputs.
- Create `site/assets/brand/site.webmanifest`: icon manifest.
- Create `scripts/render-brand-icons.mjs`: atomic SVG-to-PNG renderer using the existing Sharp discovery contract.
- Create `site/tests/brand-logo.test.mjs`: geometry, safety, raster, template, metadata, and determinism tests.
- Modify `site/content/assets.mjs`: add brand assets to the explicit copied-asset manifest.
- Modify `site/templates/home.html`, `scripts/build-public-site.mjs`: replace inline three-bar marks with `/assets/brand/macmlx-mark.svg`.
- Modify `site/lib/metadata.mjs`, `site/templates/home.html`, `site/templates/article.html`: emit favicon, touch icon, and manifest links once per document.
- Modify `site/social-card.html`: use the canonical mark.
- Modify `site/tests/social-card.test.mjs`, `site/assets/social/og-en.png`, `site/assets/social/og-zh.png`: lock and refresh social cards.
- Modify `.github/workflows/ci.yml`, `site/README.md`: validate brand sources and document refresh commands.

### Task 1: Canonical vector and optical favicon

**Files:**
- Create: `site/tests/brand-logo.test.mjs`
- Create: `site/assets/brand/macmlx-mark.svg`
- Create: `site/assets/brand/favicon.svg`

- [ ] **Step 1: Write failing vector contract tests**

Create `site/tests/brand-logo.test.mjs` with tests that read both SVG files and assert:

```js
assert.match(mark, /viewBox="0 0 128 128"/);
assert.match(mark, /fill="#F3F1EA"/);
assert.match(mark, /stroke="#111311"/);
assert.match(mark, /fill="#7196FF"/);
assert.match(mark, /fill="#89E67A"/);
assert.doesNotMatch(mark, /<script|https?:|data:/i);
assert.match(favicon, /viewBox="0 0 128 128"/);
assert.doesNotMatch(favicon, /#89E67A/);
assert.doesNotMatch(favicon, /<script|https?:|data:/i);
```

Parse each SVG with `xmllint --noout` in the verification step.

- [ ] **Step 2: Run RED**

Run `node --test site/tests/brand-logo.test.mjs`.

Expected: FAIL because the brand SVG files do not exist.

- [ ] **Step 3: Create the canonical Signal M SVG**

Create `site/assets/brand/macmlx-mark.svg` with this exact geometry:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" role="img" aria-labelledby="title">
  <title id="title">macMLX Signal M</title>
  <rect x="4" y="4" width="120" height="120" rx="34" fill="#F3F1EA"/>
  <path d="M28 88V39l36 38 36-38v49" fill="none" stroke="#111311" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="64" cy="77" r="8.5" fill="#7196FF"/>
  <circle cx="100" cy="39" r="6" fill="#89E67A"/>
</svg>
```

Create `site/assets/brand/favicon.svg` with the same squircle and `M`, a 16-unit stroke, the blue activation node, and no green signal node.

- [ ] **Step 4: Run GREEN**

Run:

```sh
node --test site/tests/brand-logo.test.mjs
xmllint --noout site/assets/brand/macmlx-mark.svg site/assets/brand/favicon.svg
```

Expected: all focused tests PASS; `xmllint` produces no output.

- [ ] **Step 5: Commit vector sources**

Stage the two SVG files and focused test. Commit with a Lore-formatted message beginning `feat(brand): give macMLX the Signal M mark`.

### Task 2: Deterministic raster icons and manifest

**Files:**
- Modify: `site/tests/brand-logo.test.mjs`
- Create: `scripts/render-brand-icons.mjs`
- Create: `site/assets/brand/apple-touch-icon.png`
- Create: `site/assets/brand/icon-192.png`
- Create: `site/assets/brand/icon-512.png`
- Create: `site/assets/brand/site.webmanifest`

- [ ] **Step 1: Add failing raster and manifest tests**

Add tests that parse each PNG signature and IHDR dimensions, requiring 180×180, 192×192, and 512×512 RGBA or RGB output. Parse `site.webmanifest` and assert:

```js
assert.equal(manifest.name, "macMLX");
assert.equal(manifest.short_name, "macMLX");
assert.equal(manifest.start_url, "/");
assert.equal(manifest.display, "standalone");
assert.equal(manifest.background_color, "#111311");
assert.equal(manifest.theme_color, "#111311");
assert.deepEqual(manifest.icons, [
  { src: "/assets/brand/icon-192.png", sizes: "192x192", type: "image/png" },
  { src: "/assets/brand/icon-512.png", sizes: "512x512", type: "image/png" },
]);
```

Add a Sharp-conditional test that renders twice into separate temporary directories and compares SHA-256 hashes for all three PNGs.

- [ ] **Step 2: Run RED**

Run `node --test site/tests/brand-logo.test.mjs`.

Expected: FAIL because the renderer, PNGs, and manifest do not exist.

- [ ] **Step 3: Implement atomic rendering**

Create `scripts/render-brand-icons.mjs` following `scripts/render-social-cards.mjs` Sharp discovery and rollback patterns. Export:

```js
export async function renderBrandIcons({
  source = new URL("../site/assets/brand/macmlx-mark.svg", import.meta.url),
  outputDirectory = new URL("../site/assets/brand/", import.meta.url),
  sharpImpl,
} = {})
```

Render a transparent 180-, 192-, and 512-pixel square from the canonical SVG into a unique staging directory. Validate every PNG before atomically replacing the tracked trio. Restore the previous trio if any render or replacement fails.

- [ ] **Step 4: Create the exact web manifest**

Create `site/assets/brand/site.webmanifest` containing only the asserted name, short name, start URL, display mode, colors, and two icon entries.

- [ ] **Step 5: Render tracked PNGs and verify GREEN**

Run:

```sh
MACMLX_NODE_MODULES=/Users/kevin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node scripts/render-brand-icons.mjs
MACMLX_NODE_MODULES=/Users/kevin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node --test site/tests/brand-logo.test.mjs
node --check scripts/render-brand-icons.mjs
```

Expected: three icons render, every focused test passes, and syntax checking is silent.

- [ ] **Step 6: Commit deterministic icon outputs**

Stage the renderer, manifest, PNGs, and test. Commit with a Lore-formatted message beginning `feat(brand): make Signal M portable across web surfaces`.

### Task 3: Integrate brand assets into every page

**Files:**
- Modify: `site/tests/brand-logo.test.mjs`
- Modify: `site/content/assets.mjs`
- Modify: `site/templates/home.html`
- Modify: `site/templates/article.html`
- Modify: `scripts/build-public-site.mjs`
- Modify: `site/lib/metadata.mjs`
- Modify: `site/assets/css/main.css`

- [ ] **Step 1: Add failing build integration tests**

Build the prepared site in the test and assert every generated HTML document contains exactly one each of:

```html
<link rel="icon" href="/assets/brand/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/assets/brand/apple-touch-icon.png">
<link rel="manifest" href="/assets/brand/site.webmanifest">
```

Assert home and article headers use `<img class="brand-mark" src="/assets/brand/macmlx-mark.svg" alt="">`, all local brand URLs resolve, the explicit asset manifest includes all six brand files, and the old three `<i>` bars are absent.

- [ ] **Step 2: Run RED**

Run `node --test site/tests/brand-logo.test.mjs`.

Expected: FAIL because pages and the copied asset manifest do not reference the new brand system.

- [ ] **Step 3: Add brand assets to the explicit manifest**

Add these paths to `site/content/assets.mjs`:

```js
"brand/macmlx-mark.svg",
"brand/favicon.svg",
"brand/apple-touch-icon.png",
"brand/icon-192.png",
"brand/icon-512.png",
"brand/site.webmanifest",
```

- [ ] **Step 4: Replace inline marks**

Replace every `<span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>` in home, article header rendering, and footers with:

```html
<img class="brand-mark" src="/assets/brand/macmlx-mark.svg" alt="">
```

Keep the linked wordmark's existing localized `aria-label`. Update `.brand-mark` CSS to a block image with the existing 27-pixel desktop and 24-pixel compact dimensions; remove all `.brand-mark i` rules.

- [ ] **Step 5: Centralize document icon metadata**

Add this metadata helper and insert its return value once in `site/templates/home.html` and `site/templates/article.html` through the existing template-token rendering path. Do not duplicate tags in page-specific rendering.

```js
export function renderBrandLinks() {
  return '<link rel="icon" href="/assets/brand/favicon.svg" type="image/svg+xml"><link rel="apple-touch-icon" href="/assets/brand/apple-touch-icon.png"><link rel="manifest" href="/assets/brand/site.webmanifest">';
}
```

- [ ] **Step 6: Run GREEN and build checks**

Run:

```sh
node --test site/tests/brand-logo.test.mjs site/tests/home-build.test.mjs site/tests/build-content.test.mjs site/tests/metadata.test.mjs
node scripts/build-public-site.mjs
node scripts/crawl-public-site.mjs
node scripts/test-public-site.mjs
```

Expected: focused tests pass; the crawler reports 83 generated files and 28 HTML documents; the bilingual regression reports 97 nodes.

- [ ] **Step 7: Commit page integration**

Stage only the manifest, templates, builder, metadata, CSS, and tests. Commit with a Lore-formatted message beginning `feat(site): carry Signal M through every page`.

### Task 4: Social cards, CI, and release documentation

**Files:**
- Modify: `site/tests/brand-logo.test.mjs`
- Modify: `site/tests/social-card.test.mjs`
- Modify: `site/social-card.html`
- Modify: `site/assets/social/og-en.png`
- Modify: `site/assets/social/og-zh.png`
- Modify: `.github/workflows/ci.yml`
- Modify: `site/README.md`

- [ ] **Step 1: Add failing social and maintenance tests**

Assert the social-card source embeds `/assets/brand/macmlx-mark.svg` or its exact canonical SVG geometry and no longer contains `.mark i` bars. Assert CI syntax-checks `scripts/render-brand-icons.mjs`, validates both brand SVGs with `xmllint`, and the README documents the exact render command with `MACMLX_NODE_MODULES`.

- [ ] **Step 2: Run RED**

Run `node --test site/tests/brand-logo.test.mjs site/tests/social-card.test.mjs site/tests/ci-workflow.test.mjs site/tests/maintenance.test.mjs`.

Expected: FAIL because the social source, CI, and runbook still describe the old mark.

- [ ] **Step 3: Integrate Signal M into the social source**

Replace the old three-bar `.mark` HTML and CSS in `site/social-card.html` with the exact canonical Signal M SVG. Preserve the existing 1200×630 layout, registry-driven locale copy, and version.

- [ ] **Step 4: Update CI and runbook**

Add:

```sh
node --check scripts/render-brand-icons.mjs
xmllint --noout site/assets/brand/macmlx-mark.svg site/assets/brand/favicon.svg
```

to the website CI job. Document the exact brand render command before social-card refresh in `site/README.md`.

- [ ] **Step 5: Regenerate social cards and verify GREEN**

Run:

```sh
MACMLX_NODE_MODULES=/Users/kevin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node scripts/render-social-cards.mjs
MACMLX_NODE_MODULES=/Users/kevin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node --test site/tests/*.test.mjs
node scripts/build-public-site.mjs
node scripts/crawl-public-site.mjs
node scripts/test-public-site.mjs
git diff --check
```

Expected: every test passes, both 1200×630 social PNGs validate, the deterministic build passes, and the diff is clean.

- [ ] **Step 6: Commit social and maintenance integration**

Stage the social source and PNGs, CI, README, and related tests. Commit with a Lore-formatted message beginning `feat(brand): make Signal M the public face of macMLX`.

### Task 5: Combine with the approved www redirect release

**Files:**
- Execute: `docs/superpowers/plans/2026-07-11-www-redirect.md`
- Verify: `scripts/verify-cloudflare-deploy.mjs`
- Verify: `scripts/verify-www-redirect.mjs`

- [ ] **Step 1: Execute the www redirect plan through local GREEN**

Complete Tasks 1–3 of `docs/superpowers/plans/2026-07-11-www-redirect.md` through its full local verification before changing DNS.

- [ ] **Step 2: Dry-run both production configs**

From `/Users/kevin/Projects/macmlx/public`, run:

```sh
WRANGLER_LOG_PATH=/tmp/macmlx-site-final.log bun wrangler deploy --dry-run --config /Users/kevin/Projects/macmlx/.worktrees/engine-scroll-story/wrangler.jsonc --env=""
WRANGLER_LOG_PATH=/tmp/macmlx-www-final.log bun wrangler deploy --dry-run --config /Users/kevin/Projects/macmlx/.worktrees/engine-scroll-story/wrangler.www.jsonc
```

Expected: both dry-runs succeed; the apex reads the full static asset set and the redirect Worker reports no bindings.

- [ ] **Step 3: Deploy and verify the apex logo release**

Deploy `wrangler.jsonc`, run `node scripts/verify-cloudflare-deploy.mjs https://macmlx.app/`, and browser-check the Signal M mark in desktop/mobile, light/dark, English/Chinese, and favicon contexts. Record the apex version ID.

- [ ] **Step 4: Cut over and verify www**

Confirm and delete only `www.macmlx.app CNAME macmlx.app` in Cloudflare, deploy `wrangler.www.jsonc`, run `node scripts/verify-www-redirect.mjs`, and rerun the 58-check apex verifier. Record the redirect Worker version ID and DNS rollback value.

- [ ] **Step 5: Final release commit**

Commit any final tested runbook evidence with Lore trailers listing the complete test count, build/crawler evidence, both Worker version IDs, browser QA, and the unchanged rollback paths.
