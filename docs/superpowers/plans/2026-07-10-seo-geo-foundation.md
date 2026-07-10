# SEO/GEO Static Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce crawlable English `/` and Simplified Chinese `/zh/` home pages from the existing visual landing page without client-side translation or new dependencies.

**Architecture:** Keep the current landing-page HTML as a source template with its existing bilingual attributes, then localize it at build time with a small dependency-free renderer. A route manifest owns canonical paths and metadata; the build script emits complete static HTML into ignored `public/` output. Browser JavaScript retains theme, copy, reveal, and engine-story behavior but only migrates legacy `?lang=` home links.

**Tech Stack:** Node.js built-ins, static HTML, CSS, vanilla JavaScript, `node:test`, existing site regression script, in-app browser QA.

---

## Sequence and Dependency

This is plan 1 of 3. Complete it before:

1. `docs/superpowers/plans/2026-07-10-seo-geo-content-hub.md`
2. `docs/superpowers/plans/2026-07-10-seo-geo-discovery-validation.md`

## File Map and Boundaries

- Create `site/lib/localize.mjs` — bilingual markup localization, HTML escaping, and strict token rendering.
- Create `site/lib/routes.mjs` — route validation and URL/output-path helpers.
- Create `site/content/project.mjs` — origin, release, locale, and home-page metadata.
- Create `site/routes.mjs` — the English and Chinese home route manifest; later plans extend it.
- Create `site/templates/home.html` — the current home page copied as a source template with build tokens and root-absolute assets.
- Create `site/assets/css/main.css` and `site/assets/css/no-js.css` — tracked source copies of the approved shared styles.
- Create `site/assets/js/main.js` — tracked source copy of the approved progressive-enhancement script.
- Create `site/assets/images/` and `site/assets/og-image.svg` — tracked website image sources copied unchanged into generated output.
- Create `site/tests/localize.test.mjs` — unit tests for localization and token safety.
- Create `site/tests/routes.test.mjs` — unit tests for route symmetry and validation.
- Create `site/tests/build-home.test.mjs` — integration test for generated home pages.
- Create `scripts/build-public-site.mjs` — deterministic static-site build entry point.
- Modify `site/assets/js/main.js:4-71,191-197` — remove runtime translation and add query migration.
- Modify `site/assets/css/main.css:120-148,339-346,351-414` — anchor-style language switch and home-to-hub navigation hooks.
- Modify `scripts/test-public-site.mjs:68-79,146-153,243-289` — generated-language and migration contracts.

`public/` remains ignored and separately deployed. Commit `site/`, `scripts/build-public-site.mjs`, and test sources; do not force-add generated `public/` files.

### Task 1: Build and Test the Strict Localization Primitive

**Files:**
- Create: `site/tests/localize.test.mjs`
- Create: `site/lib/localize.mjs`

- [ ] **Step 1: Write the failing localization tests**

Create `site/tests/localize.test.mjs`:

```js
import assert from "node:assert/strict";
import test from "node:test";
import {
  escapeHTML,
  localizeBilingualMarkup,
  renderTokens,
} from "../lib/localize.mjs";

test("escapeHTML protects text and attributes", () => {
  assert.equal(escapeHTML(`<Mac & "MLX">`), "&lt;Mac &amp; &quot;MLX&quot;&gt;");
});

test("localizeBilingualMarkup emits one static language", () => {
  const source = `<h1 class="hero" data-en="Your Mac.<br><em>One engine.</em>" data-zh="你的 Mac。<br><em>一个引擎。</em>">fallback</h1>`;
  assert.equal(
    localizeBilingualMarkup(source, "en"),
    `<h1 class="hero">Your Mac.<br><em>One engine.</em></h1>`,
  );
  assert.equal(
    localizeBilingualMarkup(source, "zh-Hans"),
    `<h1 class="hero">你的 Mac。<br><em>一个引擎。</em></h1>`,
  );
});

test("localizeBilingualMarkup rejects incomplete language pairs", () => {
  assert.throws(
    () => localizeBilingualMarkup(`<p data-en="English">fallback</p>`, "en"),
    /unpaired bilingual attribute/,
  );
});

test("renderTokens escapes normal values and permits named trusted fragments", () => {
  const rendered = renderTokens(
    `<html lang="{{htmlLang}}"><body>{{title}}{{{languageLink}}}</body></html>`,
    {
      htmlLang: "zh-CN",
      title: `<unsafe>`,
      languageLink: `<a href="/">EN</a>`,
    },
    new Set(["languageLink"]),
  );
  assert.equal(
    rendered,
    `<html lang="zh-CN"><body>&lt;unsafe&gt;<a href="/">EN</a></body></html>`,
  );
});

test("renderTokens fails on missing, unresolved, or untrusted raw tokens", () => {
  assert.throws(() => renderTokens("{{missing}}", {}), /missing token: missing/);
  assert.throws(
    () => renderTokens("{{{unsafe}}}", { unsafe: "<b>no</b>" }),
    /raw token is not trusted: unsafe/,
  );
  assert.throws(
    () => renderTokens("{{first}} {{second}}", { first: "done" }),
    /missing token: second/,
  );
});
```

- [ ] **Step 2: Run the tests and verify the red state**

Run:

```bash
node --test site/tests/localize.test.mjs
```

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `site/lib/localize.mjs`.

- [ ] **Step 3: Implement the minimal localization module**

Create `site/lib/localize.mjs`:

```js
const tokenPattern = /\{\{\{([a-zA-Z0-9_.-]+)\}\}\}|\{\{([a-zA-Z0-9_.-]+)\}\}/g;
const bilingualElementPattern = /<([a-z][\w:-]*)([^<>]*\sdata-en="([^"]*)"[^<>]*\sdata-zh="([^"]*)"[^<>]*)>([\s\S]*?)<\/\1>/gi;

export function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function localizeBilingualMarkup(source, locale) {
  if (locale !== "en" && locale !== "zh-Hans") {
    throw new Error(`unsupported locale: ${locale}`);
  }
  if (/data-en=/.test(source) !== /data-zh=/.test(source)) {
    throw new Error("unpaired bilingual attribute");
  }

  let output = source;
  let previous;
  do {
    previous = output;
    output = output.replace(
      bilingualElementPattern,
      (_match, tag, attributes, english, chinese) => {
        const cleanAttributes = attributes
          .replace(/\sdata-en="[^"]*"/g, "")
          .replace(/\sdata-zh="[^"]*"/g, "");
        const content = locale === "zh-Hans" ? chinese : english;
        return `<${tag}${cleanAttributes}>${content}</${tag}>`;
      },
    );
  } while (output !== previous && /data-(?:en|zh)=/.test(output));

  if (/data-en=|data-zh=/.test(output)) {
    throw new Error("unpaired bilingual attribute");
  }
  return output;
}

export function renderTokens(source, values, trustedRawTokens = new Set()) {
  const rendered = source.replace(tokenPattern, (_match, rawName, escapedName) => {
    const name = rawName || escapedName;
    if (!(name in values)) throw new Error(`missing token: ${name}`);
    if (rawName) {
      if (!trustedRawTokens.has(name)) throw new Error(`raw token is not trusted: ${name}`);
      return String(values[name]);
    }
    return escapeHTML(values[name]);
  });
  if (/\{\{\{?[a-zA-Z0-9_.-]+\}?\}\}/.test(rendered)) {
    throw new Error("unresolved template token");
  }
  return rendered;
}
```

- [ ] **Step 4: Run the tests and verify green**

Run:

```bash
node --test site/tests/localize.test.mjs
```

Expected: 5 tests PASS with no warnings.

- [ ] **Step 5: Commit the localization primitive**

```bash
git add site/lib/localize.mjs site/tests/localize.test.mjs
git commit -m "feat(site): make locale output static and strict" \
  -m "Constraint: Use no third-party parser or runtime translation" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: node --test site/tests/localize.test.mjs"
```

### Task 2: Define and Validate Canonical Home Routes

**Files:**
- Create: `site/content/project.mjs`
- Create: `site/routes.mjs`
- Create: `site/lib/routes.mjs`
- Create: `site/tests/routes.test.mjs`

- [ ] **Step 1: Write the failing route-manifest tests**

Create `site/tests/routes.test.mjs`:

```js
import assert from "node:assert/strict";
import test from "node:test";
import { project } from "../content/project.mjs";
import { routes } from "../routes.mjs";
import { outputFileForPath, validateRoutes } from "../lib/routes.mjs";

test("project release metadata is explicit", () => {
  assert.equal(project.version, "0.5.3");
  assert.equal(project.releaseDate, "2026-07-08");
  assert.equal(project.lastVerified, "2026-07-10");
});

test("home routes are canonical and reciprocal", () => {
  assert.doesNotThrow(() => validateRoutes(routes));
  assert.deepEqual(routes[0].paths, { en: "/", "zh-Hans": "/zh/" });
  assert.equal(routes[0].metadata.en.title, "macMLX — Native Swift inference for Apple Silicon");
  assert.equal(routes[0].metadata["zh-Hans"].title, "macMLX — Apple 芯片上的原生 Swift 推理");
});

test("outputFileForPath maps directory URLs to index files", () => {
  assert.equal(outputFileForPath("/"), "index.html");
  assert.equal(outputFileForPath("/zh/"), "zh/index.html");
  assert.throws(() => outputFileForPath("/unsafe"), /route must end with/);
});

test("validateRoutes rejects duplicate canonicals and missing counterparts", () => {
  assert.throws(
    () => validateRoutes([{ ...routes[0] }, { ...routes[0], id: "duplicate" }]),
    /duplicate route path/,
  );
  assert.throws(
    () => validateRoutes([{ ...routes[0], paths: { en: "/" } }]),
    /missing locale path/,
  );
});
```

- [ ] **Step 2: Verify the tests fail for missing modules**

Run:

```bash
node --test site/tests/routes.test.mjs
```

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `site/content/project.mjs`.

- [ ] **Step 3: Add the project and route data**

Create `site/content/project.mjs`:

```js
export const project = Object.freeze({
  name: "macMLX",
  origin: "https://macmlx.app",
  repository: "https://github.com/magicnight/mac-mlx",
  downloadURL: "https://github.com/magicnight/mac-mlx/releases/latest",
  version: "0.5.3",
  releaseDate: "2026-07-08",
  lastVerified: "2026-07-10",
  licenseURL: "https://www.apache.org/licenses/LICENSE-2.0",
  locales: ["en", "zh-Hans"],
  htmlLanguages: { en: "en", "zh-Hans": "zh-CN" },
});
```

Create `site/routes.mjs`:

```js
export const routes = [
  {
    id: "home",
    kind: "home",
    template: "home",
    paths: { en: "/", "zh-Hans": "/zh/" },
    metadata: {
      en: {
        title: "macMLX — Native Swift inference for Apple Silicon",
        description: "Run local language and vision models through a native SwiftUI app, CLI, and compatible API, all powered by one Swift in-process MLX engine.",
        socialDescription: "A native SwiftUI app, CLI, and compatible API over one in-process MLX engine.",
      },
      "zh-Hans": {
        title: "macMLX — Apple 芯片上的原生 Swift 推理",
        description: "通过原生 SwiftUI 应用、CLI 与兼容 API 在 Mac 上运行本地语言和视觉模型，共用一个 Swift 进程内 MLX 引擎。",
        socialDescription: "原生 SwiftUI 应用、CLI 与兼容 API，共用一个 Swift 进程内 MLX 引擎。",
      },
    },
  },
];
```

- [ ] **Step 4: Implement route validation and output mapping**

Create `site/lib/routes.mjs`:

```js
const locales = ["en", "zh-Hans"];

export function outputFileForPath(routePath) {
  if (!routePath.startsWith("/") || !routePath.endsWith("/")) {
    throw new Error(`route must end with /: ${routePath}`);
  }
  return routePath === "/" ? "index.html" : `${routePath.slice(1)}index.html`;
}

export function validateRoutes(routes) {
  const seenIds = new Set();
  const seenPaths = new Set();
  for (const route of routes) {
    if (!route.id || seenIds.has(route.id)) throw new Error(`duplicate route id: ${route.id}`);
    seenIds.add(route.id);
    for (const locale of locales) {
      const routePath = route.paths?.[locale];
      if (!routePath) throw new Error(`missing locale path: ${route.id}/${locale}`);
      outputFileForPath(routePath);
      if (seenPaths.has(routePath)) throw new Error(`duplicate route path: ${routePath}`);
      seenPaths.add(routePath);
      const metadata = route.metadata?.[locale];
      if (!metadata?.title || !metadata?.description || !metadata?.socialDescription) {
        throw new Error(`missing metadata: ${route.id}/${locale}`);
      }
    }
  }
}

export function canonicalURL(origin, routePath) {
  return new URL(routePath, origin).href;
}
```

- [ ] **Step 5: Run tests and commit the manifest contract**

Run:

```bash
node --test site/tests/routes.test.mjs
```

Expected: 4 tests PASS.

Commit:

```bash
git add site/content/project.mjs site/routes.mjs site/lib/routes.mjs site/tests/routes.test.mjs
git commit -m "feat(site): give each language a canonical home" \
  -m "Constraint: English stays at / and Simplified Chinese lives at /zh/" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: node --test site/tests/routes.test.mjs"
```

### Task 3: Generate Complete English and Chinese Home Documents

**Files:**
- Create: `site/templates/home.html`
- Create: `site/tests/build-home.test.mjs`
- Create: `scripts/build-public-site.mjs`
- Modify: `site/lib/routes.mjs`
- Generate: `public/index.html`
- Generate: `public/zh/index.html`

- [ ] **Step 1: Write the failing home-build integration test**

Create `site/tests/build-home.test.mjs`:

```js
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { buildSite } from "../../scripts/build-public-site.mjs";

test("buildSite emits static counterpart home pages", async () => {
  await buildSite();
  const english = await readFile(new URL("../../public/index.html", import.meta.url), "utf8");
  const chinese = await readFile(new URL("../../public/zh/index.html", import.meta.url), "utf8");

  assert.match(english, /<html lang="en">/);
  assert.match(chinese, /<html lang="zh-CN">/);
  assert.match(english, /<h1[^>]*>Your Mac\./);
  assert.match(chinese, /<h1[^>]*>你的 Mac。/);
  assert.doesNotMatch(english, /data-en=|data-zh=/);
  assert.doesNotMatch(chinese, /data-en=|data-zh=/);
  assert.match(english, /rel="canonical" href="https:\/\/macmlx\.app\/"/);
  assert.match(chinese, /rel="canonical" href="https:\/\/macmlx\.app\/zh\/"/);
  assert.match(english, /hreflang="zh-Hans" href="https:\/\/macmlx\.app\/zh\/"/);
  assert.match(chinese, /hreflang="en" href="https:\/\/macmlx\.app\/"/);
  assert.match(english, /class="language-link" href="\/zh\/"/);
  assert.match(chinese, /class="language-link" href="\/"/);
  assert.match(chinese, /src="\/assets\/js\/main\.js\?v=\d+"/);
  assert.match(chinese, /src="\/assets\/images\/engine\/mac-silicon-foundation\.webp"/);
});
```

- [ ] **Step 2: Run the integration test and confirm red**

Run:

```bash
node --test site/tests/build-home.test.mjs
```

Expected: FAIL because `scripts/build-public-site.mjs` does not exist.

- [ ] **Step 3: Create the home source template mechanically**

Create `site/templates/home.html` as an exact copy of the current `public/index.html`. Create the tracked asset source mechanically from the current approved output:

```bash
mkdir -p site/assets/css site/assets/js site/assets/images
cp public/assets/css/main.css site/assets/css/main.css
cp public/assets/css/no-js.css site/assets/css/no-js.css
cp public/assets/js/main.js site/assets/js/main.js
cp -R public/assets/images/. site/assets/images/
cp public/og-image.svg site/assets/og-image.svg
```

Update the existing test fixture path in `scripts/test-public-site.mjs` so the source-of-truth SVG follows the generated asset location:

```js
const ogImage = await readFile(new URL("public/assets/og-image.svg", root), "utf8");
```

Then make only these source-template changes:

```diff
-<html lang="en">
+<html lang="{{htmlLang}}">
-  <title>macMLX — Native Swift inference for Apple Silicon</title>
+  <title>{{title}}</title>
-  <meta name="description" content="Run local language and vision models through a native SwiftUI app, CLI, and compatible API — all powered by one Swift in-process MLX engine.">
+  <meta name="description" content="{{description}}">
-  <link rel="canonical" href="https://macmlx.app/">
+  <link rel="canonical" href="{{canonical}}">
-  <link rel="alternate" hreflang="en" href="https://macmlx.app/">
+  <link rel="alternate" hreflang="en" href="{{englishURL}}">
-  <link rel="alternate" hreflang="zh-Hans" href="https://macmlx.app/?lang=zh">
+  <link rel="alternate" hreflang="zh-Hans" href="{{chineseURL}}">
-  <link rel="alternate" hreflang="x-default" href="https://macmlx.app/">
+  <link rel="alternate" hreflang="x-default" href="{{englishURL}}">
```

Apply the same `{{canonical}}`, `{{title}}`, `{{socialDescription}}`, `{{socialImage}}`, and `{{socialImageAlt}}` tokens to Open Graph and Twitter metadata. Replace the existing JSON-LD block with `{{{jsonLd}}}`. Change every local stylesheet, script, and image URL to start with `/` so `/zh/` resolves the same assets.

Replace the language button exactly:

```html
{{{languageLink}}}
```

Add a visible Learn link after the Engine anchor:

```html
<a href="{{learnPath}}" data-en="Learn" data-zh="了解">Learn</a>
```

Set body language without runtime translation:

```html
<body data-lang="{{bodyLanguage}}">
```

Use root-absolute Markdown and asset links:

```html
<link rel="alternate" type="text/markdown" title="{{markdownTitle}}" href="{{markdownPath}}">
<link rel="stylesheet" href="/assets/css/main.css?v=2026071004">
<link rel="stylesheet" href="/assets/css/no-js.css?v=2026071004">
<script src="/assets/js/main.js?v=2026071004" defer></script>
```

- [ ] **Step 4: Implement the static build entry point**

Add this helper to `site/lib/routes.mjs`:

```js
export function counterpartPath(route, locale) {
  return route.paths[locale === "en" ? "zh-Hans" : "en"];
}
```

Create `scripts/build-public-site.mjs`:

```js
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { project } from "../site/content/project.mjs";
import { localizeBilingualMarkup, renderTokens } from "../site/lib/localize.mjs";
import {
  canonicalURL,
  counterpartPath,
  outputFileForPath,
  validateRoutes,
} from "../site/lib/routes.mjs";
import { routes } from "../site/routes.mjs";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const trustedRawTokens = new Set(["jsonLd", "languageLink"]);

function softwareEntities(locale, canonical, description) {
  return JSON.stringify({
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebSite",
        "@id": `${canonical}#website`,
        name: project.name,
        url: canonical,
        inLanguage: locale === "en" ? "en" : "zh-CN",
      },
      {
        "@type": "SoftwareApplication",
        "@id": `${project.origin}/#software`,
        name: project.name,
        url: canonical,
        codeRepository: project.repository,
        downloadUrl: project.downloadURL,
        operatingSystem: "macOS 14.0+ on Apple Silicon",
        applicationCategory: "DeveloperApplication",
        softwareVersion: project.version,
        dateModified: project.lastVerified,
        license: project.licenseURL,
        programmingLanguage: "Swift",
        runtimePlatform: "Apple MLX on Apple Silicon",
        offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        description,
      },
    ],
  }, null, 2);
}

function languageLink(locale, counterpart) {
  const label = locale === "en" ? "中文" : "EN";
  const accessible = locale === "en" ? "阅读中文版" : "Read in English";
  return `<a class="language-link" href="${counterpart}" hreflang="${locale === "en" ? "zh-Hans" : "en"}" lang="${locale === "en" ? "zh-CN" : "en"}" aria-label="${accessible}">${label}</a>`;
}

function homeTokens(route, locale) {
  const metadata = route.metadata[locale];
  const canonical = canonicalURL(project.origin, route.paths[locale]);
  const englishURL = canonicalURL(project.origin, route.paths.en);
  const chineseURL = canonicalURL(project.origin, route.paths["zh-Hans"]);
  return {
    htmlLang: project.htmlLanguages[locale],
    bodyLanguage: locale === "en" ? "en" : "zh",
    title: metadata.title,
    description: metadata.description,
    socialDescription: metadata.socialDescription,
    canonical,
    englishURL,
    chineseURL,
    socialImage: `${project.origin}/assets/og-image.svg`,
    socialImageAlt: locale === "en" ? "macMLX for Apple Silicon" : "面向 Apple 芯片的 macMLX",
    jsonLd: `<script type="application/ld+json">${softwareEntities(locale, canonical, metadata.description)}</script>`,
    languageLink: languageLink(locale, counterpartPath(route, locale)),
    learnPath: locale === "en" ? "/architecture/" : "/zh/architecture/",
    markdownTitle: locale === "en" ? "LLM-friendly project summary" : "面向大模型的项目摘要",
    markdownPath: locale === "en" ? "/llms.txt" : "/zh/llms.txt",
  };
}

export async function buildSite() {
  validateRoutes(routes);
  await cp(resolve(repositoryRoot, "site/assets"), resolve(repositoryRoot, "public/assets"), {
    recursive: true,
    force: true,
  });
  const template = await readFile(resolve(repositoryRoot, "site/templates/home.html"), "utf8");
  for (const route of routes.filter((item) => item.kind === "home")) {
    for (const locale of project.locales) {
      const localized = localizeBilingualMarkup(template, locale);
      const output = renderTokens(localized, homeTokens(route, locale), trustedRawTokens);
      const outputPath = resolve(repositoryRoot, "public", outputFileForPath(route.paths[locale]));
      await mkdir(dirname(outputPath), { recursive: true });
      await writeFile(outputPath, output.endsWith("\n") ? output : `${output}\n`, "utf8");
    }
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await buildSite();
}
```

- [ ] **Step 5: Run the build test and inspect the generated diff**

Run:

```bash
node --test site/tests/build-home.test.mjs
node scripts/build-public-site.mjs
git diff --no-index --check site/templates/home.html public/index.html || test $? -eq 1
```

Expected: the test passes; both output files exist; the diff check reports no whitespace errors. Inspect the normal diff and confirm that English home visual structure, section order, engine images, and status boundaries are unchanged.

- [ ] **Step 6: Commit source and generator only**

```bash
git add site/templates/home.html site/assets site/tests/build-home.test.mjs site/lib/routes.mjs scripts/build-public-site.mjs scripts/test-public-site.mjs
git commit -m "feat(site): generate canonical bilingual home pages" \
  -m "Constraint: Preserve the approved landing-page DOM and ignored public output" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Directive: Keep all home assets root-absolute so /zh/ remains self-contained" \
  -m "Tested: node --test site/tests/build-home.test.mjs"
```

### Task 4: Replace Runtime Translation with Legacy URL Migration

**Files:**
- Modify: `site/assets/js/main.js:4-71,191-197`
- Modify: `site/templates/home.html`
- Modify: `scripts/test-public-site.mjs`
- Test: `site/tests/build-home.test.mjs`

- [ ] **Step 1: Add failing JavaScript-source assertions**

Add to `site/tests/build-home.test.mjs` after reading both pages:

```js
  const script = await readFile(new URL("../../public/assets/js/main.js", import.meta.url), "utf8");
  assert.doesNotMatch(script, /setTranslatedContent|querySelectorAll\("\[data-en\]\[data-zh\]"\)/);
  assert.match(script, /function initialiseLanguageMigration\(\)/);
  assert.match(script, /window\.location\.replace\(destination\)/);
```

Update `scripts/test-public-site.mjs` to assert:

```js
assert.doesNotMatch(js, /setTranslatedContent|originalContent|initialLanguage/);
assert.match(js, /function initialiseLanguageMigration\(\)/);
assert.match(js, /searchParams\.get\("lang"\)/);
assert.match(js, /window\.location\.replace\(destination\)/);
```

- [ ] **Step 2: Run the tests and verify they fail on runtime translation**

Run:

```bash
node --test site/tests/build-home.test.mjs
```

Expected: FAIL because `setTranslatedContent` still exists.

- [ ] **Step 3: Implement one-way migration and retain theme behavior**

In `site/assets/js/main.js`, remove `languageKey`, `originalContent`, `cacheEnglish`, `setTranslatedContent`, `setLanguage`, `initialLanguage`, and `initialiseLanguage`.

Add after `saveValue`:

```js
  function initialiseLanguageMigration() {
    const requested = new URLSearchParams(window.location.search).get("lang");
    if (requested !== "en" && requested !== "zh") return;
    if (window.location.pathname !== "/" && window.location.pathname !== "/zh/") return;

    const targetPath = requested === "zh" ? "/zh/" : "/";
    const destination = new URL(targetPath, window.location.origin);
    destination.hash = window.location.hash;
    if (`${window.location.pathname}${window.location.hash}` !== `${destination.pathname}${destination.hash}`) {
      window.location.replace(destination);
    }
  }
```

Change `boot()` to:

```js
  function boot() {
    initialiseLanguageMigration();
    initialiseTheme();
    initialiseCopyButton();
    initialiseReveal();
    initialiseEngineStory();
  }
```

In `initialiseCopyButton`, continue reading `document.body.dataset.lang` so generated Chinese copy feedback remains Chinese.

Rebuild so the updated tracked script is copied to `public/assets/js/main.js`, then bump the script URL in `site/templates/home.html` to `main.js?v=2026071004`.

- [ ] **Step 4: Run focused and full checks**

Run:

```bash
node --test site/tests/build-home.test.mjs
node scripts/build-public-site.mjs
node --check public/assets/js/main.js
node scripts/test-public-site.mjs
```

Expected: all commands exit 0. `public/index.html` and `public/zh/index.html` contain no bilingual data attributes.

- [ ] **Step 5: Commit the migration**

```bash
git add site/templates/home.html site/tests/build-home.test.mjs scripts/test-public-site.mjs
git commit -m "fix(site): make language links crawlable" \
  -m "Constraint: Preserve legacy ?lang links without indexing them" \
  -m "Rejected: Runtime document translation | Chinese would not have a stable canonical response" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: node --check public/assets/js/main.js; node scripts/test-public-site.mjs"
```

### Task 5: Style the Real Language Link and Verify Home Visual Parity

**Files:**
- Modify: `site/assets/css/main.css:120-148,339-346,351-414`
- Modify: `site/templates/home.html`
- Modify: `scripts/test-public-site.mjs`
- Create: `output/qa/seo-geo-foundation/desktop-en.png`
- Create: `output/qa/seo-geo-foundation/desktop-zh.png`
- Create: `output/qa/seo-geo-foundation/mobile-zh.png`
- Create: `.omx/state/seo-geo-foundation/ralph-progress.json`

- [ ] **Step 1: Add the failing CSS contract**

Add to `scripts/test-public-site.mjs`:

```js
assert.match(css, /\.language-link\s*\{[^}]*display:\s*inline-flex/);
assert.match(css, /\.language-link:hover/);
assert.match(html, /class="language-link" href="\/zh\/"/);
```

Run `node scripts/test-public-site.mjs` and expect FAIL for the missing `.language-link` rule.

- [ ] **Step 2: Add the minimal shared control style**

Change the utility-control selector in `site/assets/css/main.css`:

```css
.utility-button, .language-link {
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--ink-soft);
  background: transparent;
  border: 0;
  border-radius: 10px;
}
.utility-button:hover, .language-link:hover { color: var(--ink); background: var(--paper-deep); }
.language-link { padding: 0 10px; font-size: 11px; font-weight: 700; letter-spacing: .04em; }
```

At the mobile breakpoint use:

```css
.language-link { padding-inline: 7px; }
```

Do not reintroduce the removed `.language-button` styles.

- [ ] **Step 3: Build and run automated verification**

Run:

```bash
node scripts/build-public-site.mjs
node --test site/tests/localize.test.mjs site/tests/routes.test.mjs site/tests/build-home.test.mjs
node scripts/test-public-site.mjs
node --check public/assets/js/main.js
git diff --check
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 4: Run browser visual verification**

Serve `public/` locally and verify:

```text
1440 x 900: / in English, light and dark
1440 x 900: /zh/ in Chinese, light and dark
390 x 844: /zh/ in Chinese, light and dark
Legacy: /?lang=zh#engine lands on /zh/#engine
Navigation: language links go to exact counterpart URLs
Runtime: no console errors or failed formal resources
```

Capture the three named screenshots. Compare them to the existing approved home baselines with `visual-verdict`. Persist this exact JSON shape in `.omx/state/seo-geo-foundation/ralph-progress.json`:

```json
{
  "scope": "seo-geo-foundation",
  "threshold": 90,
  "threshold_pass": true,
  "latest_verdict": {
    "score": 90,
    "verdict": "pass",
    "category_match": true,
    "differences": [],
    "suggestions": [],
    "reasoning": "The generated locale pages preserve the approved landing-page hierarchy, themes, and engine story."
  },
  "next_actions": []
}
```

Replace `90` and the empty arrays with observed values. If the score is below 90, do not make another edit until the verdict has been recorded; apply one targeted visual correction and rerun the verdict.

- [ ] **Step 5: Record the foundation checkpoint**

Run:

```bash
git status --short
git check-ignore -v public/index.html public/zh/index.html
```

Expected: source files are tracked or staged as intended; both public files remain ignored.

Commit tracked source and tests:

```bash
git add site scripts/build-public-site.mjs scripts/test-public-site.mjs
git commit -m "feat(site): preserve the landing page across static locales" \
  -m "Constraint: Generated public output is deployed separately" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Directive: Run visual-verdict after any home-template or shared-theme change" \
  -m "Tested: static build, Node tests, desktop/mobile browser QA, visual-verdict"
```

## Foundation Completion Gate

Do not start plan 2 until all are true:

- `/` and `/zh/` are complete independent HTML documents.
- The query-string Chinese URL only migrates and is not canonical.
- The exact landing-page layout and engine story pass visual-verdict at 90 or higher.
- No runtime translation code remains.
- Node unit, integration, existing site, syntax, and diff checks pass.
- Generated `public/` remains ignored and no force-add occurred.
