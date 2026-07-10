# SEO/GEO Discovery and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete per-page metadata, structured data, Markdown alternates, LLM indexes, sitemap, robots, PNG social previews, internal-link validation, and final browser evidence for the bilingual site.

**Architecture:** Keep search and answer-engine outputs as deterministic derivatives of the same route and content registries used for visible HTML. One metadata module owns canonical, hreflang, social tags, and JSON-LD; Markdown and discovery modules own text and XML surfaces. A local crawler verifies that every internal URL resolves before browser visual and interaction QA.

**Tech Stack:** Node.js built-ins, static HTML/XML/Markdown/text, JSON-LD, CSS-rendered PNG social cards captured through the in-app browser, `node:test`, in-app browser QA.

---

## Sequence and Dependency

This is plan 3 of 3. Start only after the foundation and content-hub plans pass their completion gates.

## File Map and Boundaries

- Create `site/lib/metadata.mjs` — canonical, reciprocal hreflang, social metadata, and JSON-LD.
- Create `site/lib/markdown.mjs` — page Markdown and LLM index rendering.
- Create `site/lib/discovery.mjs` — sitemap and robots generation.
- Create `site/tests/metadata.test.mjs` — per-page metadata and visible-data parity tests.
- Create `site/tests/markdown.test.mjs` — Markdown/HTML fact and locale parity tests.
- Create `site/tests/discovery.test.mjs` — sitemap and crawler-policy tests.
- Create `site/social-card.html` — deterministic locale social-card renderer.
- Create `site/README.md` — release refresh, source review, build, validation, and deployment boundaries.
- Create `site/assets/social/og-en.png` and `site/assets/social/og-zh.png` — approved 1200 x 630 PNG assets.
- Create `scripts/crawl-public-site.mjs` — local route and asset crawler.
- Create `site/tests/crawl.test.mjs` — full generated-output link contract.
- Modify `scripts/build-public-site.mjs` — emit metadata, Markdown, LLM indexes, sitemap, robots, and copy approved social assets.
- Modify `scripts/test-public-site.mjs` — site-wide uniqueness, language, social, and discovery checks.
- Create `output/qa/seo-geo-final/` — final screenshots and browser audit.
- Create `.omx/state/seo-geo-final/ralph-progress.json` — final visual verdict and source bindings.

### Task 1: Centralize Canonical, Social, and JSON-LD Metadata

**Files:**
- Create: `site/tests/metadata.test.mjs`
- Create: `site/lib/metadata.mjs`
- Modify: `scripts/build-public-site.mjs`

- [ ] **Step 1: Write failing metadata tests**

Create `site/tests/metadata.test.mjs`:

```js
import assert from "node:assert/strict";
import test from "node:test";
import { project } from "../content/project.mjs";
import { pages } from "../content/pages.mjs";
import { renderHead } from "../lib/metadata.mjs";

test("renderHead emits reciprocal canonical and hreflang metadata", () => {
  const route = pages.find((item) => item.id === "architecture");
  const head = renderHead({ route, locale: "zh-Hans", project });
  assert.match(head, /<title>macMLX 如何在 Apple 芯片上运行大模型 — macMLX<\/title>/);
  assert.match(head, /rel="canonical" href="https:\/\/macmlx\.app\/zh\/architecture\/"/);
  assert.match(head, /hreflang="en" href="https:\/\/macmlx\.app\/architecture\/"/);
  assert.match(head, /hreflang="zh-Hans" href="https:\/\/macmlx\.app\/zh\/architecture\/"/);
  assert.match(head, /hreflang="x-default" href="https:\/\/macmlx\.app\/architecture\/"/);
  assert.match(head, /og:locale" content="zh_CN"/);
  assert.match(head, /twitter:card" content="summary_large_image"/);
  assert.match(head, /type="text\/markdown"/);
});

test("article JSON-LD is valid and does not invent ratings", () => {
  const route = pages.find((item) => item.id === "compare-omlx");
  const head = renderHead({ route, locale: "en", project });
  const jsonText = head.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)[1];
  const data = JSON.parse(jsonText);
  assert.equal(data["@graph"][0]["@type"], "TechArticle");
  assert.equal(data["@graph"][1]["@type"], "BreadcrumbList");
  assert.doesNotMatch(jsonText, /aggregateRating|review/);
});

test("metadata values stay locale-specific and route-unique", () => {
  const titles = new Set();
  for (const route of pages) {
    for (const locale of project.locales) {
      const title = route.metadata[locale].title;
      assert.ok(!titles.has(`${locale}:${title}`), `duplicate title: ${locale}:${title}`);
      titles.add(`${locale}:${title}`);
      assert.ok(route.metadata[locale].description.length >= (locale === "en" ? 70 : 35));
      assert.ok(route.metadata[locale].description.length <= 180);
    }
  }
});
```

- [ ] **Step 2: Verify the red state**

Run `node --test site/tests/metadata.test.mjs`.

Expected: FAIL because `site/lib/metadata.mjs` does not exist.

- [ ] **Step 3: Implement the metadata renderer**

Create `site/lib/metadata.mjs`:

```js
import { escapeHTML } from "./localize.mjs";
import { canonicalURL } from "./routes.mjs";

const copyLocale = (locale) => locale === "en" ? "en" : "zh";

function breadcrumbs(route, locale, project) {
  const home = locale === "en" ? { name: "Home", path: "/" } : { name: "首页", path: "/zh/" };
  const segments = route.paths[locale].split("/").filter(Boolean);
  const withoutLocale = locale === "en" ? segments : segments.slice(1);
  const items = [home];
  if (withoutLocale.length > 1) {
    const parentPath = locale === "en" ? `/${withoutLocale[0]}/` : `/zh/${withoutLocale[0]}/`;
    const parentName = locale === "en"
      ? withoutLocale[0].replaceAll("-", " ")
      : ({ models: "模型指南", compare: "产品对比", releases: "版本" }[withoutLocale[0]] || withoutLocale[0]);
    items.push({ name: parentName, path: parentPath });
  }
  items.push({ name: route.title[locale], path: route.paths[locale] });
  return items.map((item, index) => ({
    "@type": "ListItem",
    position: index + 1,
    name: item.name,
    item: canonicalURL(project.origin, item.path),
  }));
}

function articleGraph(route, locale, project) {
  const canonical = canonicalURL(project.origin, route.paths[locale]);
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "TechArticle",
        "@id": `${canonical}#article`,
        headline: route.title[locale],
        description: route.metadata[locale].description,
        inLanguage: project.htmlLanguages[locale],
        dateModified: route.lastVerified,
        mainEntityOfPage: canonical,
        about: { "@id": `${project.origin}/#software` },
      },
      {
        "@type": "BreadcrumbList",
        "@id": `${canonical}#breadcrumbs`,
        itemListElement: breadcrumbs(route, locale, project),
      },
    ],
  };
}

function homeGraph(route, locale, project) {
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebSite",
        "@id": `${project.origin}/#website`,
        name: project.name,
        url: project.origin,
        inLanguage: ["en", "zh-CN"],
      },
      {
        "@type": "SoftwareApplication",
        "@id": `${project.origin}/#software`,
        name: project.name,
        url: project.origin,
        codeRepository: project.repository,
        downloadUrl: project.downloadURL,
        operatingSystem: "macOS 14.0+ on Apple Silicon",
        applicationCategory: "DeveloperApplication",
        softwareVersion: project.version,
        dateModified: project.lastVerified,
        license: project.licenseURL,
        programmingLanguage: "Swift",
        runtimePlatform: "Apple MLX on Apple Silicon",
        image: `${project.origin}/assets/social/${locale === "en" ? "og-en.png" : "og-zh.png"}`,
        offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        description: route.metadata[locale].description,
      },
    ],
  };
}

export function renderHead({ route, locale, project }) {
  const metadata = route.metadata[locale];
  const canonical = canonicalURL(project.origin, route.paths[locale]);
  const englishURL = canonicalURL(project.origin, route.paths.en);
  const chineseURL = canonicalURL(project.origin, route.paths["zh-Hans"]);
  const socialImage = `${project.origin}/assets/social/${locale === "en" ? "og-en.png" : "og-zh.png"}`;
  const socialAlt = locale === "en" ? "macMLX on Apple Silicon" : "Apple 芯片上的 macMLX";
  const markdownPath = `/content/${locale === "en" ? "en" : "zh"}/${route.id}.md`;
  const jsonLd = route.kind === "article"
    ? articleGraph(route, locale, project)
    : homeGraph(route, locale, project);
  const tags = [
    `<title>${escapeHTML(metadata.title)}</title>`,
    `<meta name="description" content="${escapeHTML(metadata.description)}">`,
    `<link rel="canonical" href="${canonical}">`,
    `<link rel="alternate" hreflang="en" href="${englishURL}">`,
    `<link rel="alternate" hreflang="zh-Hans" href="${chineseURL}">`,
    `<link rel="alternate" hreflang="x-default" href="${englishURL}">`,
    `<link rel="alternate" type="text/markdown" href="${markdownPath}" title="${locale === "en" ? "Markdown version" : "Markdown 版本"}">`,
    `<meta property="og:type" content="${route.kind === "home" ? "website" : "article"}">`,
    `<meta property="og:site_name" content="macMLX">`,
    `<meta property="og:locale" content="${locale === "en" ? "en_US" : "zh_CN"}">`,
    `<meta property="og:url" content="${canonical}">`,
    `<meta property="og:title" content="${escapeHTML(metadata.title)}">`,
    `<meta property="og:description" content="${escapeHTML(metadata.socialDescription)}">`,
    `<meta property="og:image" content="${socialImage}">`,
    `<meta property="og:image:type" content="image/png">`,
    `<meta property="og:image:width" content="1200">`,
    `<meta property="og:image:height" content="630">`,
    `<meta property="og:image:alt" content="${socialAlt}">`,
    `<meta name="twitter:card" content="summary_large_image">`,
    `<meta name="twitter:title" content="${escapeHTML(metadata.title)}">`,
    `<meta name="twitter:description" content="${escapeHTML(metadata.socialDescription)}">`,
    `<meta name="twitter:image" content="${socialImage}">`,
    `<meta name="twitter:image:alt" content="${socialAlt}">`,
  ];
  tags.push(`<script type="application/ld+json">${JSON.stringify(jsonLd, null, 2)}</script>`);
  return tags.join("\n  ");
}

export function localeCopyKey(locale) {
  return copyLocale(locale);
}
```

- [ ] **Step 4: Make both home and article builders use `renderHead`**

Remove `temporaryArticleHead` and `softwareEntities` from `scripts/build-public-site.mjs`. Import `renderHead` and pass `renderHead({ route, locale, project })` as the article template's trusted `head` token.

In `site/templates/home.html`, replace the existing title, description, canonical, hreflang, Open Graph, Twitter, and JSON-LD elements with one raw token:

```html
{{{head}}}
```

Keep the theme-color, color-scheme, favicon, stylesheets, and scripts outside that token. In `homeTokens`, remove `title`, `description`, `socialDescription`, `canonical`, `englishURL`, `chineseURL`, `socialImage`, `socialImageAlt`, and `jsonLd`; add:

```js
head: renderHead({ route, locale, project }),
```

Change the home trusted-token set to:

```js
new Set(["head", "languageLink"])
```

Rebuild and assert that every generated document has exactly one `<title>`, description, canonical, and JSON-LD script.

Do not add `aggregateRating`, `review`, `FAQPage`, or comparison scores.

- [ ] **Step 5: Run tests and commit metadata ownership**

Run:

```bash
node --test site/tests/metadata.test.mjs site/tests/build-home.test.mjs site/tests/build-content.test.mjs
node scripts/build-public-site.mjs
```

Expected: all tests pass and every page contains one canonical.

Commit:

```bash
git add site/lib/metadata.mjs site/tests/metadata.test.mjs scripts/build-public-site.mjs
git commit -m "feat(site): make page identity explicit to crawlers" \
  -m "Constraint: Structured data must mirror visible facts and cannot invent ratings" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Tested: metadata, home-build, and content-build Node tests"
```

### Task 2: Generate Markdown Alternates and LLM Indexes

**Files:**
- Create: `site/tests/markdown.test.mjs`
- Create: `site/lib/markdown.mjs`
- Modify: `scripts/build-public-site.mjs`
- Generate: `public/content/en/*.md`
- Generate: `public/content/zh/*.md`
- Generate: `public/llms.txt`, `public/llms-full.txt`, `public/zh/llms.txt`, `public/zh/llms-full.txt`

- [ ] **Step 1: Write failing Markdown parity tests**

Create `site/tests/markdown.test.mjs`:

```js
import assert from "node:assert/strict";
import test from "node:test";
import { facts } from "../content/facts.mjs";
import { pages } from "../content/pages.mjs";
import { renderLLMSIndex, renderPageMarkdown } from "../lib/markdown.mjs";

test("page Markdown carries the direct answer, status, and canonical source", () => {
  const page = pages.find((item) => item.id === "architecture");
  const markdown = renderPageMarkdown({ page, locale: "en", facts });
  assert.match(markdown, /^# How macMLX runs LLMs on Apple Silicon/m);
  assert.match(markdown, /Canonical: https:\/\/macmlx\.app\/architecture\//);
  assert.match(markdown, /Status: Released/);
  assert.match(markdown, /Swift-native in-process inference/);
  assert.match(markdown, /Last verified: 2026-07-10/);
});

test("LLM indexes are complete and language-specific", () => {
  const english = renderLLMSIndex({ locale: "en", pages, facts, full: false });
  const chinese = renderLLMSIndex({ locale: "zh-Hans", pages, facts, full: true });
  assert.match(english, /Latest release: v0\.5\.3/);
  assert.match(english, /https:\/\/macmlx\.app\/architecture\//);
  assert.match(chinese, /最新版本：v0\.5\.3/);
  assert.match(chinese, /Swift 原生进程内推理/);
  assert.doesNotMatch(chinese, /^# How macMLX/m);
});
```

Run `node --test site/tests/markdown.test.mjs`. Expected: FAIL because `site/lib/markdown.mjs` is missing.

- [ ] **Step 2: Implement Markdown rendering from registries**

Create `site/lib/markdown.mjs`:

```js
import { project } from "../content/project.mjs";

const status = {
  en: { released: "Released", development: "In development", planned: "Planned" },
  zh: { released: "已发布", development: "开发中", planned: "规划中" },
};

const copyKey = (locale) => locale === "en" ? "en" : "zh";

function canonical(page, locale) {
  return new URL(page.paths[locale], project.origin).href;
}

export function renderPageMarkdown({ page, locale, facts }) {
  const copy = copyKey(locale);
  const selectedFacts = (page.factIds || []).map((id) => facts.find((fact) => fact.id === id));
  const title = page.title?.[locale] || page.metadata[locale].title;
  const directAnswer = page.directAnswer?.[locale] || page.metadata[locale].description;
  const lines = [
    `# ${title}`,
    "",
    locale === "en" ? `Canonical: ${canonical(page, locale)}` : `规范网址：${canonical(page, locale)}`,
    locale === "en" ? `Last verified: ${page.lastVerified}` : `最后核验：${page.lastVerified}`,
    "",
    directAnswer,
    "",
  ];
  if (selectedFacts.length) {
    lines.push(locale === "en" ? "## Verified facts" : "## 已核验事实", "");
    for (const fact of selectedFacts) {
      lines.push(
        `### ${fact[copy].title}`,
        "",
        `${locale === "en" ? "Status" : "状态"}: ${status[copy][fact.status]}`,
        `${locale === "en" ? "Since" : "起始版本"}: ${fact.sinceVersion}`,
        fact[copy].summary,
        `${locale === "en" ? "Sources" : "来源"}: ${fact.sourceUrls.join(" · ")}`,
        "",
      );
    }
  }
  return `${lines.join("\n").trim()}\n`;
}

export function renderLLMSIndex({ locale, pages, facts, full }) {
  const copy = copyKey(locale);
  const lines = locale === "en"
    ? ["# macMLX", "", "> Native Swift inference for Apple Silicon.", "", `Latest release: v${project.version}`, `Last verified: ${project.lastVerified}`, ""]
    : ["# macMLX", "", "> 面向 Apple 芯片的原生 Swift 推理。", "", `最新版本：v${project.version}`, `最后核验：${project.lastVerified}`, ""];
  lines.push(locale === "en" ? "## Pages" : "## 页面", "");
  for (const page of pages) {
    const title = page.title?.[locale] || page.metadata[locale].title;
    const directAnswer = page.directAnswer?.[locale] || page.metadata[locale].description;
    lines.push(`- [${title}](${canonical(page, locale)}): ${directAnswer}`);
  }
  if (full) {
    lines.push("", locale === "en" ? "## Fact registry" : "## 事实注册表", "");
    for (const fact of facts) {
      lines.push(`### ${fact[copy].title}`, `${locale === "en" ? "Status" : "状态"}: ${status[copy][fact.status]}`, fact[copy].summary, `Source: ${fact.sourceUrls.join(" · ")}`, "");
    }
  }
  return `${lines.join("\n").trim()}\n`;
}
```

- [ ] **Step 3: Emit all Markdown and LLM files in the builder**

After HTML generation, loop through every route and locale. Write page Markdown to:

```js
const markdownLocale = locale === "en" ? "en" : "zh";
const markdownPath = resolve(repositoryRoot, "public/content", markdownLocale, `${route.id}.md`);
```

Write indexes exactly:

```text
public/llms.txt          locale=en, full=false
public/llms-full.txt     locale=en, full=true
public/zh/llms.txt       locale=zh-Hans, full=false
public/zh/llms-full.txt  locale=zh-Hans, full=true
```

Pass `pages: routes` so the home page is included in the index. For home-page Markdown, use its metadata description as `directAnswer` and omit `factIds`.

- [ ] **Step 4: Run parity tests and commit**

Run:

```bash
node --test site/tests/markdown.test.mjs
node scripts/build-public-site.mjs
rg -n "v0\.5\.3|规划中|Planned" public/llms*.txt public/zh/llms*.txt public/content
```

Expected: tests pass and both languages expose released and planned boundaries.

Commit:

```bash
git add site/lib/markdown.mjs site/tests/markdown.test.mjs scripts/build-public-site.mjs
git commit -m "feat(site): expose citation-ready text alternates" \
  -m "Constraint: LLM-oriented files are derivatives, not ranking guarantees" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Tested: Markdown Node tests and generated fact/status scan"
```

### Task 3: Generate Sitemap and Maintainable Crawler Policy

**Files:**
- Create: `site/tests/discovery.test.mjs`
- Create: `site/lib/discovery.mjs`
- Modify: `scripts/build-public-site.mjs`
- Generate: `public/sitemap.xml`
- Generate: `public/robots.txt`

- [ ] **Step 1: Write failing discovery tests**

Create `site/tests/discovery.test.mjs`:

```js
import assert from "node:assert/strict";
import test from "node:test";
import { project } from "../content/project.mjs";
import { routes } from "../routes.mjs";
import { renderRobots, renderSitemap } from "../lib/discovery.mjs";

test("sitemap contains every canonical once with reciprocal alternates", () => {
  const xml = renderSitemap({ project, routes });
  assert.equal((xml.match(/<url>/g) ?? []).length, 26);
  assert.equal((xml.match(/<loc>https:\/\/macmlx\.app\//g) ?? []).length, 26);
  assert.equal((xml.match(/hreflang="en"/g) ?? []).length, 26);
  assert.equal((xml.match(/hreflang="zh-Hans"/g) ?? []).length, 26);
  assert.doesNotMatch(xml, /\?lang=zh|<changefreq>|<priority>/);
});

test("robots is small, allows search discovery, and declares one sitemap", () => {
  const robots = renderRobots(project.origin);
  assert.match(robots, /User-agent: \*/);
  assert.match(robots, /User-agent: OAI-SearchBot/);
  assert.match(robots, /User-agent: GPTBot/);
  assert.equal((robots.match(/Sitemap:/g) ?? []).length, 1);
  assert.doesNotMatch(robots, /Claude-Web|cohere-training-data-crawler|meta-externalagent/);
});
```

Run `node --test site/tests/discovery.test.mjs`. Expected: FAIL because `site/lib/discovery.mjs` is missing.

- [ ] **Step 2: Implement XML and robots renderers**

Create `site/lib/discovery.mjs`:

```js
import { escapeHTML } from "./localize.mjs";

export function renderSitemap({ project, routes }) {
  const urls = [];
  for (const route of routes) {
    const english = new URL(route.paths.en, project.origin).href;
    const chinese = new URL(route.paths["zh-Hans"], project.origin).href;
    for (const locale of project.locales) {
      const loc = new URL(route.paths[locale], project.origin).href;
      const lastmod = route.lastVerified || project.lastVerified;
      urls.push(`  <url>\n    <loc>${escapeHTML(loc)}</loc>\n    <lastmod>${lastmod}</lastmod>\n    <xhtml:link rel="alternate" hreflang="en" href="${escapeHTML(english)}"/>\n    <xhtml:link rel="alternate" hreflang="zh-Hans" href="${escapeHTML(chinese)}"/>\n    <xhtml:link rel="alternate" hreflang="x-default" href="${escapeHTML(english)}"/>\n  </url>`);
    }
  }
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">\n${urls.join("\n")}\n</urlset>\n`;
}

export function renderRobots(origin) {
  return `# macMLX crawler policy\n\nUser-agent: *\nAllow: /\n\n# ChatGPT search discovery\nUser-agent: OAI-SearchBot\nAllow: /\n\n# OpenAI model-training crawler; managed independently from search\nUser-agent: GPTBot\nAllow: /\n\nSitemap: ${origin}/sitemap.xml\n`;
}
```

- [ ] **Step 3: Emit and validate discovery files**

Write both outputs from `scripts/build-public-site.mjs`, then run:

```bash
node --test site/tests/discovery.test.mjs
node scripts/build-public-site.mjs
xmllint --noout public/sitemap.xml
```

Expected: tests pass and XML parses without errors.

- [ ] **Step 4: Commit discovery generation**

```bash
git add site/lib/discovery.mjs site/tests/discovery.test.mjs scripts/build-public-site.mjs
git commit -m "feat(site): publish a complete language-aware discovery graph" \
  -m "Constraint: Crawler policy distinguishes search from training without a maintenance-heavy bot list" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: discovery Node tests and xmllint"
```

### Task 4: Produce Locale PNG Social Cards

**Files:**
- Create: `site/social-card.html`
- Create: `site/assets/social/og-en.png`
- Create: `site/assets/social/og-zh.png`
- Modify: `scripts/build-public-site.mjs`
- Modify: `site/tests/metadata.test.mjs`
- Create: `output/qa/seo-geo-final/social-en.png`
- Create: `output/qa/seo-geo-final/social-zh.png`

- [ ] **Step 1: Add the failing asset contract**

Append to `site/tests/metadata.test.mjs`:

```js
import { stat } from "node:fs/promises";

test("approved locale social images exist", async () => {
  for (const name of ["og-en.png", "og-zh.png"]) {
    const metadata = await stat(new URL(`../assets/social/${name}`, import.meta.url));
    assert.ok(metadata.size > 20_000, `${name} should be a real PNG asset`);
  }
});
```

Run `node --test site/tests/metadata.test.mjs`. Expected: FAIL with `ENOENT`.

- [ ] **Step 2: Create the exact HTML social-card source**

Create `site/social-card.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>macMLX social card source</title>
  <style>
    * { box-sizing: border-box; }
    html, body { width: 1200px; height: 630px; margin: 0; overflow: hidden; }
    body { color: #131613; background: #f3f1ea; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Arial, sans-serif; }
    .card { position: relative; width: 1200px; height: 630px; padding: 68px 76px; overflow: hidden; }
    .grid { position: absolute; inset: 0; opacity: .45; background-image: linear-gradient(rgba(19,22,19,.12) 1px, transparent 1px), linear-gradient(90deg, rgba(19,22,19,.12) 1px, transparent 1px); background-size: 70px 70px; mask-image: linear-gradient(to bottom right, #000, transparent 78%); }
    header, section { position: relative; z-index: 1; }
    header { display: flex; align-items: center; gap: 16px; font-size: 24px; }
    header > span:last-child { margin-left: auto; padding: 9px 14px; border: 1px solid rgba(19,22,19,.28); border-radius: 999px; font: 13px/1.2 "SFMono-Regular", monospace; }
    .mark { width: 46px; height: 46px; display: grid; place-items: center; color: #9ac6ff; background: #131613; border-radius: 13px; font: 800 22px/1 "SFMono-Regular", monospace; }
    section { margin-top: 72px; }
    section > p { margin: 0 0 22px; color: #0868e8; font: 13px/1.2 "SFMono-Regular", monospace; letter-spacing: .12em; }
    h1 { max-width: 980px; margin: 0; font-size: 78px; line-height: .94; letter-spacing: -.06em; }
    h1 em { color: #0868e8; font-family: Iowan Old Style, Baskerville, serif; font-weight: 400; }
    small { display: block; margin-top: 34px; color: #515750; font-size: 22px; }
    .status { position: absolute; right: 78px; bottom: 72px; width: 12px; height: 12px; border-radius: 50%; background: #27a85f; box-shadow: 0 0 0 8px rgba(39,168,95,.14); }
  </style>
</head>
<body>
  <main class="card" data-locale="en">
    <div class="grid"></div>
    <header><span class="mark">M</span><strong>macMLX</strong><span>v0.5.3</span></header>
    <section>
      <p>APPLE SILICON · SWIFT · MLX</p>
      <h1 data-en="One native engine.<br><em>App · CLI · API</em>" data-zh="一个原生引擎。<br><em>应用 · CLI · API</em>">One native engine.<br><em>App · CLI · API</em></h1>
      <small data-en="Local language and vision models on your Mac" data-zh="在你的 Mac 上运行本地语言与视觉模型">Local language and vision models on your Mac</small>
    </section>
    <span class="status" aria-hidden="true"></span>
  </main>
  <script>
    const language = new URLSearchParams(window.location.search).get("lang") === "zh" ? "zh" : "en";
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.querySelector(".card").dataset.locale = language;
    document.querySelectorAll("[data-en][data-zh]").forEach((node) => {
      node.innerHTML = node.dataset[language];
    });
  </script>
</body>
</html>
```

This card is a build-asset source, not an indexable site route.

- [ ] **Step 3: Capture exact PNG assets with the in-app browser**

Serve the repository locally, open `site/social-card.html?lang=en` and `?lang=zh`, set viewport to 1200 x 630, and capture the viewport for each language. Save to:

```text
site/assets/social/og-en.png
site/assets/social/og-zh.png
```

Reset the temporary viewport after capture.

- [ ] **Step 4: Run visual-verdict before accepting the assets**

Compare both cards to the approved landing-page screenshot with category hint `premium technical-product social preview`. Require 90 or higher. Reject cards with clipped text, unreadable Chinese, excess empty space, theme mismatch, or browser UI. Copy accepted screenshots to the two QA paths.

- [ ] **Step 5: Verify format, dimensions, copy, and build staging**

Run:

```bash
file site/assets/social/*.png
sips -g pixelWidth -g pixelHeight site/assets/social/*.png
node --test site/tests/metadata.test.mjs
```

Expected: both files are PNG, exactly 1200 x 630, and the metadata test passes.

Update the builder's asset-copy step so `site/assets/social/` is copied to `public/assets/social/` without recompression.

- [ ] **Step 6: Commit source and approved assets**

```bash
git add site/social-card.html site/assets/social site/tests/metadata.test.mjs scripts/build-public-site.mjs
git commit -m "feat(site): make social previews broadly interoperable" \
  -m "Constraint: Use crawlable PNG rather than SVG-only social metadata" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: 1200x630 dimension checks, metadata tests, visual-verdict"
```

### Task 5: Crawl Every Generated Internal URL

**Files:**
- Create: `scripts/crawl-public-site.mjs`
- Create: `site/tests/crawl.test.mjs`
- Modify: `scripts/test-public-site.mjs`

- [ ] **Step 1: Write the failing crawl test**

Create `site/tests/crawl.test.mjs`:

```js
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";
import { crawlPublicSite } from "../../scripts/crawl-public-site.mjs";
import { buildSite } from "../../scripts/build-public-site.mjs";

async function textOutputHashes() {
  const root = new URL("../../public/", import.meta.url);
  const files = [];
  async function visit(directory, prefix = "") {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = new URL(`${prefix}${entry.name}${entry.isDirectory() ? "/" : ""}`, root);
      if (entry.isDirectory()) await visit(path, `${prefix}${entry.name}/`);
      else if (/\.(?:html|md|txt|xml)$/.test(entry.name)) files.push(`${prefix}${entry.name}`);
    }
  }
  await visit(root);
  const hashes = {};
  for (const file of files.sort()) {
    const bytes = await readFile(new URL(file, root));
    hashes[file] = createHash("sha256").update(bytes).digest("hex");
  }
  return hashes;
}

test("all generated HTML routes and internal assets resolve", async () => {
  const result = await crawlPublicSite();
  assert.equal(result.missing.length, 0, result.missing.join("\n"));
  assert.equal(result.htmlFiles.length, 26);
  assert.ok(result.checkedLinks > 100);
});

test("two unchanged builds produce byte-identical text outputs", async () => {
  await buildSite();
  const first = await textOutputHashes();
  await buildSite();
  const second = await textOutputHashes();
  assert.deepEqual(second, first);
});
```

Run `node --test site/tests/crawl.test.mjs`. Expected: FAIL because the crawler script is missing.

- [ ] **Step 2: Implement the local output crawler**

Create `scripts/crawl-public-site.mjs`:

```js
import { readdir, readFile, stat } from "node:fs/promises";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../public");

async function filesUnder(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await filesUnder(path));
    else files.push(path);
  }
  return files;
}

function internalReferences(html) {
  return [...html.matchAll(/\b(?:href|src)="([^"]+)"/g)]
    .map((match) => match[1])
    .filter((value) => value.startsWith("/") && !value.startsWith("//"));
}

function referenceFile(reference) {
  const pathname = new URL(reference, "https://macmlx.app").pathname;
  if (pathname.endsWith("/")) return resolve(root, `.${pathname}`, "index.html");
  return resolve(root, `.${pathname}`);
}

export async function crawlPublicSite() {
  const files = await filesUnder(root);
  const htmlFiles = files.filter((path) => extname(path) === ".html");
  const missing = [];
  let checkedLinks = 0;
  for (const htmlFile of htmlFiles) {
    const html = await readFile(htmlFile, "utf8");
    for (const reference of internalReferences(html)) {
      checkedLinks += 1;
      const target = referenceFile(reference);
      try {
        const metadata = await stat(target);
        if (!metadata.isFile()) missing.push(`${htmlFile}: ${reference}`);
      } catch {
        missing.push(`${htmlFile}: ${reference}`);
      }
    }
  }
  return { htmlFiles, checkedLinks, missing };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const result = await crawlPublicSite();
  if (result.missing.length) {
    console.error(result.missing.join("\n"));
    process.exitCode = 1;
  } else {
    console.log(`crawl passed (${result.htmlFiles.length} pages, ${result.checkedLinks} internal references)`);
  }
}
```

- [ ] **Step 3: Add site-wide metadata and language checks**

Extend `scripts/test-public-site.mjs` to enumerate every generated HTML file instead of inspecting only the home page. For each document assert:

```text
one title
one description
one canonical
three hreflang links
one h1
matching html lang and body language
no unresolved tokens
no ?lang=zh canonical or navigation URL
existing PNG social image
parseable JSON-LD
unique title, description, and canonical across the same locale
```

Keep the existing engine-story, theme, reduced-motion, asset, and no-JavaScript assertions scoped to the English home template/output.

- [ ] **Step 4: Run crawler and complete automated suite**

Run:

```bash
node scripts/build-public-site.mjs
node --test site/tests/*.test.mjs
node scripts/crawl-public-site.mjs
node scripts/test-public-site.mjs
node --check public/assets/js/main.js
xmllint --noout public/sitemap.xml
git diff --check
```

Expected: 26 pages, more than 100 checked internal references, zero missing items, and every command exits 0.

- [ ] **Step 5: Document the exact release-maintenance workflow**

Create `site/README.md`:

````markdown
# macMLX Website Source

`site/` is the tracked source for the separately deployed static output in `public/`.

## Build

```bash
node scripts/build-public-site.mjs
```

The build performs no network access. It fails when routes, translations, facts, sources, verification dates, or content blocks are invalid.

## Verify

```bash
node --test site/tests/*.test.mjs
node scripts/crawl-public-site.mjs
node scripts/test-public-site.mjs
node --check public/assets/js/main.js
xmllint --noout public/sitemap.xml
```

## Release refresh

1. Update `site/content/project.mjs` with the released version and real release date.
2. Update `site/content/releases.mjs` and link the GitHub release plus changelog.
3. Reclassify affected entries in `site/content/facts.mjs` as released, development, or planned.
4. Recheck only affected competitor entries against official documentation or repositories; record the compared snapshot and verification date.
5. Update a verification date only when the associated facts were actually checked.
6. Build, inspect the generated output, run the complete verification suite, and repeat browser visual QA for changed templates or shared styles.

## Source policy

Product and competitor facts require official documentation, official release notes, or the official repository. Do not use community comments, unsourced comparison posts, search snippets, or inferred internals as factual claims. Label analysis as analysis.

## Deployment boundary

The repository build does not submit Search Console, Bing Webmaster Tools, or IndexNow changes. Those are separately authorized post-deployment operations.
````

- [ ] **Step 6: Commit the final local integrity checks**

```bash
git add scripts/crawl-public-site.mjs scripts/test-public-site.mjs site/tests/crawl.test.mjs site/README.md
git commit -m "test(site): prevent broken discovery surfaces" \
  -m "Constraint: Generated routes, metadata, assets, and language graphs must agree before deployment" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: full Node suite, local crawler, xmllint, JavaScript syntax"
```

### Task 6: Final Browser QA, Visual Verdict, and Completion Review

**Files:**
- Create: `output/qa/seo-geo-final/home-en-light.png`
- Create: `output/qa/seo-geo-final/home-zh-dark.png`
- Create: `output/qa/seo-geo-final/architecture-en-light.png`
- Create: `output/qa/seo-geo-final/api-zh-dark.png`
- Create: `output/qa/seo-geo-final/compare-mobile-zh.png`
- Create: `output/qa/seo-geo-final/browser-audit.json`
- Create: `.omx/state/seo-geo-final/ralph-progress.json`

- [ ] **Step 1: Run representative browser routes**

Test:

```text
1440 x 900 / English light
1440 x 900 /zh/ Chinese dark
1440 x 900 /architecture/ English light
1440 x 900 /zh/api-compatibility/ Chinese dark
390 x 844 /zh/compare/lm-studio/ Chinese light
390 x 844 /models/choosing-a-model/ English dark
```

For every route confirm exact counterpart language navigation, breadcrumbs, direct answer, verification date, related links, no horizontal overflow, no console error, and no failed formal resource.

- [ ] **Step 2: Verify migration, no-JavaScript, reduced motion, and missing optional images**

Confirm:

```text
/?lang=zh#engine -> /zh/#engine
/?lang=en#engine -> /#engine
article content and language links remain complete with JavaScript disabled
home engine story becomes complete static flow with JavaScript disabled
reduced motion shows all content without transition dependence
missing an optional illustration does not hide text, status, sources, or related links
```

Restore source and generated hashes after every forced failure test.

- [ ] **Step 3: Capture screenshots and run final visual-verdict**

Capture the five named screenshots. Compare home pages to existing approved baselines and article pages to the approved content-hub category. Require at least 90 for both the preserved home and new editorial system.

Persist `.omx/state/seo-geo-final/ralph-progress.json` with the exact JSON returned by `visual-verdict`, plus `scope`, `threshold`, `threshold_pass`, current source SHA-256 values, every retained screenshot path and SHA-256 value, and `next_actions`. Use only observed values from this run. Do not claim completion when reasoning is empty, a source hash differs, `next_actions` is non-empty, or either visual score is below 90.

- [ ] **Step 4: Request final code review**

Use `superpowers:requesting-code-review`. Review against the approved design and all three plans. Any critical or important finding returns to a failing-test-first fix cycle; rerun the relevant visual verdict after visual changes.

- [ ] **Step 5: Run fresh verification immediately before completion**

Run:

```bash
node scripts/build-public-site.mjs
node --test site/tests/*.test.mjs
node scripts/crawl-public-site.mjs
node scripts/test-public-site.mjs
node --check public/assets/js/main.js
xmllint --noout public/sitemap.xml
file public/assets/social/*.png
sips -g pixelWidth -g pixelHeight public/assets/social/*.png
git diff --check
git status --short
```

Verify browser-audit and state hashes match current source and screenshot files. Expected: all commands exit 0, visual scores are at least 90, review has zero critical/important findings, and `next_actions` is empty.

- [ ] **Step 6: Report deployment boundary honestly**

The final report must list:

```text
changed source files
generated public route count
English/Chinese parity evidence
metadata and crawler files generated
automated and browser evidence
visual-verdict scores
competitor verification date
remaining deployment-only actions
```

Do not say Search Console, Bing Webmaster Tools, or IndexNow was configured. Offer those as separately authorized post-deployment actions.

## Final Completion Gate

- 26 canonical HTML pages and their Markdown alternates exist.
- Titles, descriptions, canonicals, and hreflang graphs are unique and reciprocal.
- Sitemap and robots parse and match the route graph.
- Social targets are 1200 x 630 PNG files.
- LLM indexes match current release and status boundaries.
- Local crawler reports zero missing resources.
- Home and content pages pass desktop/mobile, English/Chinese, light/dark, no-JavaScript, and reduced-motion checks.
- Final visual-verdict scores are at least 90.
- Final review has zero critical or important findings.
- External search-engine submissions remain unperformed unless separately authorized.
