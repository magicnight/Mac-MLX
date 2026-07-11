import assert from "node:assert/strict";
import { access, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { copiedAssetPaths } from "../site/content/assets.mjs";
import { competitors } from "../site/content/competitors.mjs";
import { facts, macmlxComparisonProfile } from "../site/content/facts.mjs";
import { faqs } from "../site/content/faqs.mjs";
import { pages } from "../site/content/pages.mjs";
import { project } from "../site/content/project.mjs";
import { releases } from "../site/content/releases.mjs";
import { acquireBuildLock, replaceGeneratedOutput } from "../site/lib/build-output.mjs";
import { validateContentHub, validateReleaseIdentity } from "../site/lib/content.mjs";
import { renderContentBlocks } from "../site/lib/content-renderer.mjs";
import { renderNotFound, renderRobots, renderSitemap } from "../site/lib/discovery.mjs";
import { escapeHTML, renderTokens } from "../site/lib/localize.mjs";
import { renderLLMSIndexes, renderMarkdownDocuments } from "../site/lib/markdown.mjs";
import { renderMetadata } from "../site/lib/metadata.mjs";
import { siteURL, validateProject } from "../site/lib/project-schema.mjs";
import { breadcrumbItems } from "../site/lib/breadcrumbs.mjs";
import { outputFileForPath, validateCanonicalPath } from "../site/lib/routes.mjs";
import { renderHomeTemplate, renderSocialImage } from "../site/lib/site-rendering.mjs";
import { socialCardCaptures, socialCaptureInstructions, socialCardSourceDigest, validateSocialPNG } from "../site/lib/social-card.mjs";
import { homeRoutes, routes } from "../site/routes.mjs";

const repositoryRoot = fileURLToPath(new URL("../", import.meta.url));
const sourceAssets = join(repositoryRoot, "site/assets");
const homeTemplateFile = join(repositoryRoot, "site/templates/home.html");
const articleTemplateFile = join(repositoryRoot, "site/templates/article.html");
const socialImageTemplateFile = join(repositoryRoot, "site/templates/og-image.svg");
const cloudflarePolicyDirectory = join(repositoryRoot, "site/cloudflare");
const outputDirectory = join(repositoryRoot, "public");
const locales = Object.freeze(["en", "zh-Hans"]);

const factsById = Object.freeze(Object.fromEntries(facts.map((item) => [item.id, item])));
const competitorsById = Object.freeze(Object.fromEntries(competitors.map((item) => [item.id, item])));
const faqsById = Object.freeze(Object.fromEntries(faqs.map((item) => [item.id, item])));
const releasesById = Object.freeze(Object.fromEntries(releases.map((item) => [item.id, item])));
const pagesById = Object.freeze(Object.fromEntries(pages.map((item) => [item.id, item])));

function canonicalURL(path) {
  return siteURL(project, path);
}

function assertNonEmptyString(value, label) {
  assert.equal(typeof value, "string", `${label} must be a string`);
  assert.ok(value.trim(), `${label} must not be empty`);
}

function validateSourceData(homeTemplate, articleTemplate, today) {
  validateProject(project);
  validateReleaseIdentity(project, releases);
  validateContentHub({ facts, competitors, faqs, releases, pages, macmlxComparisonProfile }, { today, maxAgeDays: 45 });
  assert.equal(new URL(project.origin).protocol, "https:", "project origin must use HTTPS");
  assert.equal(new URL(project.origin).pathname, "/", "project origin must not contain a path");
  for (const field of ["repositoryURL", "downloadURL", "currentVersion", "releaseDate", "lastVerified", "licenseURL", "operatingSystem"]) assertNonEmptyString(project[field], `project.${field}`);
  assert.match(project.currentVersion, /^\d+\.\d+\.\d+$/, "currentVersion must be semantic");
  assert.match(project.releaseDate, /^\d{4}-\d{2}-\d{2}$/, "releaseDate must be ISO formatted");
  assert.match(project.lastVerified, /^\d{4}-\d{2}-\d{2}$/, "lastVerified must be ISO formatted");
  assert.deepEqual(Object.keys(project.locales), locales, "project locales must match route locales");

  const seenPaths = new Set();
  for (const route of routes) {
    assertNonEmptyString(route.id, "route.id");
    assert.deepEqual(Object.keys(route.paths), locales, `${route.id} must have locale parity`);
    for (const path of Object.values(route.paths)) {
      validateCanonicalPath(path);
      assert.ok(!seenPaths.has(path), `duplicate canonical path: ${path}`);
      seenPaths.add(path);
    }
  }
  assert.equal(routes.length, 13, "the complete hub must have 13 route IDs");
  assert.equal(seenPaths.size, 26, "the complete hub must have 26 localized paths");

  const englishCount = homeTemplate.match(/\bdata-en=/g)?.length ?? 0;
  const chineseCount = homeTemplate.match(/\bdata-zh=/g)?.length ?? 0;
  assert.equal(englishCount, chineseCount, "home template must have locale parity");
  assert.equal(englishCount, 97, "home template must preserve the 97 approved bilingual nodes");
  for (const marker of ["<!--article-head-->", "<!--site-header-->", "<!--breadcrumbs-->", "<!--article-content-->", "<!--site-footer-->"]) assert.equal(articleTemplate.split(marker).length, 2, `article template must contain one ${marker}`);
}

function validateHTML(html, route, locale) {
  const path = route.paths[locale];
  const metadata = project.locales[locale];
  assert.match(html, new RegExp(`<html lang="${metadata.htmlLang}">`));
  assert.match(html, new RegExp(`<link rel="canonical" href="${canonicalURL(path).replaceAll("/", "\\/")}">`));
  assert.equal(html.match(/<h1\b/g)?.length, 1, `${route.id}/${locale} must have one h1`);
  assert.doesNotMatch(html, /\{\{|\}\}/, `${route.id}/${locale} has unresolved tokens`);
  assert.doesNotMatch(html, /\bdata-(?:en|zh|lang-option)=/, `${route.id}/${locale} has runtime translation scaffolding`);
  assert.doesNotMatch(html, /(?:src|href)="assets\//, `${route.id}/${locale} has relative asset references`);
  assert.match(html, new RegExp(`hreflang="en" href="${canonicalURL(route.paths.en).replaceAll("/", "\\/")}">`));
  assert.match(html, new RegExp(`hreflang="zh-Hans" href="${canonicalURL(route.paths["zh-Hans"]).replaceAll("/", "\\/")}">`));
  for (const match of html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)) JSON.parse(match[1]);
}

function renderHome(template, locale) {
  const route = routes[0];
  const metadataHTML = renderMetadata({ project, route, locale, faqs });
  const rendered = renderHomeTemplate({ template, project, routes: homeRoutes, locale, metadataHTML });
  validateHTML(rendered, route, locale);
  return rendered;
}

function labels(locale) {
  return locale === "en"
    ? { home: "Home", architecture: "Architecture", api: "API", models: "Models", compare: "Compare", faq: "FAQ", releases: "Releases", download: "Download", source: "Source", skip: "Skip to content", verified: "Last verified:", navigation: "Primary navigation", theme: "Switch color theme", language: "查看简体中文版", breadcrumb: "Breadcrumb" }
    : { home: "首页", architecture: "架构", api: "API", models: "模型", compare: "对比", faq: "常见问题", releases: "版本", download: "下载", source: "源码", skip: "跳到正文", verified: "最后核验：", navigation: "主导航", theme: "切换颜色主题", language: "View the English version", breadcrumb: "面包屑" };
}

function localizedPath(locale, path) {
  return locale === "en" ? path : `/zh${path}`;
}

function renderHeader(locale, counterpartPath) {
  const copy = labels(locale);
  const home = locale === "en" ? "/" : "/zh/";
  const counterpartLocale = locale === "en" ? "zh-Hans" : "en";
  const languageLabel = locale === "en" ? "中" : "EN";
  return `<header class="site-header"><div class="nav-shell"><a class="wordmark" href="${escapeHTML(home)}" aria-label="macMLX ${escapeHTML(copy.home)}"><img class="brand-mark" src="/assets/brand/macmlx-mark.svg" alt=""><span>macMLX</span></a><nav class="nav-links" aria-label="${escapeHTML(copy.navigation)}"><a href="${escapeHTML(localizedPath(locale, "/architecture/"))}">${escapeHTML(copy.architecture)}</a><a href="${escapeHTML(localizedPath(locale, "/api-compatibility/"))}">${escapeHTML(copy.api)}</a><a href="${escapeHTML(localizedPath(locale, "/models/"))}">${escapeHTML(copy.models)}</a><a href="${escapeHTML(localizedPath(locale, "/compare/"))}">${escapeHTML(copy.compare)}</a><a href="${escapeHTML(localizedPath(locale, "/faq/"))}">${escapeHTML(copy.faq)}</a><a href="${escapeHTML(localizedPath(locale, "/releases/"))}">${escapeHTML(copy.releases)}</a></nav><div class="nav-actions"><button class="utility-button" id="theme-toggle" type="button" aria-label="${escapeHTML(copy.theme)}" title="${escapeHTML(copy.theme)}"><svg class="sun-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.5"></circle><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg><svg class="moon-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 15.2A8.6 8.6 0 0 1 8.8 4a8.6 8.6 0 1 0 11.2 11.2Z"></path></svg></button><a class="language-button" href="${escapeHTML(counterpartPath)}" hreflang="${escapeHTML(counterpartLocale)}" aria-label="${escapeHTML(copy.language)}">${escapeHTML(languageLabel)}</a><a class="nav-download" href="${escapeHTML(project.downloadURL)}">${escapeHTML(copy.download)}</a></div></div></header>`;
}

function renderFooter(locale) {
  const copy = labels(locale);
  const home = locale === "en" ? "/" : "/zh/";
  return `<footer class="site-footer"><div class="page-shell footer-layout"><div><a class="wordmark footer-wordmark" href="${escapeHTML(home)}"><img class="brand-mark" src="/assets/brand/macmlx-mark.svg" alt=""><span>macMLX</span></a><p>${escapeHTML(locale === "en" ? "Native Swift inference for Apple Silicon." : "面向 Apple 芯片的原生 Swift 推理。")}</p></div><div class="footer-links"><a href="${escapeHTML(localizedPath(locale, "/architecture/"))}">${escapeHTML(copy.architecture)}</a><a href="${escapeHTML(localizedPath(locale, "/api-compatibility/"))}">${escapeHTML(copy.api)}</a><a href="${escapeHTML(localizedPath(locale, "/models/"))}">${escapeHTML(copy.models)}</a><a href="${escapeHTML(localizedPath(locale, "/compare/"))}">${escapeHTML(copy.compare)}</a><a href="${escapeHTML(localizedPath(locale, "/faq/"))}">${escapeHTML(copy.faq)}</a><a href="${escapeHTML(localizedPath(locale, "/releases/"))}">${escapeHTML(copy.releases)}</a><a href="${escapeHTML(project.repositoryURL)}">${escapeHTML(copy.source)}</a></div></div><div class="page-shell footer-bottom"><span>© 2026 macMLX contributors</span><span>${escapeHTML(locale === "en" ? "Local by design. Open by choice." : "为本地而生，因开放而自由。")}</span></div></footer>`;
}

function renderBreadcrumbs(page, locale) {
  const copy = labels(locale);
  const hierarchy = breadcrumbItems(page, locale);
  const items = hierarchy.map((item, index) => {
    if (index === hierarchy.length - 1) return `<span aria-current="page">${escapeHTML(item.name)}</span>`;
    return `<a href="${escapeHTML(item.path)}">${escapeHTML(item.name)}</a><span aria-hidden="true">/</span>`;
  });
  return `<nav class="breadcrumbs" aria-label="${escapeHTML(copy.breadcrumb)}">${items.join("")}</nav>`;
}

function renderArticle(template, page, locale) {
  const copy = labels(locale);
  const counterpartLocale = locale === "en" ? "zh-Hans" : "en";
  const content = renderContentBlocks(page.blocks, { locale, factsById, competitorsById, faqsById, releasesById, pagesById, macmlxComparisonProfile });
  const trusted = template
    .replace("<!--article-head-->", renderMetadata({ project, route: page, page, locale, faqs }))
    .replace("<!--site-header-->", renderHeader(locale, page.paths[counterpartLocale]))
    .replace("<!--breadcrumbs-->", renderBreadcrumbs(page, locale))
    .replace("<!--article-content-->", content)
    .replace("<!--site-footer-->", renderFooter(locale));
  const rendered = `${renderTokens(trusted, { htmlLang: project.locales[locale].htmlLang, skipLabel: copy.skip, eyebrow: page[locale].eyebrow, title: page[locale].title, directAnswer: page[locale].directAnswer, verifiedLabel: copy.verified, lastVerified: page.lastVerified }).trimEnd()}\n`;
  validateHTML(rendered, page, locale);
  return rendered;
}

async function writeText(root, relativePath, content) {
  const destination = join(root, relativePath);
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, content, "utf8");
}

async function validateAssets() {
  for (const relativePath of copiedAssetPaths) await access(join(sourceAssets, relativePath));
}

async function copyAssets(destinationDirectory) {
  for (const relativePath of copiedAssetPaths) {
    const destination = join(destinationDirectory, relativePath);
    await mkdir(dirname(destination), { recursive: true });
    await copyFile(join(sourceAssets, relativePath), destination);
  }
}

export async function validateSocialAssets(assetRoot = sourceAssets) {
  const missing = [];
  for (const capture of socialCardCaptures) {
    const relativePath = capture.source.replace(/^site\/assets\//, "");
    try {
      const source = join(assetRoot, relativePath);
      await access(source);
      validateSocialPNG(await readFile(source), capture.source, {
        expectedSourceDigest: socialCardSourceDigest({ project, locale: capture.locale }),
      });
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      missing.push(capture.source);
    }
  }
  if (missing.length > 0) {
    throw new Error(`Social PNG capture required before build. Open site/social-card.html?locale=en and site/social-card.html?locale=zh at 1200x630. Save reviewed captures to ${missing.join(" and ")}. ${socialCaptureInstructions}`);
  }
}

export async function copySocialAssets(root) {
  for (const capture of socialCardCaptures) {
    const relativeSource = capture.source.replace(/^site\/assets\//, "");
    const relativeOutput = capture.output.replace(/^public\//, "");
    const destination = join(root, relativeOutput);
    await mkdir(dirname(destination), { recursive: true });
    await copyFile(join(sourceAssets, relativeSource), destination);
  }
}

async function validateGeneratedOutput(root, documents, markdownDocuments, discoveryFiles, deploymentFiles) {
  for (const [path, html] of documents) {
    assert.equal(await readFile(join(root, outputFileForPath(path)), "utf8"), html);
    for (const match of html.matchAll(/(?:src|href)="(\/assets\/[^"?#]+)/g)) await access(join(root, match[1]));
  }
  for (const [relativePath, content] of [...markdownDocuments, ...discoveryFiles, ...deploymentFiles]) {
    assert.equal(await readFile(join(root, relativePath), "utf8"), content);
  }
  await access(join(root, "assets/og-image.svg"));
  for (const capture of socialCardCaptures) await access(join(root, capture.output.replace(/^public\//, "")));
  const sitemap = await readFile(join(root, "sitemap.xml"), "utf8");
  assert.equal(sitemap.match(/<url>/g)?.length, 26);
  assert.doesNotMatch(sitemap, /\?lang=/);
}

function utcDate(clock) {
  return clock().toISOString().slice(0, 10);
}

export async function prepareSite({ today, clock = () => new Date() } = {}) {
  const [homeTemplate, articleTemplate, socialImageTemplate, headersPolicy, redirectsPolicy] = await Promise.all([
    readFile(homeTemplateFile, "utf8"),
    readFile(articleTemplateFile, "utf8"),
    readFile(socialImageTemplateFile, "utf8"),
    readFile(join(cloudflarePolicyDirectory, "_headers"), "utf8"),
    readFile(join(cloudflarePolicyDirectory, "_redirects"), "utf8"),
  ]);

  // Registry, route, template, and asset validation intentionally completes
  // before the lock or staged output causes any filesystem write.
  validateSourceData(homeTemplate, articleTemplate, today ?? utcDate(clock));
  await validateAssets();

  const documents = new Map();
  for (const locale of locales) documents.set(homeRoutes[locale], renderHome(homeTemplate, locale));
  for (const page of pages) for (const locale of locales) documents.set(page.paths[locale], renderArticle(articleTemplate, page, locale));

  const context = { project, routes, pages, facts, competitors, faqs, releases };
  const markdownDocuments = renderMarkdownDocuments(context);
  const discoveryFiles = new Map([
    ...renderLLMSIndexes(context),
    ["robots.txt", renderRobots(project)],
    ["sitemap.xml", renderSitemap({ project, routes })],
    ["404.html", renderNotFound({ project, locale: "en" })],
    ["zh/404.html", renderNotFound({ project, locale: "zh-Hans" })],
  ]);
  const deploymentFiles = new Map([
    ["_headers", headersPolicy],
    ["_redirects", redirectsPolicy],
  ]);

  const socialImage = renderSocialImage({ template: socialImageTemplate, project });
  return { documents, markdownDocuments, discoveryFiles, deploymentFiles, socialImage };
}

export async function buildSite() {
  const { documents, markdownDocuments, discoveryFiles, deploymentFiles, socialImage } = await prepareSite();
  await validateSocialAssets();

  const releaseBuildLock = await acquireBuildLock(repositoryRoot);
  try {
    const stagedDirectory = await mkdtemp(join(repositoryRoot, ".public-build-"));
    try {
      await copyAssets(join(stagedDirectory, "assets"));
      await copySocialAssets(stagedDirectory);
      await writeText(stagedDirectory, "assets/og-image.svg", socialImage);
      for (const [path, html] of documents) await writeText(stagedDirectory, outputFileForPath(path), html);
      for (const [relativePath, content] of markdownDocuments) await writeText(stagedDirectory, relativePath, content);
      for (const [relativePath, content] of discoveryFiles) await writeText(stagedDirectory, relativePath, content);
      for (const [relativePath, content] of deploymentFiles) await writeText(stagedDirectory, relativePath, content);
      await validateGeneratedOutput(stagedDirectory, documents, markdownDocuments, discoveryFiles, deploymentFiles);
      await replaceGeneratedOutput({ workspaceDirectory: repositoryRoot, outputDirectory, stagedDirectory });
    } finally {
      await rm(stagedDirectory, { recursive: true, force: true });
    }
  } finally {
    await releaseBuildLock();
  }

  console.log(`Built ${documents.size} HTML pages, ${markdownDocuments.size} Markdown pages, ${discoveryFiles.size} discovery files, and ${deploymentFiles.size} deployment policy files in public/`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await buildSite();
