import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test, { before } from "node:test";

import { outputFileForPath } from "../lib/routes.mjs";
import { routes } from "../routes.mjs";
import { prepareSite } from "../../scripts/build-public-site.mjs";

const root = new URL("../../", import.meta.url);
let documents;
let generatedDocuments;

before(async () => {
  ({ documents: generatedDocuments } = await prepareSite());
});

test("the build emits all 32 localized HTML documents", async () => {
  documents = new Map();
  for (const route of routes) {
    for (const [locale, path] of Object.entries(route.paths)) {
      await access(new URL(`public/${outputFileForPath(path)}`, root));
      documents.set(`${route.id}:${locale}`, generatedDocuments.get(path));
    }
  }
  assert.equal(documents.size, 32);
});

test("every article is localized, answer-first, semantic, source-linked, and cross-locale", () => {
  for (const route of routes.filter((item) => item.kind === "article")) {
    for (const [locale, path] of Object.entries(route.paths)) {
      const html = documents.get(`${route.id}:${locale}`);
      assert.match(html, new RegExp(`<html lang="${locale === "en" ? "en" : "zh-CN"}">`));
      assert.equal(html.match(/<h1\b/g)?.length, 1, `${route.id}/${locale} needs one h1`);
      assert.match(html, /class="article-answer"/);
      assert.match(html, /class="breadcrumbs"/);
      assert.match(html, /class="article-body"/);
      assert.match(html, /class="content-section sources"/);
      assert.match(html, /class="related-pages"/);
      assert.match(html, /<time datetime="2026-07-19">2026-07-19<\/time>/);
      assert.match(html, /class="site-header"/);
      assert.match(html, /class="site-footer"/);
      assert.match(html, new RegExp(`<link rel="canonical" href="https:\/\/macmlx\\.app${path.replaceAll("/", "\\/")}">`));
      assert.match(html, new RegExp(`href="${route.paths[locale === "en" ? "zh-Hans" : "en"]}" hreflang="${locale === "en" ? "zh-Hans" : "en"}"`));
      assert.doesNotMatch(html, /\{\{|\}\}|\bdata-(?:en|zh)=|only native Swift GUI|16 detected model_type families/i);
      for (const match of html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)) JSON.parse(match[1]);
    }
  }
});

test("generated content exposes the audited facts and exact visible structures", () => {
  const architecture = documents.get("architecture:en");
  const api = documents.get("api-compatibility:en");
  const models = documents.get("models:en");
  const vlm = documents.get("vision-language-models:en");
  const faq = documents.get("faq:en");
  const compare = documents.get("compare:en");
  const release = documents.get("release-v0-8-0:en");
  assert.match(architecture, /separate processes keep separate in-memory engine instances/);
  assert.match(architecture, /data-status="released"/);
  assert.match(architecture, /data-status="planned"/);
  assert.match(architecture, /eligibility-gated continuous batching/i);
  assert.match(architecture, /2\.5(?:&ndash;|–|&#x2013;|-)3\.2×/);
  assert.match(architecture, /Paged KV, block sharing, and CoW/);
  assert.match(architecture, /adaptive memory (?:controller|guard)/i);
  assert.match(api, /\/x\/models/);
  assert.match(api, /Structured output/);
  assert.match(api, /KV-cache quantization/);
  assert.match(api, /Messages API only/);
  assert.match(api, /\/api\/version, \/api\/tags, \/api\/show, \/api\/chat, \/api\/generate/);
  assert.match(models, /Seed-OSS-36B/);
  assert.match(models, /InternLM3-8B/);
  assert.match(models, /tokenizer\.json/);
  assert.match(models, /checkpoint-specific, not family-wide performance guarantees/);
  assert.match(models, /theoretical only/i);
  assert.match(vlm, /14 VLM model_type families/);
  assert.equal(faq.match(/<details>/g)?.length, 8);
  for (const name of ["Ollama", "LM Studio", "oMLX", "Swama", "SwiftLM"]) assert.match(compare, new RegExp(name));
  assert.match(compare, /Dated factual comparison/);
  assert.match(compare, /comparison-limitations/);
  assert.match(release, /current audited baseline/i);
  assert.match(release, /Paged KV, block sharing, and CoW/);
  assert.match(release, /Compatibility and upgrade notes/);
});

test("article illustrations and all root-absolute assets exist", async () => {
  for (const path of [
    "/assets/images/generated/macmlx-shared-core.webp",
    "/assets/images/generated/macmlx-unified-memory.webp",
    "/assets/images/generated/macmlx-inference-pipeline.webp",
  ]) await access(new URL(`public${path}`, root));

  for (const html of documents.values()) {
    for (const match of html.matchAll(/(?:src|href)="(\/assets\/[^"?#]+)/g)) await access(new URL(`public${match[1]}`, root));
  }
});

test("all generated root-local page links resolve inside the route catalog", () => {
  const routePaths = new Set(routes.flatMap((route) => Object.values(route.paths)));
  for (const [key, html] of documents) {
    for (const match of html.matchAll(/href="(\/[^"?#]*\/)(?:[?#][^"]*)?"/g)) {
      if (match[1].startsWith("/assets/")) continue;
      assert.ok(routePaths.has(match[1]), `${key} links to an unknown local page: ${match[1]}`);
    }
  }
});

test("repeated content builds are byte-identical", async () => {
  const beforeBuild = new Map(documents);
  const rebuilt = (await prepareSite()).documents;
  for (const route of routes) for (const [locale, path] of Object.entries(route.paths)) {
    assert.equal(rebuilt.get(path), beforeBuild.get(`${route.id}:${locale}`));
  }
});

test("build validation uses an injectable actual UTC date without changing rendered dates", async () => {
  await assert.doesNotReject(prepareSite({ today: "2026-07-19" }));
  await assert.rejects(prepareSite({ today: "2026-08-25" }), /stale (?:competitor|fact)/);
  assert.match(generatedDocuments.get("/architecture/"), /2026-07-19/);
});
