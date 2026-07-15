import assert from "node:assert/strict";
import { access } from "node:fs/promises";
import test from "node:test";

test("the content hub has a semantic block renderer", async () => {
  await assert.doesNotReject(
    access(new URL("../lib/content-renderer.mjs", import.meta.url)),
    "expected the semantic content renderer",
  );
});

const fact = {
  id: "core",
  status: "released",
  sinceVersion: "0.5.3",
  lastVerified: "2026-07-10",
  sourceUrls: ["https://example.test/source"],
  en: { title: "Swift core", summary: "Runs <locally>.", detail: "One process & one engine." },
  "zh-Hans": { title: "Swift 核心", summary: "本地运行。", detail: "一个进程与一个引擎。" },
};

const context = {
  locale: "en",
  factsById: { core: fact },
  faqsById: { python: { en: { question: "Python <required>?", answer: "Not on the default path." }, "zh-Hans": { question: "需要 Python 吗？", answer: "默认路径不需要。" } } },
  competitorsById: { tool: { name: "Tool & Co", verifiedVersion: "v1", snapshotDate: "2026-07-01", lastVerified: "2026-07-10", officialSources: ["https://example.test/tool"], en: { summary: "A tool.", limitations: ["A limitation."], dimensions: { platform: "macOS", runtime: "Python", models: "Models", interfaces: "API", focus: "Serving" } } } },
  releasesById: {},
  pagesById: { architecture: { paths: { en: "/architecture/" }, en: { title: "Architecture", description: "How it works." } } },
  macmlxComparisonProfile: { en: Object.fromEntries(["platform", "runtime", "models", "interfaces", "focus"].map((key) => [key, { text: `Registry ${key}`, sourceFactIds: ["core"] }])) },
};

test("semantic blocks render facts, code, tables, FAQs, comparisons, sources, and related links", async () => {
  const { renderContentBlocks } = await import("../lib/content-renderer.mjs");
  assert.equal(typeof renderContentBlocks, "function");
  const html = renderContentBlocks([
    { type: "paragraph", text: { en: "Answer first.", "zh-Hans": "先给答案。" } },
    { type: "facts", heading: { en: "Facts", "zh-Hans": "事实" }, factIds: ["core"] },
    { type: "code", heading: { en: "Example", "zh-Hans": "示例" }, language: "sh", code: "curl '<unsafe>'" },
    { type: "table", heading: { en: "Matrix", "zh-Hans": "矩阵" }, caption: { en: "Compatibility", "zh-Hans": "兼容性" }, headers: { en: ["Surface", "Boundary"], "zh-Hans": ["接口", "边界"] }, rows: { en: [["Chat", "Released"]], "zh-Hans": [["聊天", "已发布"]] } },
    { type: "faq", heading: { en: "FAQ", "zh-Hans": "常见问题" }, faqIds: ["python"] },
    { type: "comparison", heading: { en: "Comparison", "zh-Hans": "对比" }, competitorIds: ["tool"] },
    { type: "sources", heading: { en: "Sources", "zh-Hans": "来源" }, factIds: ["core"] },
    { type: "related", heading: { en: "Related", "zh-Hans": "相关" }, relatedIds: ["architecture"] },
  ], context);
  assert.match(html, /class="fact-card" data-status="released"/);
  assert.match(html, />Released</);
  assert.match(html, /<pre class="article-code"><code class="language-sh">curl &#39;&lt;unsafe&gt;&#39;<\/code><\/pre>/);
  assert.match(html, /<table class="article-table">/);
  assert.match(html, /<caption>Compatibility<\/caption>/);
  assert.match(html, /<details><summary>Python &lt;required&gt;\?<\/summary>/);
  assert.match(html, /<table class="comparison-table">/);
  assert.match(html, /Tool &amp; Co/);
  assert.match(html, /<section class="content-section sources">/);
  assert.match(html, /<nav class="related-pages"/);
  assert.doesNotMatch(html, /<unsafe>|Python <required>|Tool & Co/);
});

test("source blocks reject every free-form URL bypass", async () => {
  const { renderContentBlocks } = await import("../lib/content-renderer.mjs");
  for (const url of ["javascript:alert(1)", "http://example.com/source", "https://example.com/source"]) {
    assert.throws(() => renderContentBlocks([{ type: "sources", heading: { en: "Sources", "zh-Hans": "来源" }, sources: [{ label: "Free form", url }] }], context), /free-form sources are not allowed/);
  }
});

test("renderer emits meaningful illustrations and rejects unknown or broken blocks", async () => {
  const { renderContentBlocks } = await import("../lib/content-renderer.mjs");
  const html = renderContentBlocks([{ type: "illustration", src: "/assets/images/generated/macmlx-shared-core.webp", width: 2048, height: 1152, alt: { en: "Shared core diagram", "zh-Hans": "共享核心图" }, caption: { en: "One core.", "zh-Hans": "一个核心。" } }], context);
  assert.match(html, /alt="Shared core diagram"/);
  assert.match(html, /width="2048" height="1152"/);
  assert.throws(() => renderContentBlocks([{ type: "unknown" }], context), /unsupported content block/);
  assert.throws(() => renderContentBlocks([{ type: "facts", heading: { en: "Facts" }, factIds: ["missing"] }], context), /unknown fact/);
});

test("comparison rendering requires registered macMLX cells and shows competitor limitations", async () => {
  const { renderContentBlocks } = await import("../lib/content-renderer.mjs");
  const ungoverned = structuredClone(context);
  delete ungoverned.macmlxComparisonProfile;
  assert.throws(
    () => renderContentBlocks([{ type: "comparison", heading: { en: "Comparison", "zh-Hans": "对比" }, competitorIds: ["tool"] }], ungoverned),
    /macMLX comparison profile/,
  );
  const governed = structuredClone(context);
  governed.macmlxComparisonProfile = {
    en: Object.fromEntries(["platform", "runtime", "models", "interfaces", "focus"].map((key) => [key, { text: `Registry ${key}`, sourceFactIds: ["core"] }])),
  };
  governed.competitorsById.tool.en.limitations = ["A sourced limitation."];
  const html = renderContentBlocks([{ type: "comparison", heading: { en: "Comparison", "zh-Hans": "对比" }, competitorIds: ["tool"] }], governed);
  assert.match(html, /Registry runtime/);
  assert.match(html, /class="comparison-limitations"/);
  assert.match(html, /A sourced limitation/);
});

test("release rendering includes localized compatibility and upgrade notes", async () => {
  const { renderContentBlocks } = await import("../lib/content-renderer.mjs");
  const releaseContext = structuredClone(context);
  releaseContext.releasesById.current = {
    releaseDate: "2026-07-08",
    shippedFactIds: ["core"], limitationFactIds: ["core"], developmentFactIds: ["core"], plannedFactIds: ["core"],
    en: { title: "Release", summary: "Summary", compatibilityNotes: "Compatibility note", upgradeNotes: "Upgrade note" },
  };
  const html = renderContentBlocks([{ type: "release", heading: { en: "Release", "zh-Hans": "版本" }, releaseIds: ["current"] }], releaseContext);
  assert.match(html, /Compatibility note/);
  assert.match(html, /Upgrade note/);
});

test("release rendering omits empty lifecycle groups", async () => {
  const { renderContentBlocks } = await import("../lib/content-renderer.mjs");
  const releaseContext = structuredClone(context);
  releaseContext.releasesById.release = {
    releaseDate: "2026-07-08",
    shippedFactIds: ["core"], limitationFactIds: ["core"], developmentFactIds: ["core"], plannedFactIds: ["core"],
    en: { title: "Release", summary: "Summary", compatibilityNotes: "Compatibility note", upgradeNotes: "Upgrade note" },
  };
  releaseContext.releasesById.release = {
    ...releaseContext.releasesById.release,
    developmentFactIds: [],
  };
  const releaseHTML = renderContentBlocks([
    { type: "release", heading: { en: "Release", "zh-Hans": "版本" }, releaseIds: ["release"] },
  ], releaseContext);
  assert.match(releaseHTML, />Shipped</);
  assert.match(releaseHTML, />Planned</);
  assert.doesNotMatch(releaseHTML, /Development after the tag/);
});
