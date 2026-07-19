import assert from "node:assert/strict";
import test from "node:test";

import { facts, macmlxComparisonProfile } from "../content/facts.mjs";
import { pages } from "../content/pages.mjs";
import { project } from "../content/project.mjs";
import { releases } from "../content/releases.mjs";

const factsById = new Map(facts.map((fact) => [fact.id, fact]));
const releasedV070FactIds = [
  "silicon-activity-panel",
  "bottleneck-classifier",
  "silicon-sampling",
  "benchmark-attribution",
  "ocr-recognition",
];
const coldCacheDevelopmentFactIds = [
  "cold-cache-byte-budget",
  "cold-cache-weight-identity",
  "cold-cache-persistent-index",
  "cold-cache-serial-writer",
];

function assertImmutableV070Source(factId) {
  const item = factsById.get(factId);
  assert.ok(item, `missing ${factId}`);
  assert.equal(item.status, "released", `${factId} must be released`);
  assert.equal(item.sinceVersion, "0.7.0", `${factId} must ship in v0.7.0`);
  assert.ok(
    item.sourceUrls.some((url) => url.includes("/releases/tag/v0.7.0") || url.includes("/blob/v0.7.0/")),
    `${factId} must cite immutable v0.7.0 evidence`,
  );
}

test("project and release registries identify v0.7.0 as current", () => {
  assert.deepEqual(
    {
      currentVersion: project.currentVersion,
      releaseDate: project.releaseDate,
      lastVerified: project.lastVerified,
    },
    { currentVersion: "0.7.0", releaseDate: "2026-07-18", lastVerified: "2026-07-19" },
  );
  assert.equal(releases[0].id, "v0-7-0");
  assert.equal(releases[0].version, project.currentVersion);
  assert.equal(releases[0].releaseDate, project.releaseDate);
  assert.equal(releases[0].lastVerified, project.lastVerified);
  assert.ok(releases.some((item) => item.id === "v0-6-2"));
  assert.ok(releases.some((item) => item.id === "v0-5-3"));
});

test("v0.7.0 silicon and OCR capabilities are governed by immutable tagged evidence", () => {
  releasedV070FactIds.forEach(assertImmutableV070Source);
  const shipped = releases[0].shippedFactIds;
  for (const factId of releasedV070FactIds) assert.ok(shipped.includes(factId), `v0.7.0 must ship ${factId}`);
});

test("silicon observability facts keep estimated-versus-measured and availability boundaries", () => {
  for (const locale of ["en", "zh-Hans"]) {
    const panel = factsById.get("silicon-activity-panel")[locale];
    assert.match(`${panel.summary} ${panel.detail}`, locale === "en" ? /sudoless|admin rights/i : /sudo|管理员/);
    assert.match(panel.detail, locale === "en" ? /observability|not a performance guarantee/i : /可观测性|性能保证/);

    const sampling = factsById.get("silicon-sampling")[locale];
    assert.match(sampling.detail, /dlopen/);
    assert.match(sampling.detail, locale === "en" ? /estimate/i : /估算/);
    assert.match(sampling.detail, locale === "en" ? /unavailable/i : /不可用/);

    const classifier = factsById.get("bottleneck-classifier")[locale];
    assert.match(classifier.detail, locale === "en" ? /in-process/i : /进程内/);
    assert.match(classifier.detail, locale === "en" ? /heuristic|not a profiler/i : /启发式|剖析器/);

    const attribution = factsById.get("benchmark-attribution")[locale];
    assert.match(attribution.detail, locale === "en" ? /unavailable/i : /不可用/);
    assert.match(attribution.detail, locale === "en" ? /confidence/i : /置信度/);

    const ocr = factsById.get("ocr-recognition")[locale];
    assert.match(ocr.detail, /dots_ocr/);
    assert.match(ocr.detail, /deepseek-ocr/);
    assert.match(ocr.summary, /GLM-OCR/);
  }
});

test("v0.8 cold-cache work stays development on main, not a released or shipped claim", () => {
  for (const factId of coldCacheDevelopmentFactIds) {
    const item = factsById.get(factId);
    assert.ok(item, `missing ${factId}`);
    assert.equal(item.status, "development", `${factId} must stay in development`);
    assert.equal(item.sinceVersion, "post-0.7.0", `${factId} must be tracked after v0.7.0`);
    assert.ok(
      item.sourceUrls.some((url) => url.includes("/blob/main/")),
      `${factId} must reference post-tag main`,
    );
    assert.match(`${item.en.summary} ${item.en.detail}`, /main|unreleased/i);
    assert.match(item.en.detail, /not part of (?:any|a) tagged release/i);
  }
  assert.deepEqual(releases[0].developmentFactIds, coldCacheDevelopmentFactIds);
  for (const factId of coldCacheDevelopmentFactIds) {
    assert.ok(!releases[0].shippedFactIds.includes(factId), `${factId} must not be shipped`);
  }
  const tiered = factsById.get("tiered-cache");
  assert.match(tiered.en.detail, /does not provide released block sharing or paged KV allocation/);
});

test("model support facts preserve measured and theoretical boundaries", () => {
  const trackG = factsById.get("track-g-tested-models");
  for (const throughput of ["18.2", "80.3", "21.7", "18.7"]) {
    assert.match(trackG.en.detail, new RegExp(throughput.replace(".", "\\.")));
    assert.match(trackG["zh-Hans"].detail, new RegExp(throughput.replace(".", "\\.")));
  }

  const internLM3 = factsById.get("internlm3-theoretical");
  assert.equal(internLM3.status, "released", "shipped code must stay in the released lifecycle");
  assert.equal(internLM3.supportTier, "theoretical", "displayed model support must stay theoretical");
  assert.match(internLM3.en.detail, /tokenizer\.json/);
  assert.match(internLM3.en.detail, /tokenizer\.model/);
  assert.match(internLM3["zh-Hans"].detail, /tokenizer\.json/);
  assert.match(internLM3["zh-Hans"].detail, /tokenizer\.model/);
});

test("API tool-loop row uses product-facing language without changing route coverage", () => {
  const apiPage = pages.find((page) => page.id === "api-compatibility");
  const matrix = apiPage.blocks.find((block) => block.type === "table");
  assert.deepEqual(matrix.rows.en.find((row) => row[0] === "Tool loops"), [
    "Tool loops",
    "OpenAI, Anthropic, and GUI MCP routes",
    "Multi-turn tool routing released in v0.6.0",
  ]);
  assert.deepEqual(matrix.rows["zh-Hans"].find((row) => row[0] === "工具循环"), [
    "工具循环",
    "OpenAI、Anthropic 与 GUI MCP 路由",
    "多轮工具路由于 v0.6.0 发布",
  ]);
});

test("planned work stays separate from released KV-cache quantization", () => {
  for (const factId of ["paged-kv", "adaptive-memory-guard", "sampling-expanded"]) {
    assert.equal(factsById.get(factId)?.status, "planned");
  }
  const sampling = factsById.get("sampling-expanded");
  assert.match(`${sampling.en.summary} ${sampling.en.detail}`, /top-k/);
  assert.match(`${sampling.en.summary} ${sampling.en.detail}`, /min-p/);
  assert.match(`${sampling.en.summary} ${sampling.en.detail}`, /seed/);
  assert.doesNotMatch(`${sampling.en.summary} ${sampling.en.detail}`, /KV quantization/i);
  assert.equal(factsById.get("kv-cache-quantization")?.status, "released");
});

test("comparison profile cites the v0.7 serving, model, and silicon-observability facts", () => {
  for (const locale of ["en", "zh-Hans"]) {
    assert.ok(macmlxComparisonProfile[locale].models.sourceFactIds.includes("track-g-tested-models"));
    assert.ok(macmlxComparisonProfile[locale].models.sourceFactIds.includes("internlm3-theoretical"));
    assert.ok(macmlxComparisonProfile[locale].interfaces.sourceFactIds.includes("structured-output"));
    assert.ok(macmlxComparisonProfile[locale].interfaces.sourceFactIds.includes("integrated-tool-routing"));
    assert.deepEqual(
      macmlxComparisonProfile[locale].focus.sourceFactIds,
      ["continuous-batching", "trie-lcp", "structured-output", "speculative-decoding", "silicon-activity-panel", "bottleneck-classifier"],
    );
  }
  assert.match(macmlxComparisonProfile.en.focus.text, /eligibility-gated continuous batching/);
  assert.match(macmlxComparisonProfile.en.focus.text, /silicon-bottleneck observability/);
});
