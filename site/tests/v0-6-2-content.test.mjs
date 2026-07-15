import assert from "node:assert/strict";
import test from "node:test";

import { facts, macmlxComparisonProfile } from "../content/facts.mjs";
import { pages } from "../content/pages.mjs";
import { project } from "../content/project.mjs";
import { releases } from "../content/releases.mjs";

const factsById = new Map(facts.map((fact) => [fact.id, fact]));
const releasedV062FactIds = [
  "integrated-tool-routing",
  "trie-lcp",
  "continuous-batching",
  "fixed-prefill-throttle",
  "structured-output",
  "speculative-decoding",
  "api-compat-pack",
  "kv-cache-quantization",
  "chat-template-overrides",
  "track-g-tested-models",
  "internlm3-theoretical",
];

function assertImmutableV062Source(factId) {
  const item = factsById.get(factId);
  assert.ok(item, `missing ${factId}`);
  assert.equal(item.status, "released", `${factId} must be released`);
  assert.ok(
    item.sourceUrls.some((url) => url.includes("/releases/tag/v0.6.2") || url.includes("/blob/v0.6.2/")),
    `${factId} must cite immutable v0.6.2 evidence`,
  );
}

test("project and release registries identify v0.6.2 as current", () => {
  assert.deepEqual(
    {
      currentVersion: project.currentVersion,
      releaseDate: project.releaseDate,
      lastVerified: project.lastVerified,
    },
    { currentVersion: "0.6.2", releaseDate: "2026-07-11", lastVerified: "2026-07-15" },
  );
  assert.equal(releases[0].id, "v0-6-2");
  assert.equal(releases[0].version, project.currentVersion);
  assert.equal(releases[0].releaseDate, project.releaseDate);
  assert.equal(releases[0].lastVerified, project.lastVerified);
  assert.ok(releases.some((item) => item.id === "v0-5-3"));
});

test("v0.6 capabilities are governed by immutable tagged evidence", () => {
  releasedV062FactIds.forEach(assertImmutableV062Source);
  assertImmutableV062Source("hf-cache-discovery");
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

test("comparison profile cites the v0.6 serving and model facts", () => {
  for (const locale of ["en", "zh-Hans"]) {
    assert.ok(macmlxComparisonProfile[locale].models.sourceFactIds.includes("track-g-tested-models"));
    assert.ok(macmlxComparisonProfile[locale].models.sourceFactIds.includes("internlm3-theoretical"));
    assert.ok(macmlxComparisonProfile[locale].interfaces.sourceFactIds.includes("structured-output"));
    assert.ok(macmlxComparisonProfile[locale].interfaces.sourceFactIds.includes("integrated-tool-routing"));
    assert.deepEqual(
      macmlxComparisonProfile[locale].focus.sourceFactIds,
      ["continuous-batching", "trie-lcp", "structured-output", "speculative-decoding"],
    );
  }
  assert.match(macmlxComparisonProfile.en.focus.text, /eligibility-gated continuous batching/);
});
