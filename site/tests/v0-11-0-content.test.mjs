import assert from "node:assert/strict";
import test from "node:test";

import { facts } from "../content/facts.mjs";
import { faqs } from "../content/faqs.mjs";
import { pages } from "../content/pages.mjs";
import { project } from "../content/project.mjs";
import { releases } from "../content/releases.mjs";

const factsById = new Map(facts.map((fact) => [fact.id, fact]));
const current = releases[0];
const v0100 = releases.find((item) => item.id === "v0-10-0");
const releasedV0110FactIds = [
  "responses-api",
  "anthropic-claude-code",
  "reasoning-round-trip",
  "history-boundary-snapshot",
  "foundation-models-checked",
  "vlm-full-path-images",
];
// Facts whose claim the v0.11.0 tag no longer holds. They stay in the release
// that recorded them and cite the last tag where they were true.
const supersededBy = {
  "structured-output-generable-schemas": "structured-output-schemas",
};
const citesTag = (item, tag) => item.sourceUrls.some((url) => url.includes(`/releases/tag/${tag}`) || url.includes(`/blob/${tag}/`));

function both(factId) {
  const item = factsById.get(factId);
  assert.ok(item, `missing ${factId}`);
  return { en: item.en, zh: item["zh-Hans"] };
}

test("project and release registries identify v0.11.0 as current", () => {
  assert.deepEqual(
    { currentVersion: project.currentVersion, releaseDate: project.releaseDate, lastVerified: project.lastVerified },
    { currentVersion: "0.11.0", releaseDate: "2026-11-17", lastVerified: "2026-11-17" },
  );
  assert.equal(current.id, "v0-11-0");
  assert.equal(current.version, project.currentVersion);
  assert.equal(current.releaseDate, project.releaseDate);
  assert.equal(current.lastVerified, project.lastVerified);
  assert.deepEqual([...current.officialSources], [
    "https://github.com/magicnight/mac-mlx/releases/tag/v0.11.0",
    "https://github.com/magicnight/mac-mlx/blob/v0.11.0/CHANGELOG.md",
    "https://github.com/magicnight/mac-mlx/blob/v0.11.0/docs/model-support.md",
  ]);
  for (const id of ["v0-10-0", "v0-9-0", "v0-8-0", "v0-7-0", "v0-6-2", "v0-5-3"]) {
    const item = releases.find((release) => release.id === id);
    assert.ok(item, `${id} must remain a historical record`);
    assert.match(item.en.summary, /^Historically,/, `${id} must read as history`);
    assert.match(item["zh-Hans"].summary, /^作为历史记录，/, `${id} must read as history`);
  }
});

test("v0.11.0 facts ship in v0.11.0 on the current immutable tag", () => {
  for (const factId of releasedV0110FactIds) {
    const item = factsById.get(factId);
    assert.ok(item, `missing ${factId}`);
    assert.equal(item.status, "released", `${factId} must be released`);
    assert.equal(item.sinceVersion, "0.11.0", `${factId} was delivered in v0.11.0`);
    assert.ok(citesTag(item, `v${project.currentVersion}`), `${factId} must cite the v${project.currentVersion} tag`);
    assert.ok(current.shippedFactIds.includes(factId), `v0.11.0 must ship ${factId}`);
    assert.ok(!v0100.shippedFactIds.includes(factId), `${factId} must not be backdated into v0.10.0`);
    assert.ok(item.pageIds.length > 0, `${factId} fronts a current page`);
  }
  assert.deepEqual(current.developmentFactIds, [], "nothing on main is ahead of the v0.11.0 tag");
  for (const item of facts) {
    assert.notEqual(item.status, "development", `${item.id} is still marked as unreleased main work`);
    assert.doesNotMatch(item.sinceVersion, /^post-/, `${item.id} still names a post-tag version`);
  }
});

test("superseded facts stay in the release that recorded them, on that release's tag", () => {
  for (const [oldId, newId] of Object.entries(supersededBy)) {
    const old = factsById.get(oldId);
    assert.deepEqual([...old.pageIds], [], `${oldId} no longer fronts a current page`);
    assert.ok(!current.shippedFactIds.includes(oldId), `v0.11.0 ships ${newId} instead of ${oldId}`);
    assert.ok(!current.limitationFactIds.includes(oldId), `${oldId} is not a v0.11.0 limitation`);
    assert.ok(v0100.shippedFactIds.includes(oldId), `v0.10.0 really did ship ${oldId}`);
    assert.ok(citesTag(old, "v0.10.0"), `${oldId} cites the v0.10.0 tag it describes`);
    assert.ok(!citesTag(old, `v${project.currentVersion}`), `${oldId} must not cite the tag that replaced its claim`);
    const replacement = factsById.get(newId);
    assert.ok(replacement.pageIds.length > 0, `${newId} takes over the page slot`);
    assert.ok(current.shippedFactIds.includes(newId), `v0.11.0 ships ${newId}`);
    assert.ok(!v0100.shippedFactIds.includes(newId), `${newId} must not be backdated into the v0.10.0 record`);
  }

  // The planned end-to-end check is done: the plan stays in the v0.10.0 record, the check ships now.
  const plan = factsById.get("foundation-models-e2e");
  assert.equal(plan.status, "planned");
  assert.deepEqual([...plan.pageIds], [], "the finished plan no longer fronts a current page");
  assert.ok(v0100.plannedFactIds.includes("foundation-models-e2e"));
  assert.ok(!current.plannedFactIds.includes("foundation-models-e2e"));
  assert.ok(current.shippedFactIds.includes("foundation-models-checked"));
});

test("v0.11.0 surfaces state their boundaries", () => {
  for (const factId of ["responses-api", "foundation-models-checked", "vlm-full-path-images", "structured-output-schemas", "mtp-drafter-detection"]) {
    assert.ok(current.limitationFactIds.includes(factId), `${factId} must be a visible v0.11.0 limitation`);
  }

  const responses = both("responses-api");
  assert.match(responses.en.detail, /It is stateless/);
  assert.match(responses.en.detail, /previous_response_id, item_reference items, background, and file ids are a 400/);
  assert.match(responses.en.detail, /requests are served one at a time/);
  assert.match(responses.en.detail, /not the full OpenAI Responses API/);
  assert.match(responses.zh.detail, /而非完整的 OpenAI Responses API/);

  const claudeCode = both("anthropic-claude-code");
  assert.match(claudeCode.en.detail, /budget_tokens is accepted and not enforced/);
  assert.match(claudeCode.en.detail, /not the full Anthropic API/);
  assert.match(claudeCode.zh.detail, /而非完整的 Anthropic API/);

  const reasoning = both("reasoning-round-trip");
  assert.match(reasoning.en.detail, /not encryption/);
  assert.match(reasoning.en.detail, /the Ollama endpoints are out of scope/);
  assert.match(reasoning.zh.detail, /并非加密/);

  const fm = both("foundation-models-checked");
  assert.match(fm.en.summary, /Qwen3\.6-27B-4bit/);
  assert.match(fm.en.summary, /Gemma 4 E4B/);
  assert.match(fm.en.detail, /@Guide\(\.pattern\) is a 400/);
  assert.match(fm.en.detail, /Not checked: images, other models, what an app sees with no server running, and the Ollama endpoints/);
  assert.match(fm.en.detail, /no generation quality is claimed beyond these runs/);
  assert.match(fm.zh.detail, /不对生成质量作任何宣称/);

  const images = both("vlm-full-path-images");
  assert.match(images.en.detail, /the 31B dense and 26B-A4B Gemma 4, E2B, gemma4_unified, qwen3_5_moe, and any other checkpoint load as text/);
  assert.match(images.en.detail, /records nothing, and never speculates/);
  assert.match(images.en.detail, /that is not a claim for other checkpoints/);
  assert.match(images.zh.detail, /不构成对其他检查点的宣称/);

  const schemas = both("structured-output-schemas");
  assert.match(schemas.en.detail, /null, unions, pattern, and multipleOf remain unsupported/);
  assert.match(schemas.en.detail, /a schema shape accepted here is not by itself a verified integration/);
});

test("measured v0.11.0 claims keep their scope", () => {
  const history = both("history-boundary-snapshot");
  for (const figure of ["1,662 of 1,712", "1\\.0 s instead of 2\\.8 s", "1,594 of 1,632", "256 tokens", "512 MiB"]) assert.match(history.en.detail, new RegExp(figure));
  assert.match(history.en.detail, /Those are single measured runs, not a guarantee for other models or prompts/);
  assert.match(history.en.detail, /not paged KV or block sharing/);
  assert.match(history.zh.detail, /并非对其他模型或提示的保证/);
});

test("the v0.11.0 release record states what it does not claim", () => {
  assert.match(current.en.summary, /^The current release/);
  assert.match(current.en.compatibilityNotes, /\/v1\/responses is stateless/);
  assert.match(current.en.compatibilityNotes, /covered two models and no images/);
  assert.match(current.en.compatibilityNotes, /Multi-token-prediction decoding is not part of this release/);
  assert.match(current["zh-Hans"].compatibilityNotes, /多词元预测解码不属于这一版本/);
  assert.deepEqual([...current.plannedFactIds], ["paged-kv", "adaptive-memory-guard", "sampling-expanded"]);

  const page = pages.find((item) => item.id === "release-v0-11-0");
  assert.match(page.en.directAnswer, /^v0\.11\.0 is the current audited baseline/);
  assert.match(page.en.directAnswer, /multi-token-prediction decoding is not part of this release/);

  const apis = faqs.find((item) => item.id === "apis");
  assert.match(apis.en.answer, /^v0\.11\.0 /);
  assert.match(apis.en.answer, /checked end to end on two models, without images/);
});
