import assert from "node:assert/strict";
import test from "node:test";

import { facts } from "../content/facts.mjs";
import { faqs } from "../content/faqs.mjs";
import { pages } from "../content/pages.mjs";
import { project } from "../content/project.mjs";
import { releases } from "../content/releases.mjs";

const factsById = new Map(facts.map((fact) => [fact.id, fact]));
const current = releases[0];
const v090 = releases.find((item) => item.id === "v0-9-0");
const releasedV0100FactIds = [
  "structured-output-generable-schemas",
  "usage-reporting",
  "unload-releases-memory",
  "aux-models-in-pool",
  "rerank-mlx-rerankers",
  "chat-template-native",
  "mlx-fork-0-32-3",
  "audio-app-cache",
  "dmg-committed-lockfiles",
];
// Facts whose claim the v0.10.0 tag no longer holds. They stay in the releases
// that shipped them and cite the last tag where they were true.
const supersededBy = {
  "rerank-cross-encoder": "rerank-mlx-rerankers",
  "upstream-correctness-fixes": "mlx-fork-0-32-3",
  "chat-template-overrides": "chat-template-native",
};
const citesTag = (item, tag) => item.sourceUrls.some((url) => url.includes(`/releases/tag/${tag}`) || url.includes(`/blob/${tag}/`));

function both(factId) {
  const item = factsById.get(factId);
  assert.ok(item, `missing ${factId}`);
  return { en: item.en, zh: item["zh-Hans"] };
}

test("project and release registries identify v0.10.0 as current", () => {
  assert.deepEqual(
    { currentVersion: project.currentVersion, releaseDate: project.releaseDate, lastVerified: project.lastVerified },
    { currentVersion: "0.10.0", releaseDate: "2026-10-29", lastVerified: "2026-10-29" },
  );
  assert.equal(current.id, "v0-10-0");
  assert.equal(current.version, project.currentVersion);
  assert.equal(current.releaseDate, project.releaseDate);
  assert.equal(current.lastVerified, project.lastVerified);
  assert.deepEqual([...current.officialSources], [
    "https://github.com/magicnight/mac-mlx/releases/tag/v0.10.0",
    "https://github.com/magicnight/mac-mlx/blob/v0.10.0/CHANGELOG.md",
    "https://github.com/magicnight/mac-mlx/blob/v0.10.0/docs/model-support.md",
  ]);
  for (const id of ["v0-9-0", "v0-8-0", "v0-7-0", "v0-6-2", "v0-5-3"]) {
    const item = releases.find((release) => release.id === id);
    assert.ok(item, `${id} must remain a historical record`);
    assert.match(item.en.summary, /^Historically,/, `${id} must read as history`);
    assert.match(item["zh-Hans"].summary, /^作为历史记录，/, `${id} must read as history`);
  }
  assert.match(v090.en.upgradeNotes, /For this historical release, review the v0\.9\.0 tagged changelog/);
  assert.match(v090.en.upgradeNotes, /are not part of v0\.9\.0/);
});

test("v0.10.0 facts ship in v0.10.0 on the current immutable tag", () => {
  for (const factId of releasedV0100FactIds) {
    const item = factsById.get(factId);
    assert.ok(item, `missing ${factId}`);
    assert.equal(item.status, "released", `${factId} must be released`);
    assert.equal(item.sinceVersion, "0.10.0", `${factId} was delivered in v0.10.0`);
    assert.ok(citesTag(item, `v${project.currentVersion}`), `${factId} must cite the v${project.currentVersion} tag`);
    assert.ok(current.shippedFactIds.includes(factId), `v0.10.0 must ship ${factId}`);
    assert.ok(!v090.shippedFactIds.includes(factId), `${factId} must not be backdated into v0.9.0`);
  }
  assert.deepEqual(current.developmentFactIds, [], "nothing on main is ahead of the v0.10.0 tag");
  for (const item of facts) {
    assert.notEqual(item.status, "development", `${item.id} is still marked as unreleased main work`);
    assert.doesNotMatch(item.sinceVersion, /^post-/, `${item.id} still names a post-tag version`);
  }
});

test("superseded facts stay in the releases that shipped them, on that release's tag", () => {
  for (const [oldId, newId] of Object.entries(supersededBy)) {
    const old = factsById.get(oldId);
    assert.deepEqual([...old.pageIds], [], `${oldId} no longer fronts a current page`);
    assert.ok(!current.shippedFactIds.includes(oldId), `v0.10.0 ships ${newId} instead of ${oldId}`);
    assert.ok(!current.limitationFactIds.includes(oldId), `${oldId} is not a v0.10.0 limitation`);
    assert.ok(v090.shippedFactIds.includes(oldId), `v0.9.0 really did ship ${oldId}`);
    assert.ok(citesTag(old, "v0.9.0"), `${oldId} cites the v0.9.0 tag it describes`);
    assert.ok(!citesTag(old, `v${project.currentVersion}`), `${oldId} must not cite code the current tag replaced`);
    assert.ok(factsById.get(newId).pageIds.length > 0, `${newId} takes over the page slot`);
  }
});

test("unverified v0.10.0 surfaces are visible limitations with their boundary sentences", () => {
  for (const factId of ["structured-output-generable-schemas", "rerank-mlx-rerankers", "chat-template-native", "audio-app-cache", "audio-endpoints", "audio-in-app", "mtp-drafter-detection"]) {
    assert.ok(current.limitationFactIds.includes(factId), `${factId} must be a visible v0.10.0 limitation`);
  }

  const schemas = both("structured-output-generable-schemas");
  assert.match(schemas.en.detail, /no end-to-end Xcode 27 or Foundation Models connection is claimed as verified/);
  assert.match(schemas.en.detail, /this is schema and wire compatibility, not a verified integration/);
  assert.match(schemas.en.detail, /driven against a live Qwen3\.6 checkpoint for structured output/);
  assert.match(schemas.en.detail, /null, unions, pattern, and multipleOf remain unsupported/);
  assert.match(schemas.en.detail, /a recursive schema is a 400/);
  assert.match(schemas.zh.detail, /不宣称已验证 Xcode 27 或 Foundation Models 的端到端连接/);

  const rerank = both("rerank-mlx-rerankers");
  assert.match(rerank.en.detail, /The XLM-RoBERTa path \(BGE\) and Jina v3 were not run on a real checkpoint/);
  assert.match(rerank.en.detail, /Jina v3 reports a cosine similarity rather than a probability/);
  assert.match(rerank.en.detail, /bi-encoder cosine stays the fallback for checkpoints that are not rerankers/);
  assert.match(rerank.en.detail, /2e-6/);
  assert.match(rerank.zh.detail, /XLM-RoBERTa 路径（BGE）与 Jina v3 尚未在真实检查点上运行/);

  const audio = both("audio-app-cache");
  assert.match(audio.en.detail, /No test here runs audio inference/);
  assert.match(audio.en.detail, /no transcription or synthesis quality is claimed/);
  assert.match(audio.en.detail, /Copies v0\.9\.0 left there are not removed/);
  assert.match(audio.en.detail, /Still fetched into the default Hugging Face cache/);
  assert.match(audio.zh.detail, /没有任何测试运行音频推理/);
  assert.match(audio.zh.detail, /不对转写或合成质量作任何宣称/);

  const templates = both("chat-template-native");
  assert.match(templates.en.detail, /tool calling on that model is not supported/);
  assert.match(templates.en.detail, /Template support is still not universal/);
  assert.match(templates.zh.detail, /不支持该模型的工具调用/);
});

test("measured v0.10.0 claims keep their scope", () => {
  const unload = both("unload-releases-memory");
  for (const figure of ["7,021 MB", "7,035 MB", "15 MB", "512 MiB"]) assert.match(unload.en.detail, new RegExp(figure));
  assert.match(unload.en.detail, /That is one measured checkpoint, not a guarantee for every model/);
  assert.match(unload.en.detail, /which an unload does not release/);
  assert.match(unload.zh.detail, /并非对所有模型的保证/);

  const usage = both("usage-reporting");
  assert.match(usage.en.detail, /withheld from requests that carry an Origin header/);
  assert.match(usage.en.detail, /not every OpenAI streaming option/);
  assert.match(usage.zh.detail, /带 Origin 请求头的请求不会得到该值/);

  const pool = both("aux-models-in-pool");
  assert.match(pool.en.summary, /^In the app,/);
  assert.match(pool.en.detail, /The CLI's server has no pool/);
  assert.match(pool.en.detail, /is now a 404 instead of unloading the chat model regardless/);
  assert.match(pool.en.detail, /not OpenAI-compatible model management/);

  const fork = both("mlx-fork-0-32-3");
  for (const issue of ["mlx#3922", "mlx#4009", "mlx#4372", "mlx#4660", "mlx#4431", "mlx#4775"]) assert.match(`${fork.en.summary} ${fork.en.detail}`, new RegExp(issue));
  assert.match(fork.en.summary, /now carries five of its own/);
  assert.match(fork.en.detail, /no macMLX path selects today/);
  assert.match(fork.en.detail, /none through a real checkpoint/);
  assert.match(fork.zh.detail, /均未通过真实检查点复现/);
  assert.match(fork.en.detail, /checked by reading rather than by an output comparison/);
  assert.match(fork.en.detail, /Gemma3Text's separately reworked prefill is not covered/);
  assert.match(fork.en.detail, /not a performance claim/);
  assert.match(fork.zh.detail, /而非性能宣称/);

  const lockfiles = both("dmg-committed-lockfiles");
  assert.match(lockfiles.en.detail, /it is not a reproducible-build or code-signing claim, and the DMG remains unsigned/);
  assert.match(lockfiles.zh.detail, /DMG 仍未签名/);
});

test("the v0.10.0 release record states what it does not claim", () => {
  assert.match(current.en.summary, /^The current release/);
  assert.match(current.en.compatibilityNotes, /is not a verified Xcode 27 or Foundation Models end-to-end connection/);
  assert.match(current.en.compatibilityNotes, /the XLM-RoBERTa \(BGE\) and Jina v3 rerank paths were not run on a real checkpoint/);
  assert.match(current.en.compatibilityNotes, /no test runs audio inference/);
  assert.match(current["zh-Hans"].compatibilityNotes, /并不等于已验证的 Xcode 27 或 Foundation Models 端到端连接/);
  assert.match(current["zh-Hans"].compatibilityNotes, /没有任何测试运行音频推理/);

  const page = pages.find((item) => item.id === "release-v0-10-0");
  assert.match(page.en.directAnswer, /Schema compatibility is not a verified Xcode 27 connection/);
  assert.match(page.en.directAnswer, /were not run on real checkpoints/);
  const historical = pages.find((item) => item.id === "release-v0-9-0");
  assert.match(historical.en.title, /\(historical\)/);
  assert.deepEqual(historical.blocks.find((block) => block.type === "sources").factIds, []);

  const apis = faqs.find((item) => item.id === "apis");
  assert.match(apis.en.answer, /^v0\.10\.0 /);
  assert.match(apis.en.answer, /Schema compatibility with Apple's client is not a verified end-to-end Xcode 27 connection/);
});

test("planned work is phrased without a version-number boundary", () => {
  const planned = facts.filter((item) => item.status === "planned");
  assert.ok(planned.some((item) => item.id === "foundation-models-e2e"));
  for (const item of planned) {
    for (const locale of ["en", "zh-Hans"]) {
      assert.doesNotMatch(`${item[locale].summary} ${item[locale].detail}`, /v?\d+\.\d+\.\d+/, `${item.id} must not tie its boundary to a version number`);
    }
  }
  const e2e = both("foundation-models-e2e");
  assert.match(e2e.en.detail, /no end-to-end connection through Xcode 27 or a Foundation Models app has been verified/);
  assert.match(e2e.en.detail, /no generation quality through either path is claimed/);
  assert.ok(current.plannedFactIds.includes("foundation-models-e2e"));
  assert.ok(!v090.plannedFactIds.includes("foundation-models-e2e"), "the plan postdates the v0.9.0 record");
});

test("the MTP detection fact no longer leans on what 'today' can load", () => {
  const mtp = both("mtp-drafter-detection");
  assert.doesNotMatch(mtp.en.detail, /today/);
  assert.doesNotMatch(mtp.zh.detail, /今天/);
  assert.match(mtp.en.detail, /deliberately not wired to use them for decoding/);
  assert.match(mtp.en.detail, /not released multi-token-prediction decoding/);
});
