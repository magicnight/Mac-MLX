import assert from "node:assert/strict";
import { access } from "node:fs/promises";
import test from "node:test";

import { competitors } from "../content/competitors.mjs";
import { facts, macmlxComparisonProfile } from "../content/facts.mjs";
import { faqs } from "../content/faqs.mjs";
import { pages } from "../content/pages.mjs";
import { releases } from "../content/releases.mjs";
import { project } from "../content/project.mjs";

const requiredRegistries = ["facts", "competitors", "faqs", "releases", "pages"];

test("the content hub has every evidence registry", async () => {
  for (const name of requiredRegistries) {
    await assert.doesNotReject(
      access(new URL(`../content/${name}.mjs`, import.meta.url)),
      `missing ${name} registry`,
    );
  }
});

test("the content hub has a strict cross-registry validator", async () => {
  await assert.doesNotReject(
    access(new URL("../lib/content.mjs", import.meta.url)),
    "expected the content validator",
  );
});

test("the validator exposes the complete evidence contract", async () => {
  const content = await import("../lib/content.mjs");
  assert.equal(typeof content.validateContentHub, "function");
  assert.equal(typeof content.validateFacts, "function");
  assert.equal(typeof content.validateCompetitors, "function");
  assert.equal(typeof content.validateFAQs, "function");
  assert.equal(typeof content.validateReleases, "function");
  assert.equal(typeof content.validatePages, "function");
  assert.equal(typeof content.validateComparisonProfile, "function");
  assert.equal(typeof content.validateFactPageReferences, "function");
});

test("fact, competitor, FAQ, and release registries pass strict validation", async () => {
  const { validateContentHub } = await import("../lib/content.mjs");
  assert.doesNotThrow(() => validateContentHub({ facts, competitors, faqs, releases, pages, macmlxComparisonProfile }, { today: "2026-07-10", maxAgeDays: 45, allowEmptyPages: true }));
  assert.equal(facts.filter((fact) => fact.status === "released").length > 0, true);
  assert.equal(facts.filter((fact) => fact.status === "development").length > 0, true);
  assert.equal(facts.filter((fact) => fact.status === "planned").length > 0, true);
  assert.deepEqual(competitors.map((item) => [item.id, item.verifiedVersion]), [
    ["ollama", "v0.31.2"],
    ["lm-studio", "0.4.19 Build 2"],
    ["omlx", "v0.4.4"],
    ["swama", "v2.2.0"],
    ["swiftlm", "b648"],
  ]);
  assert.equal(faqs.length, 8);
  assert.equal(releases[0].version, "0.5.3");
});

test("validation rejects stale dates, mutable release sources, and broken references", async () => {
  const { validateFacts, validateFAQs } = await import("../lib/content.mjs");
  const stale = { ...facts[0], lastVerified: "2026-01-01" };
  assert.throws(() => validateFacts([stale], { today: "2026-07-10", maxAgeDays: 45 }), /stale fact/);

  const mutable = { ...facts.find((fact) => fact.status === "released"), sourceUrls: ["https://github.com/magicnight/mac-mlx/blob/main/CHANGELOG.md"] };
  assert.throws(() => validateFacts([mutable], { today: "2026-07-10", maxAgeDays: 45 }), /immutable release source/);

  const brokenFAQs = [{ ...faqs[0], factIds: ["not-a-fact"] }, ...faqs.slice(1)];
  assert.throws(() => validateFAQs(brokenFAQs, new Set(facts.map((fact) => fact.id))), /unknown fact/);
});

test("release status lists reject facts from the wrong lifecycle", async () => {
  const { validateReleases } = await import("../lib/content.mjs");
  const factIds = new Set(facts.map((fact) => fact.id));
  const factsById = new Map(facts.map((fact) => [fact.id, fact]));
  for (const [field, wrongFactId, expectedStatus] of [
    ["shippedFactIds", "continuous-batching", "released"],
    ["limitationFactIds", "continuous-batching", "released"],
    ["developmentFactIds", "swift-in-process", "development"],
    ["plannedFactIds", "swift-in-process", "planned"],
  ]) {
    const mismatch = { ...releases[0], [field]: [wrongFactId] };
    assert.throws(
      () => validateReleases([mismatch], factIds, { today: "2026-07-10", maxAgeDays: 45, factsById }),
      new RegExp(`${field}.*${expectedStatus}`),
    );
  }
});

test("validation rejects future dates and sources outside official ownership", async () => {
  const { validateCompetitors, validateFacts, validateReleases } = await import("../lib/content.mjs");
  assert.throws(() => validateFacts([{ ...facts[0], lastVerified: "2026-07-11" }], { today: "2026-07-10", maxAgeDays: 45 }), /future fact/);
  assert.throws(() => validateFacts([{ ...facts[0], sourceUrls: [...facts[0].sourceUrls, "https://example.com/claim"] }], { today: "2026-07-10", maxAgeDays: 45 }), /unapproved macMLX source/);

  const futureCompetitor = { ...competitors[0], snapshotDate: "2026-07-11" };
  assert.throws(() => validateCompetitors([futureCompetitor], { today: "2026-07-10", maxAgeDays: 45 }), /future competitor snapshot/);
  assert.throws(() => validateCompetitors([{ ...competitors[0], lastVerified: "2026-07-11" }], { today: "2026-07-10", maxAgeDays: 45 }), /future competitor verification/);
  const foreignCompetitor = { ...competitors[0], officialSources: [...competitors[0].officialSources, "https://example.com/ollama"] };
  assert.throws(() => validateCompetitors([foreignCompetitor], { today: "2026-07-10", maxAgeDays: 45 }), /unapproved official source/);

  const futureRelease = { ...releases[0], releaseDate: "2026-07-11" };
  assert.throws(() => validateReleases([futureRelease], new Set(facts.map((fact) => fact.id)), { today: "2026-07-10", maxAgeDays: 45, factsById: new Map(facts.map((fact) => [fact.id, fact])) }), /future release date/);
  assert.throws(() => validateReleases([{ ...releases[0], lastVerified: "2026-07-11" }], new Set(facts.map((fact) => fact.id)), { today: "2026-07-10", maxAgeDays: 45, factsById: new Map(facts.map((fact) => [fact.id, fact])) }), /future release verification/);
});

test("competitors, FAQs, releases, and the macMLX comparison profile expose governed records", async () => {
  const registry = await import("../content/facts.mjs");
  assert.ok(registry.macmlxComparisonProfile, "macMLX comparison profile must be registry-backed");
  for (const locale of ["en", "zh-Hans"]) {
    for (const cell of Object.values(registry.macmlxComparisonProfile[locale])) {
      assert.ok(cell.text);
      assert.ok(cell.sourceFactIds.length > 0);
    }
  }
  const { validateComparisonProfile } = await import("../lib/content.mjs");
  const brokenProfile = structuredClone(registry.macmlxComparisonProfile);
  brokenProfile.en.platform.sourceFactIds = ["not-registered"];
  assert.throws(() => validateComparisonProfile(brokenProfile, new Set(facts.map((fact) => fact.id))), /unknown macMLX comparison fact/);
  assert.equal(competitors.every((item) => ["en", "zh-Hans"].every((locale) => item[locale].limitations?.length > 0)), true);
  assert.deepEqual(faqs.map((item) => item.id), ["platform-installation", "python", "model-selection", "apis", "privacy", "vlm", "large-moe", "roadmap"]);
  assert.match(faqs.find((item) => item.id === "platform-installation").en.answer, /Gatekeeper/);
  assert.match(faqs.find((item) => item.id === "model-selection").en.answer, /physical unified memory/);
  assert.match(faqs.find((item) => item.id === "model-selection").en.answer, /weights, KV cache, activations/);
  assert.ok(faqs.find((item) => item.id === "roadmap").factIds.includes("trie-lcp"));
  assert.ok(faqs.find((item) => item.id === "roadmap").factIds.includes("adaptive-memory-guard"));
  for (const locale of ["en", "zh-Hans"]) {
    assert.ok(releases[0][locale].compatibilityNotes);
    assert.ok(releases[0][locale].upgradeNotes);
  }
});

test("page schema rejects free-form source URLs before rendering", async () => {
  const { validateContentHub } = await import("../lib/content.mjs");
  for (const url of ["javascript:alert(1)", "http://example.com", "https://example.com"]) {
    const brokenPages = structuredClone(pages);
    brokenPages[0].blocks.find((block) => block.type === "sources").sources = [{ label: "Bypass", url }];
    assert.throws(() => validateContentHub({ facts, competitors, faqs, releases, pages: brokenPages, macmlxComparisonProfile }, { today: "2026-07-10", maxAgeDays: 45 }), /free-form sources are not allowed/);
  }
});

test("development proof requires an exact main-branch source pathname", async () => {
  const { validateFacts } = await import("../lib/content.mjs");
  const development = facts.find((fact) => fact.status === "development");
  const querySpoof = { ...development, sourceUrls: ["https://github.com/magicnight/mac-mlx/issues/1?proof=/blob/main/fake.swift"] };
  assert.throws(() => validateFacts([querySpoof], { today: "2026-07-10", maxAgeDays: 45 }), /post-tag main/);
});

test("project and release registries share one current release identity", async () => {
  const { validateReleaseIdentity } = await import("../lib/content.mjs");
  assert.doesNotThrow(() => validateReleaseIdentity(project, releases));
  assert.throws(() => validateReleaseIdentity({ ...project, currentVersion: "9.9.9" }, releases), /currentVersion/);
  assert.throws(() => validateReleaseIdentity({ ...project, releaseDate: "2026-01-01" }, releases), /releaseDate/);
  const releasePage = pages.find((page) => page.id === "release-v0-5-3");
  assert.match(releasePage.en.title, new RegExp(project.currentVersion.replaceAll(".", "\\.")));
  assert.match(releasePage.en.directAnswer, new RegExp(project.currentVersion.replaceAll(".", "\\.")));
});
