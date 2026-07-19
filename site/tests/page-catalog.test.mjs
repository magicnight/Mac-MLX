import assert from "node:assert/strict";
import test from "node:test";

import { competitors } from "../content/competitors.mjs";
import { facts, macmlxComparisonProfile } from "../content/facts.mjs";
import { faqs } from "../content/faqs.mjs";
import { pages } from "../content/pages.mjs";
import { releases } from "../content/releases.mjs";
import { validateContentHub } from "../lib/content.mjs";
import { routes } from "../routes.mjs";

const expectedIds = [
  "home", "architecture", "api-compatibility", "models", "choosing-a-model",
  "vision-language-models", "faq", "compare", "compare-ollama",
  "compare-lm-studio", "compare-omlx", "releases", "release-v0-8-0", "release-v0-7-0", "release-v0-6-2", "release-v0-5-3",
];

test("page and route catalogs contain the exact bilingual hub", () => {
  assert.deepEqual(routes.map((route) => route.id), expectedIds);
  assert.deepEqual(pages.map((page) => page.id), expectedIds.slice(1));
  assert.equal(new Set(routes.flatMap((route) => Object.values(route.paths))).size, 32);
  assert.equal(routes.find((route) => route.id === "release-v0-8-0").paths.en, "/releases/v0-8-0/");
  assert.equal(routes.find((route) => route.id === "release-v0-8-0").paths["zh-Hans"], "/zh/releases/v0-8-0/");
  assert.equal(routes.find((route) => route.id === "release-v0-7-0").paths.en, "/releases/v0-7-0/");
  assert.equal(routes.find((route) => route.id === "release-v0-7-0").paths["zh-Hans"], "/zh/releases/v0-7-0/");
  assert.equal(routes.find((route) => route.id === "release-v0-6-2").paths.en, "/releases/v0-6-2/");
  assert.equal(routes.find((route) => route.id === "release-v0-6-2").paths["zh-Hans"], "/zh/releases/v0-6-2/");
  assert.equal(routes.find((route) => route.id === "release-v0-5-3").paths.en, "/releases/v0-5-3/");
  assert.equal(routes.find((route) => route.id === "release-v0-5-3").paths["zh-Hans"], "/zh/releases/v0-5-3/");
});

test("every page validates cross-references and has related coverage", () => {
  assert.doesNotThrow(() => validateContentHub({ facts, competitors, faqs, releases, pages, macmlxComparisonProfile }, { today: "2026-07-19", maxAgeDays: 45 }));
  assert.equal(pages.every((page) => page.relatedIds.length >= 2), true);
  assert.deepEqual(new Set(pages.flatMap((page) => page.relatedIds)), new Set(pages.map((page) => page.id)));
});

test("page catalog includes audited API, model, FAQ, comparison, and release blocks", () => {
  const byId = Object.fromEntries(pages.map((page) => [page.id, page]));
  assert.ok(byId["api-compatibility"].blocks.some((block) => block.type === "table"));
  assert.ok(byId["choosing-a-model"].blocks.some((block) => block.type === "table"));
  assert.ok(byId["vision-language-models"].blocks.some((block) => block.type === "table"));
  assert.deepEqual(byId.faq.blocks.find((block) => block.type === "faq").faqIds, faqs.map((item) => item.id));
  assert.deepEqual(byId.compare.blocks.find((block) => block.type === "comparison").competitorIds, competitors.map((item) => item.id));
  assert.deepEqual(byId["release-v0-8-0"].blocks.find((block) => block.type === "release").releaseIds, ["v0-8-0"]);
  assert.deepEqual(byId["release-v0-7-0"].blocks.find((block) => block.type === "release").releaseIds, ["v0-7-0"]);
  assert.deepEqual(byId["release-v0-6-2"].blocks.find((block) => block.type === "release").releaseIds, ["v0-6-2"]);
  assert.deepEqual(byId["release-v0-5-3"].blocks.find((block) => block.type === "release").releaseIds, ["v0-5-3"]);
});

test("every English article begins with a concise 40 to 80 word direct answer", () => {
  for (const page of pages) {
    const words = page.en.directAnswer.trim().split(/\s+/).length;
    assert.ok(words >= 40 && words <= 80, `${page.id} direct answer has ${words} words`);
  }
});

test("fact pageIds exactly equal direct fact-card usage", () => {
  const directUsage = new Map(facts.map((fact) => [fact.id, []]));
  for (const page of pages) {
    for (const block of page.blocks.filter((item) => item.type === "facts")) {
      for (const factId of block.factIds) directUsage.get(factId).push(page.id);
    }
  }
  for (const fact of facts) {
    assert.deepEqual([...fact.pageIds].sort(), directUsage.get(fact.id).sort(), `${fact.id} pageIds must equal direct usage`);
  }
  assert.ok(directUsage.get("tiered-cache").includes("choosing-a-model"));
});

test("FAQ source aggregation exactly covers every indirectly cited fact", () => {
  const faqPage = pages.find((page) => page.id === "faq");
  const faqBlock = faqPage.blocks.find((block) => block.type === "faq");
  const sourceBlock = faqPage.blocks.find((block) => block.type === "sources");
  const expected = new Set(faqBlock.faqIds.flatMap((id) => faqs.find((item) => item.id === id).factIds));
  assert.deepEqual(new Set(sourceBlock.factIds), expected);
});

test("comparison source aggregation covers every macMLX profile citation", () => {
  const expected = new Set(Object.values(macmlxComparisonProfile).flatMap((profile) => Object.values(profile).flatMap((cell) => cell.sourceFactIds)));
  for (const page of pages.filter((item) => item.blocks.some((block) => block.type === "comparison"))) {
    const sourceBlock = page.blocks.find((block) => block.type === "sources");
    for (const factId of expected) assert.ok(sourceBlock.factIds.includes(factId), `${page.id} omits comparison source fact ${factId}`);
  }
});

test("release pages keep current fact closure and historical evidence version-scoped", () => {
  const releaseFacts = (release) => new Set([
    ...release.shippedFactIds,
    ...release.limitationFactIds,
    ...release.developmentFactIds,
    ...release.plannedFactIds,
  ]);
  const currentRelease = releases.find((release) => release.id === "v0-8-0");
  const currentSources = pages.find((page) => page.id === "release-v0-8-0").blocks.find((block) => block.type === "sources");
  assert.deepEqual(new Set(currentSources.factIds), releaseFacts(currentRelease), "release-v0-8-0 fact source closure");
  assert.deepEqual(currentSources.releaseIds, ["v0-8-0"], "release-v0-8-0 official release source closure");

  for (const version of ["0.7.0", "0.6.2", "0.5.3"]) {
    const id = `release-v${version.replaceAll(".", "-")}`;
    const historicalRelease = releases.find((release) => release.version === version);
    const historicalSources = pages.find((page) => page.id === id).blocks.find((block) => block.type === "sources");
    assert.deepEqual(historicalSources.factIds, [], `${id} must not expose current fact sources`);
    assert.deepEqual(historicalSources.releaseIds, [historicalRelease.id], `${id} official release source closure`);
    const escaped = version.replaceAll(".", "\\.");
    for (const url of historicalRelease.officialSources) {
      assert.match(url, new RegExp(`/(?:releases/tag|blob)/v${escaped}(?:/|$)`), `historical source must be v${version}-tagged: ${url}`);
      assert.doesNotMatch(url, /\/main\//, `historical source must not use mutable evidence: ${url}`);
    }
  }

  const hubSources = pages.find((page) => page.id === "releases").blocks.find((block) => block.type === "sources");
  assert.deepEqual(new Set(hubSources.factIds), new Set(releases.flatMap((release) => [...releaseFacts(release)])), "releases fact source closure");
  assert.deepEqual(new Set(hubSources.releaseIds), new Set(releases.map((release) => release.id)), "releases official release source closure");
});

test("all pages receive at least two contextual related links", () => {
  const incoming = Object.fromEntries(pages.map((page) => [page.id, 0]));
  for (const page of pages) for (const id of page.relatedIds) {
    assert.ok(Object.hasOwn(incoming, id), `${page.id} links to unknown page ${id}`);
    incoming[id] += 1;
  }
  for (const [id, count] of Object.entries(incoming)) assert.ok(count >= 2, `${id} has only ${count} incoming related links`);
});
