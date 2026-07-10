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
  "compare-lm-studio", "compare-omlx", "releases", "release-v0-5-3",
];

test("page and route catalogs contain the exact bilingual hub", () => {
  assert.deepEqual(routes.map((route) => route.id), expectedIds);
  assert.deepEqual(pages.map((page) => page.id), expectedIds.slice(1));
  assert.equal(new Set(routes.flatMap((route) => Object.values(route.paths))).size, 26);
  assert.equal(routes.find((route) => route.id === "release-v0-5-3").paths.en, "/releases/v0-5-3/");
  assert.equal(routes.find((route) => route.id === "release-v0-5-3").paths["zh-Hans"], "/zh/releases/v0-5-3/");
});

test("every page validates cross-references and has related coverage", () => {
  assert.doesNotThrow(() => validateContentHub({ facts, competitors, faqs, releases, pages, macmlxComparisonProfile }, { today: "2026-07-10", maxAgeDays: 45 }));
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
  const expected = new Set(Object.values(macmlxComparisonProfile.en).flatMap((cell) => cell.sourceFactIds));
  for (const page of pages.filter((item) => item.blocks.some((block) => block.type === "comparison"))) {
    const sourceBlock = page.blocks.find((block) => block.type === "sources");
    for (const factId of expected) assert.ok(sourceBlock.factIds.includes(factId), `${page.id} omits comparison source fact ${factId}`);
  }
});

test("release pages close over every release fact source and official release source", () => {
  const release = releases[0];
  const expectedFacts = new Set([
    ...release.shippedFactIds,
    ...release.limitationFactIds,
    ...release.developmentFactIds,
    ...release.plannedFactIds,
  ]);
  for (const id of ["releases", "release-v0-5-3"]) {
    const sourceBlock = pages.find((page) => page.id === id).blocks.find((block) => block.type === "sources");
    assert.deepEqual(new Set(sourceBlock.factIds), expectedFacts, `${id} fact source closure`);
    assert.deepEqual(sourceBlock.releaseIds, [release.id], `${id} official release source closure`);
  }
});

test("all pages receive at least two contextual related links", () => {
  const incoming = Object.fromEntries(pages.map((page) => [page.id, 0]));
  for (const page of pages) for (const id of page.relatedIds) {
    assert.ok(Object.hasOwn(incoming, id), `${page.id} links to unknown page ${id}`);
    incoming[id] += 1;
  }
  for (const [id, count] of Object.entries(incoming)) assert.ok(count >= 2, `${id} has only ${count} incoming related links`);
  assert.deepEqual(incoming, {
    architecture: 7,
    "api-compatibility": 5,
    models: 5,
    "choosing-a-model": 2,
    "vision-language-models": 3,
    faq: 2,
    compare: 3,
    "compare-ollama": 2,
    "compare-lm-studio": 2,
    "compare-omlx": 2,
    releases: 4,
    "release-v0-5-3": 3,
  });
});
