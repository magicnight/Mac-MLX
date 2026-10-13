import test from "node:test";
import assert from "node:assert/strict";
import { competitors } from "../content/competitors.mjs";
import { pages } from "../content/pages.mjs";

// The compare pages name competitor versions in prose that no freshness guard
// covers, and twice now a refresh of competitors.mjs left that prose behind.
const names = competitors.map((item) => item.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
// A version after the name, allowing a possessive, an opening parenthesis or
// one capitalized qualifier in between, so "LM Studio Bionic 1.1.0" is seen.
const versionAfterName = new RegExp(`\\b(${names})(?:'s)?\\s+(?:\\(|[A-Z][a-z]+\\s+)?(v?\\d+(?:\\.\\d+)+(?: Build \\d+)?|b\\d+)`, "g");
const monthDated = /\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.? (?:\d{1,2}, )?\d{4}\b|\d{4}\s*年\s*\d{1,2}\s*月/;

function strings(value, out = []) {
  if (typeof value === "string") out.push(value);
  else if (Array.isArray(value)) for (const item of value) strings(item, out);
  else if (value && typeof value === "object") for (const item of Object.values(value)) strings(item, out);
  return out;
}

const versionsNamed = (text) => [...text.matchAll(versionAfterName)].map((match) => [match[1], match[2]]);
const comparisonPages = pages.filter((page) => page.blocks.some((block) => block.type === "comparison"));

test("the guard sees a version behind a qualifier, a possessive or a parenthesis", () => {
  assert.deepEqual(versionsNamed("macMLX v0.9.0 and LM Studio Bionic 1.1.0 using official sources"), [["LM Studio", "1.1.0"]]);
  assert.deepEqual(versionsNamed("与 LM Studio 0.4.26 Build 5。"), [["LM Studio", "0.4.26 Build 5"]]);
  assert.deepEqual(versionsNamed("oMLX's v0.6.3 snapshot and Ollama (v0.33.1), SwiftLM b704"), [["oMLX", "v0.6.3"], ["Ollama", "v0.33.1"], ["SwiftLM", "b704"]]);
  assert.deepEqual(versionsNamed("macMLX v0.9.0 runs its default engine in-process"), []);
});

test("every competitor version named in page text is the verified snapshot", () => {
  const verified = new Map(competitors.map((item) => [item.name, item.verifiedVersion]));
  let named = 0;
  for (const page of pages) {
    for (const text of strings(page)) {
      for (const [name, version] of versionsNamed(text)) {
        named += 1;
        assert.equal(version, verified.get(name), `${page.id} names "${name} ${version}" but competitors.mjs verified ${verified.get(name)}`);
      }
    }
  }
  assert.ok(named >= 6, `expected the compare pages to name competitor versions, found ${named}`);
});

test("comparison pages date their snapshots through the registry, never by month", () => {
  assert.ok(comparisonPages.length >= 4, "the compare hub and its per-competitor pages carry comparison blocks");
  for (const page of comparisonPages) for (const text of strings(page)) assert.doesNotMatch(text, monthDated, `${page.id}: ${text.slice(0, 80)}`);
});
