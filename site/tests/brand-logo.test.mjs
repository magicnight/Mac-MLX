import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const canonicalURL = new URL("../assets/brand/macmlx-mark.svg", import.meta.url);
const faviconURL = new URL("../assets/brand/favicon.svg", import.meta.url);

const expectedCanonical = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" role="img" aria-labelledby="title">
  <title id="title">macMLX Signal M</title>
  <rect x="4" y="4" width="120" height="120" rx="34" fill="#F3F1EA"/>
  <path d="M28 88V39l36 38 36-38v49" fill="none" stroke="#111311" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="64" cy="77" r="8.5" fill="#7196FF"/>
  <circle cx="100" cy="39" r="6" fill="#89E67A"/>
</svg>`;

function assertSelfContainedSVG(svg) {
  const withoutNamespace = svg.replace('xmlns="http://www.w3.org/2000/svg"', "");

  assert.doesNotMatch(svg, /<script\b/i);
  assert.doesNotMatch(withoutNamespace, /https?:/i);
  assert.doesNotMatch(svg, /data:/i);
  assert.doesNotMatch(svg, /\b(?:href|src)\s*=/i);
  assert.doesNotMatch(svg, /url\s*\(/i);
}

test("canonical Signal M locks the approved geometry and palette", async () => {
  const svg = (await readFile(canonicalURL, "utf8")).trim();

  assert.equal(svg, expectedCanonical);
  assert.match(svg, /viewBox="0 0 128 128"/);
  assert.match(svg, /fill="#F3F1EA"/);
  assert.match(svg, /stroke="#111311"/);
  assert.match(svg, /fill="#7196FF"/);
  assert.match(svg, /fill="#89E67A"/);
  assertSelfContainedSVG(svg);
});

test("favicon keeps the optical Signal M without the green signal node", async () => {
  const svg = (await readFile(faviconURL, "utf8")).trim();

  assert.match(svg, /viewBox="0 0 128 128"/);
  assert.match(svg, /<title\b[^>]*>[^<]+<\/title>/);
  assert.match(svg, /<rect x="4" y="4" width="120" height="120" rx="34" fill="#F3F1EA"\/>/);
  assert.match(svg, /<path d="M28 88V39l36 38 36-38v49" fill="none" stroke="#111311" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"\/>/);
  assert.match(svg, /<circle cx="64" cy="77" r="8.5" fill="#7196FF"\/>/);
  assert.doesNotMatch(svg, /#89E67A/i);
  assert.equal((svg.match(/<circle\b/g) ?? []).length, 1);
  assertSelfContainedSVG(svg);
});
