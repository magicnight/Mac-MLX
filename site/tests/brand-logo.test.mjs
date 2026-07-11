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

const expectedFavicon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" role="img" aria-labelledby="title">
  <title id="title">macMLX Signal M favicon</title>
  <rect x="4" y="4" width="120" height="120" rx="34" fill="#F3F1EA"/>
  <path d="M28 88V39l36 38 36-38v49" fill="none" stroke="#111311" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="64" cy="77" r="8.5" fill="#7196FF"/>
</svg>`;

const allowedAttributes = new Map([
  ["svg", new Set(["xmlns", "viewBox", "role", "aria-labelledby"])],
  ["title", new Set(["id"])],
  ["rect", new Set(["x", "y", "width", "height", "rx", "fill"])],
  ["path", new Set(["d", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin"])],
  ["circle", new Set(["cx", "cy", "r", "fill"])],
]);

function parseAttributes(element, source) {
  const attributes = new Map();
  const attributePattern = /\s*([A-Za-z_:][\w:.-]*)\s*=\s*"([^"]*)"/gy;
  let offset = 0;

  while (offset < source.length) {
    attributePattern.lastIndex = offset;
    const match = attributePattern.exec(source);
    assert.ok(match, `invalid attribute syntax on <${element}>`);
    const [, name, value] = match;
    assert.ok(allowedAttributes.get(element)?.has(name), `attribute ${name} is not allowed on <${element}>`);
    assert.ok(!attributes.has(name), `duplicate attribute ${name} on <${element}>`);
    attributes.set(name, value);
    offset = attributePattern.lastIndex;
  }

  return attributes;
}

function assertSelfContainedSVG(svg) {
  const namespacePattern = /xmlns="http:\/\/www\.w3\.org\/2000\/svg"/g;
  assert.equal((svg.match(namespacePattern) ?? []).length, 1, "the fixed SVG namespace must appear exactly once");
  assert.match(svg, /^<svg\b[^>]*\bxmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
  const withoutNamespace = svg.replace('xmlns="http://www.w3.org/2000/svg"', "");

  assert.doesNotMatch(svg, /<script\b/i);
  assert.doesNotMatch(svg, /<style\b/i);
  assert.doesNotMatch(svg, /@import\b/i);
  assert.doesNotMatch(svg, /<!DOCTYPE\b/i);
  assert.doesNotMatch(svg, /<!ENTITY\b/i);
  assert.doesNotMatch(svg, /<\?/);
  assert.doesNotMatch(withoutNamespace, /https?:/i);
  assert.doesNotMatch(withoutNamespace, /\/\//);
  assert.doesNotMatch(svg, /data:/i);
  assert.doesNotMatch(svg, /\b(?:href|src)\s*=/i);
  assert.doesNotMatch(svg, /url\s*\(/i);

  const openingElements = [];
  const tagPattern = /<\s*(\/?)\s*([A-Za-z][\w:-]*)([^>]*)>/g;
  for (const [, closing, element, rawAttributes] of svg.matchAll(tagPattern)) {
    assert.ok(allowedAttributes.has(element), `element <${element}> is not allowed`);
    if (closing) {
      assert.equal(rawAttributes.trim(), "", `closing </${element}> cannot have attributes`);
      continue;
    }

    openingElements.push(element);
    const attributeSource = rawAttributes.trim().replace(/\/$/, "").trim();
    parseAttributes(element, attributeSource);
  }

  return openingElements;
}

function assertAccessibleTitle(svg, expectedTitle) {
  const svgTag = svg.match(/<svg\b([^>]*)>/);
  const title = svg.match(/<title\b([^>]*)>([^<]+)<\/title>/);
  assert.ok(svgTag, "SVG root is required");
  assert.ok(title, "SVG title is required");

  const svgAttributes = parseAttributes("svg", svgTag[1].trim());
  const titleAttributes = parseAttributes("title", title[1].trim());
  assert.equal(svgAttributes.get("role"), "img");
  assert.equal(svgAttributes.get("aria-labelledby"), titleAttributes.get("id"));
  assert.equal(title[2], expectedTitle);
}

test("SVG safety validation rejects active and externally loaded content", () => {
  const maliciousSVGs = [
    ["script", `<svg xmlns="http://www.w3.org/2000/svg"><script>unsafe()</script></svg>`],
    ["style", `<svg xmlns="http://www.w3.org/2000/svg"><style>.mark { fill: red; }</style></svg>`],
    ["import", `<svg xmlns="http://www.w3.org/2000/svg"><style>@import "//example.com/mark.css";</style></svg>`],
    ["doctype", `<!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg"></svg>`],
    ["entity", `<!DOCTYPE svg [<!ENTITY payload "unsafe">]><svg xmlns="http://www.w3.org/2000/svg"></svg>`],
    ["processing instruction", `<?unsafe processing?><svg xmlns="http://www.w3.org/2000/svg"></svg>`],
    ["href", `<svg xmlns="http://www.w3.org/2000/svg"><circle href="asset.svg"/></svg>`],
    ["src", `<svg xmlns="http://www.w3.org/2000/svg"><circle src="asset.svg"/></svg>`],
    ["url", `<svg xmlns="http://www.w3.org/2000/svg"><circle fill="url(#paint)"/></svg>`],
    ["data URL", `<svg xmlns="http://www.w3.org/2000/svg"><title>data:image/svg+xml,unsafe</title></svg>`],
    ["HTTP URL", `<svg xmlns="http://www.w3.org/2000/svg"><title>https://example.com</title></svg>`],
  ];

  for (const [entry, svg] of maliciousSVGs) {
    assert.throws(() => assertSelfContainedSVG(svg), undefined, `${entry} must be rejected`);
  }
});

test("canonical Signal M locks the approved geometry and palette", async () => {
  const svg = (await readFile(canonicalURL, "utf8")).trim();

  assert.equal(svg, expectedCanonical);
  assert.match(svg, /viewBox="0 0 128 128"/);
  assert.match(svg, /fill="#F3F1EA"/);
  assert.match(svg, /stroke="#111311"/);
  assert.match(svg, /fill="#7196FF"/);
  assert.match(svg, /fill="#89E67A"/);
  assert.deepEqual(assertSelfContainedSVG(svg), ["svg", "title", "rect", "path", "circle", "circle"]);
  assertAccessibleTitle(svg, "macMLX Signal M");
});

test("favicon keeps the optical Signal M without the green signal node", async () => {
  const svg = (await readFile(faviconURL, "utf8")).trim();

  assert.equal(svg, expectedFavicon);
  assert.match(svg, /viewBox="0 0 128 128"/);
  assert.match(svg, /<rect x="4" y="4" width="120" height="120" rx="34" fill="#F3F1EA"\/>/);
  assert.match(svg, /<path d="M28 88V39l36 38 36-38v49" fill="none" stroke="#111311" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"\/>/);
  assert.match(svg, /<circle cx="64" cy="77" r="8.5" fill="#7196FF"\/>/);
  assert.doesNotMatch(svg, /#89E67A/i);
  assert.equal((svg.match(/<circle\b/g) ?? []).length, 1);
  assert.deepEqual(assertSelfContainedSVG(svg), ["svg", "title", "rect", "path", "circle"]);
  assertAccessibleTitle(svg, "macMLX Signal M favicon");
});
