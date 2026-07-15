import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { deflateSync, inflateSync } from "node:zlib";
import test from "node:test";
import { mkdir, mkdtemp, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { brandIconSourceDigest, renderBrandIcons, validateBrandIconPNG } from "../../scripts/render-brand-icons.mjs";
import { prepareSite, validateBrandAssets } from "../../scripts/build-public-site.mjs";
import { assetPaths, brandCopiedAssetPaths, copiedAssetPaths } from "../content/assets.mjs";
import { project } from "../content/project.mjs";
import { renderSocialCardSVG } from "../lib/social-card.mjs";
import { installManifestContract } from "../lib/install-manifest.mjs";

const canonicalURL = new URL("../assets/brand/macmlx-mark.svg", import.meta.url);
const faviconURL = new URL("../assets/brand/favicon.svg", import.meta.url);
const manifestURL = new URL("../assets/brand/site.webmanifest", import.meta.url);
const repositoryRoot = new URL("../../", import.meta.url);
const socialCardURL = new URL("../social-card.html", import.meta.url);

const brandAssetPaths = Object.freeze([
  "brand/macmlx-mark.svg",
  "brand/favicon.svg",
  "brand/apple-touch-icon.png",
  "brand/icon-192.png",
  "brand/icon-512.png",
  "brand/site.webmanifest",
]);

const brandLinks = Object.freeze([
  '<link rel="icon" href="/assets/brand/favicon.svg" type="image/svg+xml">',
  '<link rel="apple-touch-icon" href="/assets/brand/apple-touch-icon.png">',
  '<link rel="manifest" href="/assets/brand/site.webmanifest">',
]);

const brandImage = '<img class="brand-mark" src="/assets/brand/macmlx-mark.svg" alt="">';

const rasterIcons = Object.freeze([
  Object.freeze({ filename: "apple-touch-icon.png", width: 180, height: 180 }),
  Object.freeze({ filename: "icon-192.png", width: 192, height: 192 }),
  Object.freeze({ filename: "icon-512.png", width: 512, height: 512 }),
]);
const brandDigestKeyword = "macMLXSourceSHA256";

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

const allowedAttributeValues = new Map([
  ["svg", new Map([
    ["xmlns", new Set(["http://www.w3.org/2000/svg"])],
    ["viewBox", new Set(["0 0 128 128"])],
    ["role", new Set(["img"])],
    ["aria-labelledby", new Set(["title"])],
  ])],
  ["title", new Map([
    ["id", new Set(["title"])],
  ])],
  ["rect", new Map([
    ["x", new Set(["4"])],
    ["y", new Set(["4"])],
    ["width", new Set(["120"])],
    ["height", new Set(["120"])],
    ["rx", new Set(["34"])],
    ["fill", new Set(["#F3F1EA"])],
  ])],
  ["path", new Map([
    ["d", new Set(["M28 88V39l36 38 36-38v49"])],
    ["fill", new Set(["none"])],
    ["stroke", new Set(["#111311"])],
    ["stroke-width", new Set(["14", "16"])],
    ["stroke-linecap", new Set(["round"])],
    ["stroke-linejoin", new Set(["round"])],
  ])],
  ["circle", new Map([
    ["cx", new Set(["64", "100"])],
    ["cy", new Set(["77", "39"])],
    ["r", new Set(["8.5", "6"])],
    ["fill", new Set(["#7196FF", "#89E67A"])],
  ])],
]);

const allowedTitleText = new Set(["macMLX Signal M", "macMLX Signal M favicon"]);

function parseAttributes(element, source) {
  const attributes = new Map();
  const attributePattern = /\s*([A-Za-z_:][\w:.-]*)\s*=\s*"([^"]*)"/gy;
  let offset = 0;

  while (offset < source.length) {
    attributePattern.lastIndex = offset;
    const match = attributePattern.exec(source);
    assert.ok(match, `invalid attribute syntax on <${element}>`);
    const [, name, value] = match;
    const allowedValues = allowedAttributeValues.get(element)?.get(name);
    assert.ok(allowedValues, `attribute ${name} is not allowed on <${element}>`);
    assert.doesNotMatch(value, /\\/, `backslashes are not allowed in ${name} on <${element}>`);
    assert.doesNotMatch(value, /&(?:#\d+|#x[\da-f]+|[a-z][\w.-]*);/i, `character entities are not allowed in ${name} on <${element}>`);
    assert.ok(allowedValues.has(value), `value ${value} is not allowed for ${name} on <${element}>`);
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
    assert.ok(allowedAttributeValues.has(element), `element <${element}> is not allowed`);
    if (closing) {
      assert.equal(rawAttributes.trim(), "", `closing </${element}> cannot have attributes`);
      continue;
    }

    openingElements.push(element);
    const attributeSource = rawAttributes.trim().replace(/\/$/, "").trim();
    parseAttributes(element, attributeSource);
  }

  for (const [, titleText] of svg.matchAll(/<title\b[^>]*>([^<]+)<\/title>/g)) {
    assert.ok(allowedTitleText.has(titleText), `title text ${titleText} is not allowed`);
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

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function inspectPNG(buffer, label) {
  assert.ok(Buffer.isBuffer(buffer), `${label} must be a buffer`);
  assert.deepEqual([...buffer.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10], `${label} must have the PNG signature`);
  assert.ok(buffer.length > 1_000, `${label} must contain meaningful raster data`);

  let offset = 8;
  let header;
  let ihdrCount = 0;
  let iendCount = 0;
  let plteCount = 0;
  let phase = "beforeIHDR";
  const compressed = [];
  while (offset < buffer.length) {
    assert.ok(buffer.length - offset >= 12, `${label} has a truncated PNG chunk`);
    const length = buffer.readUInt32BE(offset);
    assert.ok(length <= buffer.length - offset - 12, `${label} has a truncated PNG chunk`);
    const typeStart = offset + 4;
    const dataStart = offset + 8;
    const crcOffset = dataStart + length;
    const end = crcOffset + 4;
    const type = buffer.toString("ascii", typeStart, dataStart);
    assert.match(type, /^[A-Za-z]{4}$/, `${label} has an invalid PNG chunk type`);
    assert.match(type[2], /[A-Z]/, `${label} PNG chunk ${type} sets the reserved bit`);
    assert.equal(crc32(buffer.subarray(typeStart, crcOffset)), buffer.readUInt32BE(crcOffset), `${label} has a CRC mismatch in ${type}`);

    if (type === "IHDR") {
      ihdrCount += 1;
      assert.equal(phase, "beforeIHDR", `${label} must contain a unique first IHDR`);
      assert.equal(offset, 8, `${label} IHDR must be the first chunk`);
      assert.equal(length, 13, `${label} IHDR must have the canonical length`);
      header = {
        width: buffer.readUInt32BE(dataStart),
        height: buffer.readUInt32BE(dataStart + 4),
        bitDepth: buffer[dataStart + 8],
        colorType: buffer[dataStart + 9],
        compression: buffer[dataStart + 10],
        filter: buffer[dataStart + 11],
        interlace: buffer[dataStart + 12],
      };
      phase = "beforeIDAT";
    } else if (type === "PLTE") {
      plteCount += 1;
      assert.equal(plteCount, 1, `${label} must not repeat PLTE`);
      assert.equal(phase, "beforeIDAT", `${label} PLTE must appear before IDAT`);
      assert.ok(length > 0 && length <= 768 && length % 3 === 0, `${label} has an invalid PLTE length`);
    } else if (type === "IDAT") {
      assert.notEqual(phase, "beforeIHDR", `${label} IDAT cannot precede IHDR`);
      assert.notEqual(phase, "afterIDAT", `${label} IDAT chunks must be contiguous`);
      assert.ok(length > 0, `${label} IDAT must not be empty`);
      compressed.push(buffer.subarray(dataStart, crcOffset));
      phase = "inIDAT";
    } else if (type === "IEND") {
      iendCount += 1;
      assert.ok(phase === "inIDAT" || phase === "afterIDAT", `${label} IEND must follow IDAT`);
      assert.equal(length, 0, `${label} IEND must be empty`);
      assert.equal(end, buffer.length, `${label} must not have trailing bytes`);
      phase = "ended";
    } else {
      if (/^[A-Z]/.test(type)) assert.fail(`${label} has unknown critical chunk ${type}`);
      assert.notEqual(phase, "beforeIHDR", `${label} IHDR must be the first chunk`);
      if (phase === "inIDAT") phase = "afterIDAT";
    }
    offset = end;
  }

  assert.equal(ihdrCount, 1, `${label} must contain one IHDR`);
  assert.equal(iendCount, 1, `${label} must contain one terminal IEND`);
  assert.equal(phase, "ended", `${label} must end after IEND`);
  assert.ok(compressed.length > 0, `${label} must contain IDAT data`);
  assert.deepEqual(
    { bitDepth: header.bitDepth, colorType: header.colorType, compression: header.compression, filter: header.filter, interlace: header.interlace },
    { bitDepth: 8, colorType: 6, compression: 0, filter: 0, interlace: 0 },
    `${label} must be non-interlaced 8-bit RGBA`,
  );

  const scanlines = inflateSync(Buffer.concat(compressed));
  const rowLength = 1 + (header.width * 4);
  assert.equal(scanlines.length, rowLength * header.height, `${label} has inconsistent decompressed scanlines`);
  for (let row = 0; row < header.height; row += 1) assert.ok(scanlines[row * rowLength] <= 4, `${label} has an invalid row filter`);
  assert.ok(scanlines.subarray(1).some((byte) => byte !== 0), `${label} cannot be an empty transparent raster`);
  return header;
}

function pngChunk(type, data) {
  const typeBytes = Buffer.from(type, "ascii");
  const chunk = Buffer.alloc(12 + data.length);
  chunk.writeUInt32BE(data.length, 0);
  typeBytes.copy(chunk, 4);
  data.copy(chunk, 8);
  chunk.writeUInt32BE(crc32(Buffer.concat([typeBytes, data])), 8 + data.length);
  return chunk;
}

function solidPNG(width, height) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr.set([8, 6, 0, 0, 0], 8);
  const scanlines = Buffer.alloc((1 + (width * 4)) * height);
  for (let row = 0; row < height; row += 1) {
    const start = row * (1 + (width * 4));
    scanlines[start] = 0;
    scanlines.fill(0x7f, start + 1, start + 1 + (width * 4));
  }
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", deflateSync(scanlines)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

function pngChunks(buffer) {
  const chunks = [];
  let offset = 8;
  while (offset < buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const dataStart = offset + 8;
    const end = dataStart + length + 4;
    chunks.push({ type: buffer.toString("ascii", offset + 4, dataStart), data: buffer.subarray(dataStart, dataStart + length) });
    offset = end;
  }
  return chunks;
}

function assemblePNG(chunks) {
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    ...chunks.map(({ type, data }) => pngChunk(type, data)),
  ]);
}

function sharpIsAvailable() {
  const require = createRequire(import.meta.url);
  try {
    require.resolve("sharp");
    return true;
  } catch {
    if (!process.env.MACMLX_NODE_MODULES) return false;
    try {
      require.resolve("sharp", { paths: [process.env.MACMLX_NODE_MODULES] });
      return true;
    } catch {
      return false;
    }
  }
}

function solidSharp() {
  return {
    resize(width, height) {
      return {
        png() {
          return { async toBuffer() { return solidPNG(width, height); } };
        },
      };
    },
  };
}

function bufferedSharp(buffer) {
  return {
    resize() {
      return {
        png() {
          return { async toBuffer() { return buffer; } };
        },
      };
    },
  };
}

async function assertOriginalIconState(outputDirectory, originals) {
  for (const icon of rasterIcons) {
    const original = originals.get(icon.filename);
    if (original) assert.deepEqual(await readFile(join(outputDirectory, icon.filename)), original);
    else await assert.rejects(readFile(join(outputDirectory, icon.filename)), { code: "ENOENT" });
  }
  assert.equal(await readFile(join(outputDirectory, "unrelated.svg"), "utf8"), "keep me");
  assert.deepEqual((await readdir(outputDirectory)).sort(), [...originals.keys(), "unrelated.svg"].sort());
}

function occurrenceCount(source, value) {
  return source.split(value).length - 1;
}

test("every localized document carries the Signal M metadata and wordmark contract", async () => {
  const { documents, discoveryFiles } = await prepareSite();
  assert.equal(documents.size, 28);
  const htmlDocuments = new Map([
    ...documents,
    ...[...discoveryFiles].filter(([path]) => path.endsWith(".html")),
  ]);
  assert.equal(htmlDocuments.size, 30);

  for (const [path, html] of htmlDocuments) {
    for (const link of brandLinks) assert.equal(occurrenceCount(html, link), 1, `${path} must contain ${link} exactly once`);
  }

  for (const [path, html] of documents) {
    assert.equal(occurrenceCount(html, brandImage), 2, `${path} needs one header and one footer Signal M`);
    assert.doesNotMatch(html, /<span class="brand-mark"[^>]*><i>/, `${path} must not retain the three-bar mark`);
  }

  for (const path of ["/", "/zh/", "/architecture/", "/zh/architecture/"]) {
    const html = documents.get(path);
    assert.ok(html, `missing representative document ${path}`);
    assert.match(html, new RegExp(`<a class="wordmark"[^>]+aria-label="[^"]+"[^>]*>\\s*${brandImage.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
    assert.match(html, new RegExp(`<a class="wordmark footer-wordmark"[^>]*>${brandImage.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
  }
});

test("home wordmark accessibility labels follow the rendered locale", async () => {
  const { documents } = await prepareSite();
  assert.match(documents.get("/"), /<a class="wordmark" href="#top" aria-label="macMLX Home">/);
  assert.match(documents.get("/zh/"), /<a class="wordmark" href="#top" aria-label="macMLX 首页">/);
});

test("generated home and article documents share the fourth main CSS revision", async () => {
  const { documents } = await prepareSite();
  for (const path of ["/", "/zh/", "/architecture/", "/zh/architecture/"]) {
    const html = documents.get(path);
    assert.match(html, /<link rel="stylesheet" href="\/assets\/css\/main\.css\?v=4">/, path);
    assert.doesNotMatch(html, /\/assets\/css\/main\.css\?v=3(?:"|&)/, path);
  }
});

test("Signal M CSS reserves stable desktop and compact image dimensions", async () => {
  const css = await readFile(new URL("../assets/css/main.css", import.meta.url), "utf8");
  assert.doesNotMatch(css, /\.brand-mark\s+i(?:\b|:)/);
  assert.match(css, /\.brand-mark\s*\{[^}]*display:\s*block\s*;[^}]*width:\s*27px\s*;[^}]*height:\s*27px\s*;[^}]*border-radius:\s*8px\s*;[^}]*\}/s);
  assert.match(css, /@media \(max-width: 720px\)[\s\S]*?\.brand-mark\s*\{[^}]*width:\s*24px\s*;[^}]*height:\s*24px\s*;[^}]*\}/);
});

test("the explicit asset manifests contain the exact resolvable Signal M asset set", async () => {
  assert.deepEqual(brandCopiedAssetPaths, brandAssetPaths);
  assert.deepEqual(copiedAssetPaths.filter((path) => path.startsWith("brand/")), brandAssetPaths);
  assert.deepEqual(assetPaths.filter((path) => path.startsWith("brand/")), brandAssetPaths);

  for (const relativePath of brandAssetPaths) await readFile(new URL(`site/assets/${relativePath}`, repositoryRoot));

  const { documents } = await prepareSite();
  for (const [path, html] of documents) {
    for (const match of html.matchAll(/(?:src|href)="(\/assets\/brand\/[^"?#]+)"/g)) {
      await assert.doesNotReject(readFile(new URL(`site${match[1]}`, repositoryRoot)), `${path}: ${match[1]}`);
    }
  }
});

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
    ["escaped URL with entities", `<svg xmlns="http://www.w3.org/2000/svg"><circle fill="u\\72l(&#47;&#47;example.com/paint.svg#x)"/></svg>`],
    ["data URL", `<svg xmlns="http://www.w3.org/2000/svg"><title>data:image/svg+xml,unsafe</title></svg>`],
    ["HTTP URL", `<svg xmlns="http://www.w3.org/2000/svg"><title>https://example.com</title></svg>`],
  ];

  for (const [entry, svg] of maliciousSVGs) {
    assert.throws(() => assertSelfContainedSVG(svg), undefined, `${entry} must be rejected`);
  }

  assert.throws(
    () => assertSelfContainedSVG(`<svg xmlns="http://www.w3.org/2000/svg"><circle fill="\\23 7196FF"/></svg>`),
    /backslashes are not allowed/,
  );
  assert.throws(
    () => assertSelfContainedSVG(`<svg xmlns="http://www.w3.org/2000/svg"><circle fill="&#35;7196FF"/></svg>`),
    /character entities are not allowed/,
  );
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

test("canonical brand geometry stays synchronized across both social-card render sources", async () => {
  const canonical = await readFile(canonicalURL, "utf8");
  const capture = await readFile(socialCardURL, "utf8");
  const renderedCards = [
    renderSocialCardSVG({ project, locale: "en" }),
    renderSocialCardSVG({ project, locale: "zh-Hans" }),
  ];
  const geometry = canonical.match(/<(?:rect|path|circle)\b[^>]*\/>/g) ?? [];

  assert.equal(geometry.length, 4);
  assert.match(capture, /<img class="mark" src="\.\/assets\/brand\/macmlx-mark\.svg" alt="">/);
  for (const element of geometry) assert.ok(!capture.includes(element), `HTML capture must reference rather than copy ${element}`);
  for (const sourceSVG of [canonical, ...renderedCards]) {
    for (const element of geometry) assert.ok(sourceSVG.includes(element), `Signal M drifted at ${element}`);
  }
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

test("tracked raster icons are strict PNGs at their exact target dimensions", async () => {
  const canonical = await readFile(canonicalURL);
  const expectedSourceDigest = brandIconSourceDigest(canonical);
  const dimensions = [];
  for (const icon of rasterIcons) {
    const png = await readFile(new URL(`../assets/brand/${icon.filename}`, import.meta.url));
    const header = inspectPNG(png, icon.filename);
    assert.doesNotThrow(() => validateBrandIconPNG(png, icon.width, icon.filename, { expectedSourceDigest }));
    const digestChunks = pngChunks(png).filter(({ type, data }) => type === "tEXt" && data.toString("latin1").startsWith(`${brandDigestKeyword}\0`));
    assert.equal(digestChunks.length, 1, `${icon.filename} must contain exactly one source digest`);
    assert.equal(digestChunks[0].data.toString("latin1"), `${brandDigestKeyword}\0${expectedSourceDigest}`);
    assert.deepEqual({ width: header.width, height: header.height }, { width: icon.width, height: icon.height });
    dimensions.push(`${header.width}x${header.height}`);
  }
  assert.equal(new Set(dimensions).size, rasterIcons.length, "each tracked icon must carry its own dimensions in IHDR");
});

test("brand PNG validation rejects missing, duplicate, malformed, stale, and corrupt source digests", async () => {
  const icon = rasterIcons[0];
  const png = await readFile(new URL(`../assets/brand/${icon.filename}`, import.meta.url));
  const expectedSourceDigest = brandIconSourceDigest(await readFile(canonicalURL));
  const chunks = pngChunks(png);
  const isDigest = ({ type, data }) => type === "tEXt" && data.toString("latin1").startsWith(`${brandDigestKeyword}\0`);
  const without = chunks.filter((chunk) => !isDigest(chunk));
  const digest = chunks.find(isDigest);
  const fixtures = [
    [assemblePNG(without), /exactly one source digest/i],
    [assemblePNG([chunks[0], digest, ...chunks.slice(1)]), /exactly one source digest/i],
    [assemblePNG([chunks[0], { type: "tEXt", data: Buffer.from(`${brandDigestKeyword}\0NOT-HEX`, "latin1") }, ...without.slice(1)]), /64 lowercase hexadecimal/i],
    [assemblePNG([chunks[0], { type: "tEXt", data: Buffer.from(`${brandDigestKeyword}\0${"0".repeat(64)}`, "latin1") }, ...without.slice(1)]), /does not match the current canonical SVG/i],
  ];
  const corruptCRC = Buffer.from(png);
  const digestOffset = png.indexOf(Buffer.from(`${brandDigestKeyword}\0`, "latin1"));
  assert.ok(digestOffset > 0);
  corruptCRC[digestOffset + brandDigestKeyword.length + 1] ^= 1;
  fixtures.push([corruptCRC, /CRC mismatch/i]);

  for (const [fixture, expected] of fixtures) {
    assert.throws(() => validateBrandIconPNG(fixture, icon.width, icon.filename, { expectedSourceDigest }), expected);
  }
});

test("brand PNG validation caps decompression at the exact scanline budget", async () => {
  const icon = rasterIcons[0];
  const png = await readFile(new URL(`../assets/brand/${icon.filename}`, import.meta.url));
  const chunks = pngChunks(png);
  const expectedScanlineBytes = (1 + (icon.width * 4)) * icon.height;
  const bomb = assemblePNG([
    chunks.find(({ type }) => type === "IHDR"),
    ...chunks.filter(({ type }) => type === "tEXt"),
    { type: "IDAT", data: deflateSync(Buffer.alloc(expectedScanlineBytes + 1)) },
    chunks.find(({ type }) => type === "IEND"),
  ]);
  assert.throws(
    () => validateBrandIconPNG(bomb, icon.width, icon.filename),
    /decompressed PNG data exceeds the expected scanline size/i,
  );
});

test("normal build validation rejects tracked brand PNG drift without Sharp", async (t) => {
  const assetRoot = await mkdtemp(join(tmpdir(), "macmlx-brand-source-drift-"));
  t.after(() => rm(assetRoot, { recursive: true, force: true }));
  await mkdir(join(assetRoot, "brand"), { recursive: true });
  for (const name of ["macmlx-mark.svg", ...rasterIcons.map(({ filename }) => filename)]) {
    await writeFile(join(assetRoot, "brand", name), await readFile(new URL(`../assets/brand/${name}`, import.meta.url)));
  }
  await assert.doesNotReject(validateBrandAssets(assetRoot));
  await writeFile(join(assetRoot, "brand/macmlx-mark.svg"), `${await readFile(canonicalURL, "utf8")}\n<!-- drift -->\n`);
  await assert.rejects(validateBrandAssets(assetRoot), /source digest does not match the current canonical SVG/i);
});

test("PNG validation rejects invalid critical chunks and noncontiguous image phases", async (t) => {
  const baseChunks = pngChunks(await readFile(new URL("../assets/brand/apple-touch-icon.png", import.meta.url)));
  const header = baseChunks.find(({ type }) => type === "IHDR");
  const imageData = { type: "IDAT", data: Buffer.concat(baseChunks.filter(({ type }) => type === "IDAT").map(({ data }) => data)) };
  const end = baseChunks.find(({ type }) => type === "IEND");
  const split = Math.floor(imageData.data.length / 2);
  const fixtures = [
    ["unknown critical chunk", assemblePNG([header, { type: "ABCD", data: Buffer.alloc(0) }, imageData, end]), /unknown critical chunk ABCD/],
    ["reserved-bit chunk", assemblePNG([header, { type: "abca", data: Buffer.alloc(0) }, imageData, end]), /reserved bit/i],
    ["empty PLTE", assemblePNG([header, { type: "PLTE", data: Buffer.alloc(0) }, imageData, end]), /PLTE length/],
    ["mis-sized PLTE", assemblePNG([header, { type: "PLTE", data: Buffer.alloc(4) }, imageData, end]), /PLTE length/],
    ["oversized PLTE", assemblePNG([header, { type: "PLTE", data: Buffer.alloc(771) }, imageData, end]), /PLTE length/],
    ["PLTE after IDAT", assemblePNG([header, imageData, { type: "PLTE", data: Buffer.from([0, 0, 0]) }, end]), /PLTE.*before IDAT/],
    ["interrupted IDAT", assemblePNG([
      header,
      { type: "IDAT", data: imageData.data.subarray(0, split) },
      { type: "tEXt", data: Buffer.from("key\0value") },
      { type: "IDAT", data: imageData.data.subarray(split) },
      end,
    ]), /IDAT chunks must be contiguous/],
  ];

  for (const [name, fixture, expected] of fixtures) {
    assert.throws(() => inspectPNG(fixture, name), expected);
    await t.test(`${name} is rejected before publication`, async (t) => {
      const outputDirectory = await mkdtemp(join(tmpdir(), "macmlx-brand-invalid-png-"));
      t.after(() => rm(outputDirectory, { recursive: true, force: true }));
      await assert.rejects(renderBrandIcons({ outputDirectory, sharpImpl: () => bufferedSharp(fixture) }), expected);
    });
  }
});

test("web manifest has only the exact macMLX install contract", async () => {
  const manifest = JSON.parse(await readFile(manifestURL, "utf8"));
  assert.deepEqual(manifest, installManifestContract);
});

test("Sharp rerenders all brand icons byte-identically", { skip: !sharpIsAvailable() }, async (t) => {
  const firstDirectory = await mkdtemp(join(tmpdir(), "macmlx-brand-first-"));
  const secondDirectory = await mkdtemp(join(tmpdir(), "macmlx-brand-second-"));
  t.after(() => Promise.all([firstDirectory, secondDirectory].map((path) => rm(path, { recursive: true, force: true }))));

  await renderBrandIcons({ outputDirectory: firstDirectory });
  await renderBrandIcons({ outputDirectory: secondDirectory });
  for (const icon of rasterIcons) {
    const firstHash = createHash("sha256").update(await readFile(join(firstDirectory, icon.filename))).digest("hex");
    const secondHash = createHash("sha256").update(await readFile(join(secondDirectory, icon.filename))).digest("hex");
    const trackedHash = createHash("sha256").update(await readFile(new URL(`../assets/brand/${icon.filename}`, import.meta.url))).digest("hex");
    assert.equal(firstHash, secondHash, `${icon.filename} must be deterministic`);
    assert.equal(firstHash, trackedHash, `${icon.filename} fresh render must match the tracked PNG`);
  }
});

test("brand icon refresh rolls back the complete set when a later render fails", async (t) => {
  for (const failureAt of [2, 3]) {
    await t.test(`render ${failureAt} fails`, async (t) => {
      const outputDirectory = await mkdtemp(join(tmpdir(), `macmlx-brand-atomic-${failureAt}-`));
      t.after(() => rm(outputDirectory, { recursive: true, force: true }));
      const originals = new Map(rasterIcons.map((icon, index) => [icon.filename, Buffer.from(`original-${failureAt}-${index}`)]));
      for (const [filename, contents] of originals) await writeFile(join(outputDirectory, filename), contents);
      await writeFile(join(outputDirectory, "unrelated.svg"), "keep me");

      let renders = 0;
      function failingSharp() {
        return {
          resize(width, height) {
            return {
              png() {
                return {
                  async toBuffer() {
                    renders += 1;
                    if (renders === failureAt) throw new Error(`injected render-${failureAt} failure`);
                    return solidPNG(width, height);
                  },
                };
              },
            };
          },
        };
      }

      await assert.rejects(
        renderBrandIcons({ outputDirectory, sharpImpl: failingSharp }),
        (error) => {
          assert.ok(error instanceof AggregateError);
          assert.match(error.message, new RegExp(`injected render-${failureAt} failure`));
          assert.equal(error.cause?.message, `injected render-${failureAt} failure`);
          assert.ok(error.errors.includes(error.cause));
          return true;
        },
      );
      for (const [filename, contents] of originals) assert.deepEqual(await readFile(join(outputDirectory, filename)), contents);
      assert.equal(await readFile(join(outputDirectory, "unrelated.svg"), "utf8"), "keep me");
      assert.deepEqual((await readdir(outputDirectory)).sort(), [...originals.keys(), "unrelated.svg"].sort());
    });
  }
});

test("brand icon publication restores every preexisting-state shape after a staged rename fails", async (t) => {
  const scenarios = [
    ["complete", rasterIcons.map((icon) => icon.filename)],
    ["partial", [rasterIcons[0].filename, rasterIcons[2].filename]],
    ["missing", []],
  ];
  for (const [state, existing] of scenarios) {
    for (const failureAt of [2, 3]) {
      await t.test(`${state} set with publish rename ${failureAt} failing`, async (t) => {
        const outputDirectory = await mkdtemp(join(tmpdir(), `macmlx-brand-publish-${state}-${failureAt}-`));
        t.after(() => rm(outputDirectory, { recursive: true, force: true }));
        const originals = new Map(existing.map((filename, index) => [filename, Buffer.from(`${state}-original-${index}`)]));
        for (const [filename, contents] of originals) await writeFile(join(outputDirectory, filename), contents);
        await writeFile(join(outputDirectory, "unrelated.svg"), "keep me");

        let publishes = 0;
        const fsImpl = {
          async rename(source, destination) {
            if (source.includes(".brand-icons-stage-")) {
              publishes += 1;
              if (publishes === failureAt) throw new Error(`injected publish-${failureAt} failure`);
            }
            return rename(source, destination);
          },
        };

        await assert.rejects(
          renderBrandIcons({ outputDirectory, sharpImpl: solidSharp, fsImpl }),
          new RegExp(`injected publish-${failureAt} failure`),
        );
        await assertOriginalIconState(outputDirectory, originals);
      });
    }
  }
});

test("an incomplete rollback reports both failures and preserves the remaining backup", async (t) => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "macmlx-brand-rollback-failure-"));
  t.after(() => rm(outputDirectory, { recursive: true, force: true }));
  const originals = new Map(rasterIcons.map((icon, index) => [icon.filename, Buffer.from(`rollback-original-${index}`)]));
  for (const [filename, contents] of originals) await writeFile(join(outputDirectory, filename), contents);
  await writeFile(join(outputDirectory, "unrelated.svg"), "keep me");

  const publishFailure = new Error("injected publish failure");
  const rollbackFailure = new Error("injected rollback failure");
  let publishes = 0;
  const fsImpl = {
    async rename(source, destination) {
      if (source.includes(".brand-icons-stage-")) {
        publishes += 1;
        if (publishes === 2) throw publishFailure;
      }
      if (source.includes(".brand-icons-backup-") && source.endsWith("icon-192.png")) throw rollbackFailure;
      return rename(source, destination);
    },
  };

  await assert.rejects(
    renderBrandIcons({ outputDirectory, sharpImpl: solidSharp, fsImpl }),
    (error) => {
      assert.ok(error instanceof AggregateError);
      assert.equal(error.cause, publishFailure);
      assert.ok(error.errors.includes(publishFailure));
      assert.ok(error.errors.includes(rollbackFailure));
      assert.match(error.message, /rollback was incomplete/i);
      assert.match(error.message, /\.brand-icons-backup-/);
      return true;
    },
  );

  const backupNames = (await readdir(outputDirectory)).filter((name) => name.startsWith(".brand-icons-backup-"));
  assert.equal(backupNames.length, 1);
  assert.deepEqual(
    await readFile(join(outputDirectory, backupNames[0], "icon-192.png")),
    originals.get("icon-192.png"),
  );
  assert.equal(await readFile(join(outputDirectory, "unrelated.svg"), "utf8"), "keep me");
  assert.equal((await readdir(outputDirectory)).some((name) => name.startsWith(".brand-icons-stage-")), false);
});

test("publish, rollback, staging cleanup, and lock release failures remain jointly actionable", async (t) => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "macmlx-brand-cleanup-failure-"));
  t.after(() => rm(outputDirectory, { recursive: true, force: true }));
  const originals = new Map(rasterIcons.map((icon, index) => [icon.filename, Buffer.from(`cleanup-original-${index}`)]));
  for (const [filename, contents] of originals) await writeFile(join(outputDirectory, filename), contents);
  await writeFile(join(outputDirectory, "unrelated.svg"), "keep me");

  const publishFailure = new Error("injected compound publish failure");
  const rollbackFailure = new Error("injected compound rollback failure");
  const stagingCleanupFailure = new Error("injected staging cleanup failure");
  const releaseFailure = new Error("injected lock release failure");
  const cleanupAttempts = [];
  let publishes = 0;
  const fsImpl = {
    async rename(source, destination) {
      if (source.includes(".brand-icons-stage-")) {
        publishes += 1;
        if (publishes === 2) throw publishFailure;
      }
      if (source.includes(".brand-icons-backup-") && source.endsWith("icon-192.png")) throw rollbackFailure;
      return rename(source, destination);
    },
    async rm(path, options) {
      cleanupAttempts.push(path);
      if (path.includes(".brand-icons-stage-")) throw stagingCleanupFailure;
      if (path.endsWith(".brand-icons.render.lock")) throw releaseFailure;
      return rm(path, options);
    },
  };

  await assert.rejects(
    renderBrandIcons({ outputDirectory, sharpImpl: solidSharp, fsImpl }),
    (error) => {
      assert.ok(error instanceof AggregateError);
      assert.equal(error.cause, publishFailure);
      for (const failure of [publishFailure, rollbackFailure, stagingCleanupFailure, releaseFailure]) {
        assert.ok(error.errors.includes(failure), `${failure.message} must be retained`);
      }
      assert.match(error.message, /rollback was incomplete/i);
      assert.match(error.message, /\.brand-icons-backup-/);
      return true;
    },
  );

  assert.equal(cleanupAttempts.some((path) => path.includes(".brand-icons-stage-")), true);
  assert.equal(cleanupAttempts.some((path) => path.endsWith(".brand-icons.render.lock")), true);
  const backupNames = (await readdir(outputDirectory)).filter((name) => name.startsWith(".brand-icons-backup-"));
  assert.equal(backupNames.length, 1);
  assert.deepEqual(await readFile(join(outputDirectory, backupNames[0], "icon-192.png")), originals.get("icon-192.png"));
  assert.equal(await readFile(join(outputDirectory, "unrelated.svg"), "utf8"), "keep me");
});

test("brand icon writer lock excludes overlap and releases for the next render", async (t) => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "macmlx-brand-lock-"));
  t.after(() => rm(outputDirectory, { recursive: true, force: true }));
  let announceStarted;
  let unblock;
  const started = new Promise((resolve) => { announceStarted = resolve; });
  const gate = new Promise((resolve) => { unblock = resolve; });
  let blocked = true;

  function blockingSharp() {
    return {
      resize(width, height) {
        return {
          png() {
            return {
              async toBuffer() {
                if (blocked) {
                  blocked = false;
                  announceStarted();
                  await gate;
                }
                return solidPNG(width, height);
              },
            };
          },
        };
      },
    };
  }

  const firstRender = renderBrandIcons({ outputDirectory, sharpImpl: blockingSharp });
  await started;
  await assert.rejects(
    renderBrandIcons({ outputDirectory, sharpImpl: solidSharp }),
    /Brand-icon render lock exists/,
  );
  unblock();
  await firstRender;

  await renderBrandIcons({ outputDirectory, sharpImpl: solidSharp });
  assert.equal((await readdir(outputDirectory)).some((name) => name === ".brand-icons.render.lock"), false);
});

test("brand icon writer lock never automatically reclaims an old terminated owner", async (t) => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "macmlx-brand-stale-lock-"));
  t.after(() => rm(outputDirectory, { recursive: true, force: true }));
  const lockDirectory = join(outputDirectory, ".brand-icons.render.lock");
  await mkdir(lockDirectory);
  const owner = {
    pid: 2_147_483_647,
    startedAt: "2000-01-01T00:00:00.000Z",
    token: "terminated-owner",
  };
  await writeFile(join(lockDirectory, "owner.json"), JSON.stringify(owner));

  await assert.rejects(
    renderBrandIcons({ outputDirectory, sharpImpl: solidSharp }),
    (error) => {
      assert.match(error.message, new RegExp(lockDirectory.replaceAll("/", "\\/")));
      assert.match(error.message, /terminated-owner/);
      assert.match(error.message, /manually inspect and remove/i);
      return true;
    },
  );
  assert.deepEqual(JSON.parse(await readFile(join(lockDirectory, "owner.json"), "utf8")), owner);
});

test("brand icon writer releases only the lock token it acquired", async (t) => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "macmlx-brand-lock-owner-"));
  t.after(() => rm(outputDirectory, { recursive: true, force: true }));
  const lockDirectory = join(outputDirectory, ".brand-icons.render.lock");
  let announceStarted;
  let unblock;
  const started = new Promise((resolve) => { announceStarted = resolve; });
  const gate = new Promise((resolve) => { unblock = resolve; });
  let blocked = true;
  function blockingSharp() {
    return {
      resize(width, height) {
        return {
          png() {
            return {
              async toBuffer() {
                if (blocked) {
                  blocked = false;
                  announceStarted();
                  await gate;
                }
                return solidPNG(width, height);
              },
            };
          },
        };
      },
    };
  }

  const render = renderBrandIcons({ outputDirectory, sharpImpl: blockingSharp });
  await started;
  const replacementOwner = { pid: process.pid, startedAt: new Date().toISOString(), token: "replacement-owner" };
  await writeFile(join(lockDirectory, "owner.json"), JSON.stringify(replacementOwner));
  unblock();
  await render;

  assert.deepEqual(JSON.parse(await readFile(join(lockDirectory, "owner.json"), "utf8")), replacementOwner);
});
