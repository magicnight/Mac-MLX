import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { deflateSync, inflateSync } from "node:zlib";
import test from "node:test";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { renderBrandIcons } from "../../scripts/render-brand-icons.mjs";

const canonicalURL = new URL("../assets/brand/macmlx-mark.svg", import.meta.url);
const faviconURL = new URL("../assets/brand/favicon.svg", import.meta.url);
const manifestURL = new URL("../assets/brand/site.webmanifest", import.meta.url);

const rasterIcons = Object.freeze([
  Object.freeze({ filename: "apple-touch-icon.png", width: 180, height: 180 }),
  Object.freeze({ filename: "icon-192.png", width: 192, height: 192 }),
  Object.freeze({ filename: "icon-512.png", width: 512, height: 512 }),
]);

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
    assert.equal(crc32(buffer.subarray(typeStart, crcOffset)), buffer.readUInt32BE(crcOffset), `${label} has a CRC mismatch in ${type}`);

    if (type === "IHDR") {
      ihdrCount += 1;
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
    } else if (type === "IDAT") {
      assert.ok(length > 0, `${label} IDAT must not be empty`);
      compressed.push(buffer.subarray(dataStart, crcOffset));
    } else if (type === "IEND") {
      iendCount += 1;
      assert.equal(length, 0, `${label} IEND must be empty`);
      assert.equal(end, buffer.length, `${label} must not have trailing bytes`);
    }
    offset = end;
  }

  assert.equal(ihdrCount, 1, `${label} must contain one IHDR`);
  assert.equal(iendCount, 1, `${label} must contain one terminal IEND`);
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
  const dimensions = [];
  for (const icon of rasterIcons) {
    const png = await readFile(new URL(`../assets/brand/${icon.filename}`, import.meta.url));
    const header = inspectPNG(png, icon.filename);
    assert.deepEqual({ width: header.width, height: header.height }, { width: icon.width, height: icon.height });
    dimensions.push(`${header.width}x${header.height}`);
  }
  assert.equal(new Set(dimensions).size, rasterIcons.length, "each tracked icon must carry its own dimensions in IHDR");
});

test("web manifest has only the exact macMLX install contract", async () => {
  const manifest = JSON.parse(await readFile(manifestURL, "utf8"));
  assert.deepEqual(manifest, {
    name: "macMLX",
    short_name: "macMLX",
    start_url: "/",
    display: "standalone",
    background_color: "#111311",
    theme_color: "#111311",
    icons: [
      { src: "/assets/brand/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/assets/brand/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
  });
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
    assert.equal(firstHash, secondHash, `${icon.filename} must be deterministic`);
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

      await assert.rejects(renderBrandIcons({ outputDirectory, sharpImpl: failingSharp }), new RegExp(`injected render-${failureAt} failure`));
      for (const [filename, contents] of originals) assert.deepEqual(await readFile(join(outputDirectory, filename)), contents);
      assert.equal(await readFile(join(outputDirectory, "unrelated.svg"), "utf8"), "keep me");
      assert.deepEqual((await readdir(outputDirectory)).sort(), [...originals.keys(), "unrelated.svg"].sort());
    });
  }
});
