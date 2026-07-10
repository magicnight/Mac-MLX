import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { project } from "../content/project.mjs";
import { renderSocialCardSVG, socialCaptureInstructions, socialCardCaptures, validateSocialPNG } from "../lib/social-card.mjs";
import { copySocialAssets } from "../../scripts/build-public-site.mjs";
import { renderSocialCards } from "../../scripts/render-social-cards.mjs";

const source = await readFile(new URL("../social-card.html", import.meta.url), "utf8");

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunks(png) {
  const chunks = [];
  for (let offset = 8; offset < png.length;) {
    const length = png.readUInt32BE(offset);
    const end = offset + 12 + length;
    chunks.push({ type: png.toString("ascii", offset + 4, offset + 8), offset, end, length });
    offset = end;
  }
  return chunks;
}

function removeChunks(png, type) {
  return Buffer.concat([png.subarray(0, 8), ...pngChunks(png).filter((chunk) => chunk.type !== type).map((chunk) => png.subarray(chunk.offset, chunk.end))]);
}

function mutateChunk(png, type, mutate) {
  const result = Buffer.from(png);
  const chunk = pngChunks(result).find((item) => item.type === type);
  assert.ok(chunk, `missing ${type} fixture chunk`);
  const dataStart = chunk.offset + 8;
  mutate(result.subarray(dataStart, dataStart + chunk.length));
  result.writeUInt32BE(crc32(result.subarray(chunk.offset + 4, dataStart + chunk.length)), dataStart + chunk.length);
  return result;
}

test("social card source is a deterministic 1200 by 630 no-network surface", () => {
  assert.match(source, /width:\s*1200px/);
  assert.match(source, /height:\s*630px/);
  assert.match(source, /new URLSearchParams\(window\.location\.search\)/);
  assert.match(source, /import \{ project \} from "\.\/content\/project\.mjs"/);
  assert.match(source, /project\.currentVersion/);
  assert.doesNotMatch(source, /https?:\/\//);
  assert.doesNotMatch(source, /<(?:img|link|script)[^>]+(?:src|href)="(?:https?:)?\/\//i);
});

test("social card source contains reviewed locale copy and uses registry version", () => {
  assert.match(source, /Native Swift inference/);
  assert.match(source, /原生 Swift 推理/);
  assert.match(source, /Apple Silicon/);
  assert.match(source, /Apple 芯片/);
  assert.doesNotMatch(source, new RegExp(`v${project.currentVersion.replaceAll(".", "\\.")}`));
});

test("capture contract names the exact tracked source and public targets", () => {
  assert.deepEqual(socialCardCaptures, [
    { locale: "en", query: "?locale=en", source: "site/assets/social/og-en.png", output: "public/assets/social/og-en.png" },
    { locale: "zh-Hans", query: "?locale=zh", source: "site/assets/social/og-zh.png", output: "public/assets/social/og-zh.png" },
  ]);
  assert.match(socialCaptureInstructions, /node scripts\/render-social-cards\.mjs/);
  assert.match(socialCaptureInstructions, /Sharp/);
  assert.match(socialCaptureInstructions, /MACMLX_NODE_MODULES/);
  assert.doesNotMatch(socialCaptureInstructions, /sips/i);
});

test("registry-driven vector cards contain locale copy, dimensions, and no network assets", () => {
  const english = renderSocialCardSVG({ project, locale: "en" });
  const chinese = renderSocialCardSVG({ project, locale: "zh-Hans" });
  for (const svg of [english, chinese]) {
    assert.match(svg, /<svg[^>]+width="1200"[^>]+height="630"/);
    assert.match(svg, new RegExp(`v${project.currentVersion.replaceAll(".", "\\.")}`));
    assert.doesNotMatch(svg, /(?:href|src)="https?:\/\//i);
    assert.doesNotMatch(svg, /<image\b/i);
  }
  assert.match(english, /Native Swift inference/);
  assert.match(chinese, /原生 Swift 推理/);
});

test("tracked social cards are distinct real 1200 by 630 PNGs", async () => {
  const digests = [];
  for (const capture of socialCardCaptures) {
    const png = await readFile(new URL(`../../${capture.source}`, import.meta.url));
    assert.doesNotThrow(() => validateSocialPNG(png, capture.source));
    digests.push(createHash("sha256").update(png).digest("hex"));
  }
  assert.notEqual(digests[0], digests[1]);
});

test("PNG validation rejects structural, checksum, and compressed-data corruption", async () => {
  const png = await readFile(new URL("../assets/social/og-en.png", import.meta.url));
  const badCRC = Buffer.from(png);
  badCRC[20] ^= 1;
  const invalidCompressed = mutateChunk(png, "IDAT", (data) => { data[Math.floor(data.length / 2)] ^= 1; });
  const invalidHeader = mutateChunk(png, "IHDR", (data) => { data[8] = 8; data[9] = 2; });
  const fixtures = [
    [png.subarray(0, 20), /truncated PNG chunk|missing/],
    [badCRC, /CRC mismatch/],
    [removeChunks(png, "IDAT"), /IDAT/],
    [removeChunks(png, "IEND"), /IEND/],
    [invalidCompressed, /invalid compressed PNG data/],
    [invalidHeader, /RGBA|color type/],
    [Buffer.concat([png, Buffer.from([0])]), /trailing bytes/],
  ];
  for (const [fixture, expected] of fixtures) assert.throws(() => validateSocialPNG(fixture, "fixture"), expected);
});

test("build social copy preserves the exact tracked PNG bytes", async (t) => {
  const destination = await mkdtemp(join(tmpdir(), "macmlx-social-copy-"));
  t.after(() => rm(destination, { recursive: true, force: true }));
  await copySocialAssets(destination);
  for (const capture of socialCardCaptures) {
    const sourcePNG = await readFile(new URL(`../../${capture.source}`, import.meta.url));
    const copiedPNG = await readFile(join(destination, capture.output.replace(/^public\//, "")));
    assert.deepEqual(copiedPNG, sourcePNG);
  }
});

test("Sharp rerenders are byte-identical when an explicit runtime is provided", { skip: !process.env.MACMLX_NODE_MODULES }, async (t) => {
  const firstDirectory = await mkdtemp(join(tmpdir(), "macmlx-social-first-"));
  const secondDirectory = await mkdtemp(join(tmpdir(), "macmlx-social-second-"));
  t.after(() => Promise.all([firstDirectory, secondDirectory].map((path) => rm(path, { recursive: true, force: true }))));
  await renderSocialCards({ outputDirectory: firstDirectory });
  await renderSocialCards({ outputDirectory: secondDirectory });
  for (const capture of socialCardCaptures) {
    const filename = capture.source.split("/").at(-1);
    assert.deepEqual(await readFile(join(firstDirectory, filename)), await readFile(join(secondDirectory, filename)));
  }
});

test("social card refresh rolls back the pair when the second render fails", async (t) => {
  const parent = await mkdtemp(join(tmpdir(), "macmlx-social-atomic-"));
  const outputDirectory = join(parent, "social");
  await mkdir(outputDirectory);
  const originals = new Map([
    ["og-en.png", Buffer.from("original english")],
    ["og-zh.png", Buffer.from("original chinese")],
  ]);
  for (const [filename, content] of originals) await writeFile(join(outputDirectory, filename), content);
  t.after(() => rm(parent, { recursive: true, force: true }));
  const validPNG = await readFile(new URL("../assets/social/og-en.png", import.meta.url));

  await assert.rejects(
    renderSocialCards({
      outputDirectory,
      render: async ({ capture }) => {
        if (capture.locale === "zh-Hans") throw new Error("injected second-card failure");
        return validPNG;
      },
    }),
    /injected second-card failure/,
  );

  for (const [filename, content] of originals) assert.deepEqual(await readFile(join(outputDirectory, filename)), content);
  assert.deepEqual((await readdir(parent)).sort(), ["social"]);
});
