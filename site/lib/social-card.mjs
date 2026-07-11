import { inflateSync } from "node:zlib";

export const socialCardCaptures = Object.freeze([
  Object.freeze({ locale: "en", query: "?locale=en", source: "site/assets/social/og-en.png", output: "public/assets/social/og-en.png" }),
  Object.freeze({ locale: "zh-Hans", query: "?locale=zh", source: "site/assets/social/og-zh.png", output: "public/assets/social/og-zh.png" }),
]);

export const socialCaptureInstructions = "Run node scripts/render-social-cards.mjs with Sharp available through normal module resolution, or set MACMLX_NODE_MODULES to a node_modules directory containing Sharp. The command writes registry-driven 1200x630 cards to site/assets/social/og-en.png and site/assets/social/og-zh.png.";

function escapeXML(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&apos;");
}

const cardCopy = Object.freeze({
  en: Object.freeze({ eyebrow: "ONE NATIVE ENGINE", headline: ["Native Swift inference", "for Apple Silicon"], lede: ["A SwiftUI app, CLI, and compatible API", "over one in-process MLX engine."], platform: "Apple Silicon · macOS 14+" }),
  "zh-Hans": Object.freeze({ eyebrow: "一个原生引擎", headline: ["Apple 芯片上的", "原生 Swift 推理"], lede: ["原生 SwiftUI 应用、CLI 与兼容 API，", "共用同一个 Swift 进程内 MLX 引擎。"], platform: "Apple 芯片 · macOS 14+" }),
});

export function renderSocialCardSVG({ project, locale }) {
  const copy = cardCopy[locale];
  if (!copy) throw new Error(`Unsupported social-card locale: ${locale}`);
  const headlineSize = locale === "en" ? 76 : 72;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <radialGradient id="bg" cx="85%" cy="9%" r="78%"><stop offset="0" stop-color="#25412d"/><stop offset=".28" stop-color="#182019"/><stop offset=".72" stop-color="#111311"/></radialGradient>
    <pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse"><path d="M48 0H0V48" fill="none" stroke="#8fa092" stroke-opacity=".13"/></pattern>
    <clipPath id="clip"><rect width="1200" height="630" rx="0"/></clipPath>
  </defs>
  <g clip-path="url(#clip)">
    <rect width="1200" height="630" fill="url(#bg)"/><rect width="1200" height="630" fill="url(#grid)" opacity=".62"/>
    <circle cx="1120" cy="150" r="285" fill="none" stroke="#587060" opacity=".72"/>
    <circle cx="1120" cy="150" r="340" fill="none" stroke="#405248" opacity=".62"/>
    <circle cx="1120" cy="150" r="402" fill="none" stroke="#314039" opacity=".55"/>
    <g transform="translate(78 72) scale(.3515625)">
      <rect x="4" y="4" width="120" height="120" rx="34" fill="#F3F1EA"/>
      <path d="M28 88V39l36 38 36-38v49" fill="none" stroke="#111311" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="64" cy="77" r="8.5" fill="#7196FF"/>
      <circle cx="100" cy="39" r="6" fill="#89E67A"/>
    </g>
    <text x="139" y="105" fill="#f3f1ea" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="31" font-weight="700">macMLX</text>
    <rect x="966" y="75" width="156" height="40" rx="20" fill="#171a17" stroke="#59645b"/>
    <text x="1044" y="101" text-anchor="middle" fill="#c7cec8" font-family="Menlo,monospace" font-size="17" font-weight="600">v${escapeXML(project.currentVersion)}</text>
    <text x="78" y="210" fill="#a9e88b" font-family="Menlo,monospace" font-size="18" font-weight="650" letter-spacing="2">${escapeXML(copy.eyebrow)}</text>
    <text x="78" y="302" fill="#f3f1ea" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,PingFang SC,sans-serif" font-size="${headlineSize}" font-weight="710" letter-spacing="-3">
      <tspan x="78" dy="0">${escapeXML(copy.headline[0])}</tspan><tspan x="78" dy="78">${escapeXML(copy.headline[1])}</tspan>
    </text>
    <text x="78" y="470" fill="#c7cec8" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,PingFang SC,sans-serif" font-size="25" font-weight="400">
      <tspan x="78" dy="0">${escapeXML(copy.lede[0])}</tspan><tspan x="78" dy="36">${escapeXML(copy.lede[1])}</tspan>
    </text>
    <text x="78" y="577" fill="#9ba69e" font-family="Menlo,monospace" font-size="16" font-weight="560">macmlx.app</text>
    <text x="1122" y="577" text-anchor="end" fill="#d9ded9" font-family="Menlo,monospace" font-size="16" font-weight="560">${escapeXML(copy.platform)}</text>
  </g>
</svg>
`;
}

export function validateSocialPNG(buffer, label = "social PNG") {
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (!Buffer.isBuffer(buffer) || !signature.every((byte, index) => buffer[index] === byte)) throw new Error(`${label} is not a PNG`);
  let offset = 8;
  let ihdrCount = 0;
  let idatCount = 0;
  let iendCount = 0;
  const compressed = [];
  while (offset < buffer.length) {
    if (buffer.length - offset < 12) throw new Error(`${label} has a truncated PNG chunk`);
    const length = buffer.readUInt32BE(offset);
    if (length > buffer.length - offset - 12) throw new Error(`${label} has a truncated PNG chunk`);
    const typeStart = offset + 4;
    const dataStart = offset + 8;
    const crcOffset = dataStart + length;
    const end = crcOffset + 4;
    const type = buffer.toString("ascii", typeStart, dataStart);
    if (crc32(buffer.subarray(typeStart, crcOffset)) !== buffer.readUInt32BE(crcOffset)) throw new Error(`${label} PNG CRC mismatch in ${type}`);

    if (type === "IHDR") {
      ihdrCount += 1;
      if (offset !== 8 || length !== 13) throw new Error(`${label} has an invalid IHDR chunk`);
      const width = buffer.readUInt32BE(dataStart);
      const height = buffer.readUInt32BE(dataStart + 4);
      const [bitDepth, colorType, compression, filter, interlace] = buffer.subarray(dataStart + 8, dataStart + 13);
      if (width !== 1200 || height !== 630) throw new Error(`${label} must be 1200x630`);
      if (bitDepth !== 8 || colorType !== 6) throw new Error(`${label} must use 8-bit RGBA color type 6`);
      if (compression !== 0 || filter !== 0 || interlace !== 0) throw new Error(`${label} has unsupported PNG compression, filter, or interlace settings`);
    } else if (type === "IDAT") {
      if (length === 0) throw new Error(`${label} has an empty IDAT chunk`);
      idatCount += 1;
      compressed.push(buffer.subarray(dataStart, crcOffset));
    } else if (type === "IEND") {
      iendCount += 1;
      if (length !== 0) throw new Error(`${label} has an invalid IEND chunk`);
      if (end !== buffer.length) throw new Error(`${label} has trailing bytes after IEND`);
    }
    offset = end;
  }
  if (ihdrCount !== 1) throw new Error(`${label} must contain exactly one IHDR chunk`);
  if (idatCount < 1) throw new Error(`${label} must contain at least one nonempty IDAT chunk`);
  if (iendCount !== 1) throw new Error(`${label} must contain exactly one terminal IEND chunk`);

  let scanlines;
  try {
    scanlines = inflateSync(Buffer.concat(compressed));
  } catch {
    throw new Error(`${label} has invalid compressed PNG data`);
  }
  const rowLength = 1 + (1200 * 4);
  if (scanlines.length !== rowLength * 630) throw new Error(`${label} has an inconsistent decompressed scanline length`);
  for (let row = 0; row < 630; row += 1) {
    if (scanlines[row * rowLength] > 4) throw new Error(`${label} has an invalid PNG row filter`);
  }
}

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
