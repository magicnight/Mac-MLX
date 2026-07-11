import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { inflateSync } from "node:zlib";

const require = createRequire(import.meta.url);
const icons = Object.freeze([
  Object.freeze({ filename: "apple-touch-icon.png", size: 180 }),
  Object.freeze({ filename: "icon-192.png", size: 192 }),
  Object.freeze({ filename: "icon-512.png", size: 512 }),
]);

function filesystemPath(value) {
  return value instanceof URL ? fileURLToPath(value) : resolve(value);
}

async function acquireRenderLock(outputPath, filesystem) {
  const lockDirectory = join(outputPath, ".brand-icons.render.lock");
  const token = randomUUID();
  try {
    await filesystem.mkdir(lockDirectory);
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    let ownerMetadata;
    try {
      ownerMetadata = await filesystem.readFile(join(lockDirectory, "owner.json"), "utf8");
    } catch (ownerError) {
      ownerMetadata = `<unavailable: ${ownerError.code ?? ownerError.message}>`;
    }
    throw new Error(
      `Brand-icon render lock exists at ${lockDirectory}. Owner metadata: ${ownerMetadata}. Manually inspect and remove the lock only after confirming no renderer is active.`,
      { cause: error },
    );
  }
  try {
    await filesystem.writeFile(
      join(lockDirectory, "owner.json"),
      JSON.stringify({ pid: process.pid, startedAt: new Date().toISOString(), token }),
      "utf8",
    );
  } catch (error) {
    await filesystem.rm(lockDirectory, { recursive: true, force: true });
    throw error;
  }

  let released = false;
  return async function releaseRenderLock() {
    if (released) return;
    released = true;
    let owner;
    try {
      owner = JSON.parse(await filesystem.readFile(join(lockDirectory, "owner.json"), "utf8"));
    } catch (error) {
      if (error.code === "ENOENT" || error instanceof SyntaxError) return;
      throw error;
    }
    if (owner.token === token) await filesystem.rm(lockDirectory, { recursive: true, force: true });
  };
}

async function loadSharp() {
  try {
    const module = await import("sharp");
    return module.default ?? module;
  } catch (projectError) {
    const moduleDirectory = process.env.MACMLX_NODE_MODULES;
    if (moduleDirectory) {
      try {
        const module = require(require.resolve("sharp", { paths: [resolve(moduleDirectory)] }));
        return module.default ?? module;
      } catch (runtimeError) {
        throw new Error(`Unable to load Sharp from MACMLX_NODE_MODULES=${moduleDirectory}. Point it at a node_modules directory containing Sharp.`, { cause: runtimeError });
      }
    }
    throw new Error("Sharp is required to refresh brand icons. Install it through normal module resolution or set MACMLX_NODE_MODULES to a node_modules directory containing Sharp.", { cause: projectError });
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

function validatePNG(buffer, size, label) {
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (!Buffer.isBuffer(buffer) || !signature.every((byte, index) => buffer[index] === byte)) throw new Error(`${label} is not a PNG`);
  let offset = 8;
  let ihdrCount = 0;
  let iendCount = 0;
  let plteCount = 0;
  let phase = "beforeIHDR";
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
    if (!/^[A-Za-z]{4}$/.test(type)) throw new Error(`${label} has an invalid PNG chunk type`);
    if (!/[A-Z]/.test(type[2])) throw new Error(`${label} PNG chunk ${type} sets the reserved bit`);
    if (crc32(buffer.subarray(typeStart, crcOffset)) !== buffer.readUInt32BE(crcOffset)) throw new Error(`${label} PNG CRC mismatch in ${type}`);
    if (type === "IHDR") {
      ihdrCount += 1;
      if (phase !== "beforeIHDR" || offset !== 8 || length !== 13) throw new Error(`${label} must contain a unique first IHDR chunk`);
      const [bitDepth, colorType, compression, filter, interlace] = buffer.subarray(dataStart + 8, dataStart + 13);
      if (buffer.readUInt32BE(dataStart) !== size || buffer.readUInt32BE(dataStart + 4) !== size) throw new Error(`${label} must be ${size}x${size}`);
      if (bitDepth !== 8 || colorType !== 6 || compression !== 0 || filter !== 0 || interlace !== 0) throw new Error(`${label} must be a non-interlaced 8-bit RGBA PNG`);
      phase = "beforeIDAT";
    } else if (type === "PLTE") {
      plteCount += 1;
      if (plteCount !== 1 || phase !== "beforeIDAT") throw new Error(`${label} PLTE must appear once before IDAT`);
      if (length === 0 || length > 768 || length % 3 !== 0) throw new Error(`${label} has an invalid PLTE length`);
    } else if (type === "IDAT") {
      if (phase === "afterIDAT") throw new Error(`${label} IDAT chunks must be contiguous`);
      if (phase !== "beforeIDAT" && phase !== "inIDAT") throw new Error(`${label} IDAT cannot appear in the ${phase} phase`);
      if (length === 0) throw new Error(`${label} has an empty IDAT chunk`);
      compressed.push(buffer.subarray(dataStart, crcOffset));
      phase = "inIDAT";
    } else if (type === "IEND") {
      iendCount += 1;
      if ((phase !== "inIDAT" && phase !== "afterIDAT") || length !== 0 || end !== buffer.length) throw new Error(`${label} has an invalid terminal IEND chunk`);
      phase = "ended";
    } else {
      if (/^[A-Z]/.test(type)) throw new Error(`${label} has unknown critical chunk ${type}`);
      if (phase === "beforeIHDR") throw new Error(`${label} IHDR must be the first chunk`);
      if (phase === "inIDAT") phase = "afterIDAT";
    }
    offset = end;
  }
  if (ihdrCount !== 1 || compressed.length === 0 || iendCount !== 1 || phase !== "ended") throw new Error(`${label} is missing required PNG chunks`);
  let scanlines;
  try {
    scanlines = inflateSync(Buffer.concat(compressed));
  } catch {
    throw new Error(`${label} has invalid compressed PNG data`);
  }
  const rowLength = 1 + (size * 4);
  if (scanlines.length !== rowLength * size) throw new Error(`${label} has inconsistent decompressed scanlines`);
  for (let row = 0; row < size; row += 1) {
    if (scanlines[row * rowLength] > 4) throw new Error(`${label} has an invalid row filter`);
  }
}

export async function renderBrandIcons({
  source = new URL("../site/assets/brand/macmlx-mark.svg", import.meta.url),
  outputDirectory = new URL("../site/assets/brand/", import.meta.url),
  sharpImpl,
  fsImpl,
} = {}) {
  const filesystem = { mkdir, readFile, rename, rm, writeFile, ...fsImpl };
  const outputPath = filesystemPath(outputDirectory);
  const sourcePath = filesystemPath(source);
  const unique = `${process.pid}-${randomUUID()}`;
  const stagingDirectory = join(outputPath, `.brand-icons-stage-${unique}`);
  const backupDirectory = join(outputPath, `.brand-icons-backup-${unique}`);
  const lockDirectory = join(outputPath, ".brand-icons.render.lock");
  const published = [];
  const backedUp = [];
  let preserveBackup = false;
  let primaryError;
  const operationErrors = [];
  const cleanupErrors = [];

  await filesystem.mkdir(outputPath, { recursive: true });
  const releaseLock = await acquireRenderLock(outputPath, filesystem);
  try {
    await filesystem.mkdir(stagingDirectory);
    await filesystem.mkdir(backupDirectory);
    const sharp = sharpImpl ?? await loadSharp();
    const svg = await filesystem.readFile(sourcePath);
    for (const icon of icons) {
      const png = await sharp(svg, { density: 384 })
        .resize(icon.size, icon.size, { fit: "fill" })
        .png({ adaptiveFiltering: false, compressionLevel: 9, effort: 10, palette: false })
        .toBuffer();
      validatePNG(png, icon.size, icon.filename);
      await filesystem.writeFile(join(stagingDirectory, icon.filename), png);
    }

    for (const icon of icons) {
      const destination = join(outputPath, icon.filename);
      const backup = join(backupDirectory, icon.filename);
      try {
        await filesystem.rename(destination, backup);
        backedUp.push(icon.filename);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
      await filesystem.rename(join(stagingDirectory, icon.filename), destination);
      published.push(icon.filename);
    }
  } catch (error) {
    primaryError = error;
    operationErrors.push(error);
    const rollbackErrors = [];
    for (const filename of [...published].reverse()) {
      try {
        await filesystem.rm(join(outputPath, filename), { force: true });
      } catch (rollbackError) {
        rollbackErrors.push(rollbackError);
      }
    }
    for (const filename of [...backedUp].reverse()) {
      try {
        await filesystem.rename(join(backupDirectory, filename), join(outputPath, filename));
      } catch (rollbackError) {
        rollbackErrors.push(rollbackError);
      }
    }
    if (rollbackErrors.length > 0) {
      preserveBackup = true;
      operationErrors.push(...rollbackErrors);
    }
  }

  try {
    await filesystem.rm(stagingDirectory, { recursive: true, force: true });
  } catch (error) {
    cleanupErrors.push(error);
  }
  if (!preserveBackup) {
    try {
      await filesystem.rm(backupDirectory, { recursive: true, force: true });
    } catch (error) {
      cleanupErrors.push(error);
    }
  }
  try {
    await releaseLock();
  } catch (error) {
    cleanupErrors.push(error);
  }

  if (primaryError) {
    const message = preserveBackup
      ? `${primaryError.message}. Brand icon publication failed and rollback was incomplete. Recover remaining originals from ${backupDirectory}; inspect cleanup state including ${lockDirectory}.`
      : cleanupErrors.length > 0
        ? `${primaryError.message}. Brand icon rendering or publication failed and cleanup was incomplete. Inspect ${backupDirectory} and ${lockDirectory}.`
        : `Brand icon rendering or publication failed: ${primaryError.message}`;
    throw new AggregateError([...operationErrors, ...cleanupErrors], message, { cause: primaryError });
  }
  if (cleanupErrors.length > 0) {
    throw new AggregateError(
      cleanupErrors,
      `Brand icons were published but cleanup was incomplete. Inspect ${backupDirectory} and ${lockDirectory}.`,
      { cause: cleanupErrors[0] },
    );
  }

  console.log(`Rendered ${icons.length} deterministic brand PNGs from ${basename(sourcePath)}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await renderBrandIcons();
