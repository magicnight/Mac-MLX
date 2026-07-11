export const pngSourceDigestKeyword = "macMLXSourceSHA256";
export const pngSourceDigestPrefix = Buffer.from(`${pngSourceDigestKeyword}\0`, "latin1");

export function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function isPNGSourceDigestChunk(type, data) {
  return type === "tEXt" && data.subarray(0, pngSourceDigestPrefix.length).equals(pngSourceDigestPrefix);
}

function encodePNGChunk(type, data) {
  const typeBytes = Buffer.from(type, "ascii");
  const chunk = Buffer.alloc(12 + data.length);
  chunk.writeUInt32BE(data.length, 0);
  typeBytes.copy(chunk, 4);
  data.copy(chunk, 8);
  chunk.writeUInt32BE(crc32(Buffer.concat([typeBytes, data])), 8 + data.length);
  return chunk;
}

export function embedPNGSourceDigest(buffer, digest, label = "PNG") {
  if (!Buffer.isBuffer(buffer)) throw new Error(`${label} must be a Buffer`);
  if (!/^[0-9a-f]{64}$/.test(digest)) throw new Error(`${label} source digest must be 64 lowercase hexadecimal characters`);
  const signature = buffer.subarray(0, 8);
  const chunks = [];
  for (let offset = 8; offset < buffer.length;) {
    if (buffer.length - offset < 12) throw new Error(`${label} has a truncated PNG chunk`);
    const length = buffer.readUInt32BE(offset);
    const end = offset + 12 + length;
    if (end > buffer.length) throw new Error(`${label} has a truncated PNG chunk`);
    const type = buffer.toString("ascii", offset + 4, offset + 8);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (!isPNGSourceDigestChunk(type, data)) chunks.push(buffer.subarray(offset, end));
    offset = end;
  }
  if (chunks.length === 0 || chunks[0].toString("ascii", 4, 8) !== "IHDR") throw new Error(`${label} must begin with IHDR`);
  const text = encodePNGChunk("tEXt", Buffer.from(`${pngSourceDigestKeyword}\0${digest}`, "latin1"));
  return Buffer.concat([signature, chunks[0], text, ...chunks.slice(1)]);
}
