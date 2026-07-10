export const phaseOneCopiedAssetPaths = Object.freeze([
  "css/main.css",
  "css/no-js.css",
  "images/engine/adaptive-runtime.webp",
  "images/engine/generation-controls.webp",
  "images/engine/mac-silicon-foundation.webp",
  "images/engine/moe-routing.webp",
  "images/engine/paged-kv-memory.webp",
  "js/legacy-language.mjs",
  "js/main.js",
]);

export const phaseOneGeneratedAssetPaths = Object.freeze([
  "og-image.svg",
]);

export const phaseOneAssetPaths = Object.freeze([
  ...phaseOneCopiedAssetPaths,
  ...phaseOneGeneratedAssetPaths,
]);

export const contentHubCopiedAssetPaths = Object.freeze([
  "images/generated/macmlx-inference-pipeline.webp",
  "images/generated/macmlx-shared-core.webp",
  "images/generated/macmlx-unified-memory.webp",
]);

export const copiedAssetPaths = Object.freeze([
  ...phaseOneCopiedAssetPaths,
  ...contentHubCopiedAssetPaths,
]);

export const socialCardAssetPaths = Object.freeze([
  "social/og-en.png",
  "social/og-zh.png",
]);

export const assetPaths = Object.freeze([
  ...copiedAssetPaths,
  ...phaseOneGeneratedAssetPaths,
  ...socialCardAssetPaths,
]);
