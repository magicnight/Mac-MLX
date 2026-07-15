import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { copiedAssetPaths } from "../content/assets.mjs";
import { digestPreparedSite } from "../lib/determinism.mjs";
import { prepareSite } from "../../scripts/build-public-site.mjs";

async function sourceAssets() {
  const entries = [];
  for (const path of copiedAssetPaths) entries.push([`assets/${path}`, await readFile(new URL(`../assets/${path}`, import.meta.url))]);
  entries.push(["capture-source/social-card.html", await readFile(new URL("../social-card.html", import.meta.url))]);
  return entries;
}

test("the complete generated text tree and tracked static inputs hash identically across two preparations", async () => {
  const [first, second, assets] = await Promise.all([
    prepareSite({ today: "2026-07-15" }),
    prepareSite({ today: "2026-07-15" }),
    sourceAssets(),
  ]);
  assert.equal(digestPreparedSite(first, assets), digestPreparedSite(second, assets));
});
