import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const readme = await readFile(new URL("../README.md", import.meta.url), "utf8");

test("maintenance guide documents offline verification and governed release refresh", () => {
  for (const command of [
    "node scripts/render-social-cards.mjs",
    "node scripts/build-public-site.mjs",
    "node --test site/tests/*.test.mjs",
    "node scripts/crawl-public-site.mjs",
    "node scripts/test-public-site.mjs",
  ]) assert.match(readme, new RegExp(command.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  assert.match(readme, /no network|network-free/i);
  assert.match(readme, /project and release registries/i);
  assert.match(readme, /reclassify facts/i);
  assert.match(readme, /affected competitors/i);
  assert.match(readme, /official sources/i);
  assert.match(readme, /lastVerified/);
  assert.match(readme, /browser QA/i);
  assert.match(readme, /Search Console.*Bing.*IndexNow/is);
  assert.match(readme, /separately authorized post-deployment operations/i);
});
