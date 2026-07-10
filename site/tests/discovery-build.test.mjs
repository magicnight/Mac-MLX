import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { prepareSite, validateSocialAssets } from "../../scripts/build-public-site.mjs";

test("prepared site contains all Markdown, GEO, sitemap, robots, and noindex 404 outputs", async () => {
  const { markdownDocuments, discoveryFiles } = await prepareSite({ today: "2026-07-10" });
  assert.equal(markdownDocuments.size, 26);
  assert.deepEqual([...discoveryFiles.keys()], [
    "llms.txt", "llms-full.txt", "zh/llms.txt", "zh/llms-full.txt",
    "robots.txt", "sitemap.xml", "404.html", "zh/404.html",
  ]);
  assert.equal(discoveryFiles.get("sitemap.xml").match(/<url>/g)?.length, 26);
  assert.doesNotMatch(discoveryFiles.get("sitemap.xml"), /404/);
  assert.match(discoveryFiles.get("404.html"), /noindex,follow/);
  assert.match(discoveryFiles.get("zh\/404.html") ?? discoveryFiles.get("zh/404.html"), /noindex,follow/);
});

test("social asset validation fails before publication with exact capture instructions", async (t) => {
  const empty = await mkdtemp(join(tmpdir(), "macmlx-social-assets-"));
  t.after(() => rm(empty, { recursive: true, force: true }));
  await assert.rejects(
    validateSocialAssets(empty),
    /node scripts\/render-social-cards\.mjs[\s\S]*site\/assets\/social\/og-en\.png[\s\S]*site\/assets\/social\/og-zh\.png/,
  );
});

test("social asset validation parses both tracked PNGs before publication", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "macmlx-social-corrupt-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "social"));
  const valid = await readFile(new URL("../assets/social/og-zh.png", import.meta.url));
  await writeFile(join(root, "social/og-en.png"), Buffer.from("not a png"));
  await writeFile(join(root, "social/og-zh.png"), valid);
  await assert.rejects(validateSocialAssets(root), /og-en\.png is not a PNG/);
});
