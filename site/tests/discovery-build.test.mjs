import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { prepareSite, validateSocialAssets } from "../../scripts/build-public-site.mjs";

test("prepared site contains all Markdown, GEO, sitemap, robots, and noindex 404 outputs", async () => {
  const { documents, markdownDocuments, discoveryFiles } = await prepareSite({ today: "2026-07-15" });
  assert.equal(markdownDocuments.size, 28);
  assert.deepEqual([...discoveryFiles.keys()], [
    "llms.txt", "llms-full.txt", "zh/llms.txt", "zh/llms-full.txt",
    "robots.txt", "sitemap.xml", "404.html", "zh/404.html",
  ]);
  const sitemap = discoveryFiles.get("sitemap.xml");
  const llms = discoveryFiles.get("llms.txt");
  const llmsFull = discoveryFiles.get("llms-full.txt");
  assert.equal(sitemap.match(/<url>/g)?.length, 28);
  assert.match(sitemap, /<loc>https:\/\/macmlx\.app\/releases\/v0-6-2\/<\/loc>[\s\S]*?<lastmod>2026-07-15<\/lastmod>/);
  assert.doesNotMatch(sitemap, /404/);
  assert.match(llms, /Latest release: v0\.6\.2/);
  assert.match(llmsFull, /### continuous-batching[\s\S]*?- Status: released[\s\S]*?- Title: Eligibility-gated continuous batching/);
  assert.match(llmsFull, /### paged-kv[\s\S]*?- Status: planned[\s\S]*?- Title: Paged KV, block sharing, and CoW/);
  const internLMEnglish = llmsFull.match(/### internlm3-theoretical\n([\s\S]*?)(?=\n### |\n## Canonical)/)?.[1];
  const internLMChinese = discoveryFiles.get("zh/llms-full.txt").match(/### internlm3-theoretical\n([\s\S]*?)(?=\n### |\n## 规范)/)?.[1];
  assert.match(internLMEnglish, /- Status: theoretical/);
  assert.doesNotMatch(internLMEnglish, /Status: released/);
  assert.match(internLMChinese, /- 状态：theoretical/);
  assert.doesNotMatch(internLMChinese, /已发布|状态：released/);
  for (const [path, title, label] of [
    ["/models/", "InternLM3 theoretical support", "Theoretical"],
    ["/zh/models/", "InternLM3 理论支持", "理论支持"],
  ]) {
    const card = [...documents.get(path).matchAll(/<article class="fact-card"[^>]*>[\s\S]*?<\/article>/g)]
      .map((match) => match[0])
      .find((candidate) => candidate.includes(`<h3>${title}</h3>`));
    assert.ok(card, `missing InternLM3 card on ${path}`);
    assert.match(card, new RegExp(`data-status="theoretical"[\\s\\S]*?>${label}<`));
    assert.doesNotMatch(card, />Released<|>已发布</);
  }
  assert.ok(markdownDocuments.has("content/en/release-v0-6-2.md"));
  assert.ok(markdownDocuments.has("content/zh/release-v0-6-2.md"));
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

test("social asset validation rejects a structurally valid PNG from the wrong locale", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "macmlx-social-stale-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "social"));
  const chinese = await readFile(new URL("../assets/social/og-zh.png", import.meta.url));
  await writeFile(join(root, "social/og-en.png"), chinese);
  await writeFile(join(root, "social/og-zh.png"), chinese);
  await assert.rejects(validateSocialAssets(root), /og-en\.png source digest does not match/i);
});
