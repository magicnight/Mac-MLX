import assert from "node:assert/strict";
import test from "node:test";

import { competitors } from "../content/competitors.mjs";
import { facts } from "../content/facts.mjs";
import { faqs } from "../content/faqs.mjs";
import { pages } from "../content/pages.mjs";
import { project } from "../content/project.mjs";
import { releases } from "../content/releases.mjs";
import { renderLLMSIndexes, renderMarkdownDocuments } from "../lib/markdown.mjs";
import { routes } from "../routes.mjs";

const context = { project, routes, pages, facts, competitors, faqs, releases };

test("every route has a registry-derived Markdown counterpart in both locales", () => {
  const documents = renderMarkdownDocuments(context);
  assert.equal(documents.size, 30);
  for (const route of routes) {
    for (const [locale, directory] of [["en", "en"], ["zh-Hans", "zh"]]) {
      const path = `content/${directory}/${route.id}.md`;
      const markdown = documents.get(path);
      assert.ok(markdown, `missing ${path}`);
      assert.match(markdown, /^# /);
      assert.match(markdown, /## (Direct answer|直接回答)/);
      assert.match(markdown, /## (Page facts|页面事实)/);
      assert.match(markdown, /## (Sources|来源)/);
      assert.match(markdown, /## (Related pages|相关页面)/);
      const canonicalLabel = locale === "en" ? "Canonical: " : "规范网址：";
      assert.match(markdown, new RegExp(`${canonicalLabel}${project.origin}${route.paths[locale].replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
      assert.match(markdown, /Last verified: 2026-07-19|最后核验：2026-07-19/);
      assert.doesNotMatch(markdown, /undefined|\{\{|\}\}/);
    }
  }
});

test("short llms files are navigational and full files preserve fact/status/source parity", () => {
  const indexes = renderLLMSIndexes(context);
  assert.deepEqual([...indexes.keys()], ["llms.txt", "llms-full.txt", "zh/llms.txt", "zh/llms-full.txt"]);
  assert.match(indexes.get("llms.txt"), /Latest release: v0\.7\.0/);
  assert.match(indexes.get("zh/llms.txt"), /最新版本：v0\.7\.0/);
  assert.doesNotMatch(indexes.get("llms.txt"), /## Governed facts/);
  assert.doesNotMatch(indexes.get("zh/llms.txt"), /## 受治理事实/);

  const english = indexes.get("llms-full.txt");
  const chinese = indexes.get("zh/llms-full.txt");
  assert.equal(english.match(/^### /gm)?.length, facts.length);
  assert.equal(chinese.match(/^### /gm)?.length, facts.length);
  for (const fact of facts) {
    assert.match(english, new RegExp(`### ${fact.id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
    assert.match(chinese, new RegExp(`### ${fact.id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
    assert.match(english, new RegExp(`Status: ${fact.status}`));
    assert.match(chinese, new RegExp(`状态：${fact.status}`));
    for (const url of fact.sourceUrls) {
      assert.ok(english.includes(url));
      assert.ok(chinese.includes(url));
    }
  }
  assert.equal(english.match(/^- https:\/\/macmlx\.app\//gm)?.length, routes.length);
  assert.equal(chinese.match(/^- https:\/\/macmlx\.app\/zh\//gm)?.length, routes.length);
});
