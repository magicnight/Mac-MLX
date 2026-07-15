import assert from "node:assert/strict";
import test from "node:test";

import { project } from "../content/project.mjs";
import { renderNotFound, renderRobots, renderSitemap } from "../lib/discovery.mjs";
import { routes } from "../routes.mjs";

test("sitemap lists every canonical once with lastmod and reciprocal locale alternates", () => {
  const xml = renderSitemap({ project, routes });
  assert.equal(xml.match(/<url>/g)?.length, 28);
  assert.equal(xml.match(/<loc>/g)?.length, 28);
  assert.equal(xml.match(/hreflang="en"/g)?.length, 28);
  assert.equal(xml.match(/hreflang="zh-Hans"/g)?.length, 28);
  assert.equal(xml.match(/hreflang="x-default"/g)?.length, 28);
  assert.equal(xml.match(/<lastmod>2026-07-15<\/lastmod>/g)?.length, 28);
  for (const route of routes) for (const path of Object.values(route.paths)) {
    const escaped = `${project.origin}${path}`.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    assert.equal(xml.match(new RegExp(`<loc>${escaped}<\\/loc>`, "g"))?.length, 1);
  }
  assert.doesNotMatch(xml, /\?lang=/);
});

test("sitemap XML-escapes route values", () => {
  const customRoutes = [{ id: "home", kind: "home", paths: { en: "/search?a=1&b=2", "zh-Hans": "/zh/search?a=1&b=2" } }];
  const xml = renderSitemap({ project, routes: customRoutes });
  assert.match(xml, /https:\/\/macmlx\.app\/search\?a=1&amp;b=2/);
  assert.doesNotMatch(xml, /a=1&b=2/);
});

test("robots allows general and named OpenAI crawlers with exactly one sitemap", () => {
  const robots = renderRobots(project);
  assert.match(robots, /User-agent: \*\nAllow: \//);
  assert.match(robots, /User-agent: OAI-SearchBot\nAllow: \//);
  assert.match(robots, /User-agent: GPTBot\nAllow: \//);
  assert.equal(robots.match(/^Sitemap:/gm)?.length, 1);
  assert.doesNotMatch(robots, /ChatGPT-User|OpenAIbot/i);
});

test("localized not-found documents are noindex and link to real locale homes", () => {
  const english = renderNotFound({ project, locale: "en" });
  const chinese = renderNotFound({ project, locale: "zh-Hans" });
  assert.match(english, /<meta name="robots" content="noindex,follow">/);
  assert.match(chinese, /<meta name="robots" content="noindex,follow">/);
  assert.match(english, /href="\/">Return home/);
  assert.match(chinese, /href="\/zh\/">返回首页/);
  for (const html of [english, chinese]) {
    assert.doesNotMatch(html, /rel="canonical"/);
    assert.equal(html.match(/<h1>/g)?.length, 1);
  }
});
