import assert from "node:assert/strict";
import test from "node:test";

import { faqs } from "../content/faqs.mjs";
import { pages } from "../content/pages.mjs";
import { project } from "../content/project.mjs";
import { renderMetadata } from "../lib/metadata.mjs";
import { routes } from "../routes.mjs";
import { prepareSite } from "../../scripts/build-public-site.mjs";

function jsonLD(html) {
  const scripts = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
  assert.equal(scripts.length, 1);
  return JSON.parse(scripts[0][1]);
}

test("home metadata is localized, reciprocal, Markdown-addressable, and uses stable entity IDs", () => {
  for (const locale of ["en", "zh-Hans"]) {
    const html = renderMetadata({ project, route: routes[0], locale, faqs });
    const socialLocale = locale === "en" ? "en" : "zh";
    assert.equal(html.match(/<title>/g)?.length, 1);
    assert.equal(html.match(/<meta name="description"/g)?.length, 1);
    assert.equal(html.match(/<link rel="canonical"/g)?.length, 1);
    assert.match(html, /hreflang="en"/);
    assert.match(html, /hreflang="zh-Hans"/);
    assert.match(html, /hreflang="x-default"/);
    assert.match(html, new RegExp(`type="text/markdown" href="https://macmlx\\.app/content/${socialLocale}/home\\.md"`));
    assert.match(html, new RegExp(`og:image" content="https://macmlx\\.app/assets/social/og-${socialLocale}\\.png"`));
    assert.match(html, /og:image:width" content="1200"/);
    assert.match(html, /og:image:height" content="630"/);

    const graph = jsonLD(html)["@graph"];
    const website = graph.find((node) => node["@type"] === "WebSite");
    const software = graph.find((node) => node["@type"] === "SoftwareApplication");
    assert.equal(website["@id"], "https://macmlx.app/#website");
    assert.equal(software["@id"], "https://macmlx.app/#software");
    assert.equal(project.currentVersion, "0.7.0");
    assert.equal(project.lastVerified, "2026-07-19");
    assert.equal(software.softwareVersion, "0.7.0");
    assert.equal(software.dateModified, "2026-07-19");
    assert.equal(software.codeRepository, project.repositoryURL);
    assert.equal(software.downloadUrl, project.downloadURL);
    assert.equal(software.offers.price, "0");
    assert.equal(software.aggregateRating, undefined);
    assert.equal(software.review, undefined);
    assert.equal(software.screenshot, undefined);
  }
});

test("article metadata emits TechArticle and breadcrumbs, with visible FAQ data only on FAQ", () => {
  const faqPage = pages.find((page) => page.id === "faq");
  const articlePage = pages.find((page) => page.id === "architecture");
  for (const page of [faqPage, articlePage]) {
    const html = renderMetadata({ project, route: page, page, locale: "en", faqs });
    const graph = jsonLD(html)["@graph"];
    assert.ok(graph.some((node) => node["@type"] === "TechArticle"));
    assert.ok(graph.some((node) => node["@type"] === "BreadcrumbList"));
    const faqGraph = graph.find((node) => node["@type"] === "FAQPage");
    if (page.id === "faq") {
      assert.equal(faqGraph.mainEntity.length, faqs.length);
      assert.equal(faqGraph.mainEntity[0].name, faqs[0].en.question);
      assert.equal(faqGraph.mainEntity[0].acceptedAnswer.text, faqs[0].en.answer);
    } else {
      assert.equal(faqGraph, undefined);
    }
  }
});

test("the site builder uses centralized metadata exactly once in every HTML document", async () => {
  const { documents } = await prepareSite({ today: "2026-07-19" });
  assert.equal(documents.size, 30);
  for (const [path, html] of documents) {
    const socialLocale = path.startsWith("/zh/") ? "zh" : "en";
    assert.equal(html.match(/<title>/g)?.length, 1, path);
    assert.equal(html.match(/<meta name="description"/g)?.length, 1, path);
    assert.equal(html.match(/<link rel="canonical"/g)?.length, 1, path);
    assert.equal(html.match(/<script type="application\/ld\+json">/g)?.length, 1, path);
    assert.match(html, new RegExp(`type="text/markdown" href="https://macmlx\\.app/content/${socialLocale}/`));
    assert.match(html, new RegExp(`og:image" content="https://macmlx\\.app/assets/social/og-${socialLocale}\\.png"`));
  }
});

test("nested article DOM and JSON-LD breadcrumbs share Home, parent, and current hierarchy", async () => {
  const { documents } = await prepareSite({ today: "2026-07-19" });
  const nested = pages.filter((page) => page.paths.en.split("/").filter(Boolean).length > 1);
  for (const page of nested) {
    for (const locale of ["en", "zh-Hans"]) {
      const html = documents.get(page.paths[locale]);
      const nav = html.match(/<nav class="breadcrumbs"[\s\S]*?<\/nav>/)?.[0];
      assert.ok(nav, `${page.id}/${locale} needs visible breadcrumbs`);
      const visible = [
        ...[...nav.matchAll(/<a href="([^"]+)">([^<]+)<\/a>/g)].map((match) => ({ href: match[1], name: match[2] })),
        { href: page.paths[locale], name: nav.match(/<span aria-current="page">([^<]+)<\/span>/)?.[1] },
      ];
      const graph = jsonLD(html)["@graph"];
      const structured = graph.find((node) => node["@type"] === "BreadcrumbList").itemListElement;
      assert.equal(visible.length, 3, `${page.id}/${locale} visible hierarchy`);
      assert.equal(structured.length, 3, `${page.id}/${locale} structured hierarchy`);
      assert.deepEqual(structured.map((item) => item.name), visible.map((item) => item.name));
      assert.deepEqual(structured.map((item) => new URL(item.item).pathname), visible.map((item) => item.href));
    }
  }
});

test("the v0.7.0 release page exposes immutable release identity with localized canonical metadata", async () => {
  const { documents } = await prepareSite({ today: "2026-07-19" });
  const releasePage = pages.find((page) => page.id === "release-v0-7-0");
  assert.ok(releasePage);

  for (const locale of ["en", "zh-Hans"]) {
    const path = releasePage.paths[locale];
    const html = documents.get(path);
    assert.ok(html, `missing ${path}`);
    assert.match(html, new RegExp(`<link rel="canonical" href="${project.origin}${path}">`));

    const graph = jsonLD(html)["@graph"];
    const article = graph.find((node) => node["@type"] === "TechArticle");
    const softwareRelease = graph.find((node) => node["@type"] === "SoftwareApplication");
    const breadcrumbs = graph.find((node) => node["@type"] === "BreadcrumbList");
    assert.equal(article.dateModified, "2026-07-19");
    assert.equal(new URL(article.mainEntityOfPage).pathname, path);
    assert.equal(softwareRelease["@id"], "https://github.com/magicnight/mac-mlx/releases/tag/v0.7.0#software-release");
    assert.equal(softwareRelease.url, "https://github.com/magicnight/mac-mlx/releases/tag/v0.7.0");
    assert.equal(softwareRelease.softwareVersion, "0.7.0");
    assert.equal(softwareRelease.datePublished, "2026-07-18");
    assert.equal(softwareRelease.dateModified, undefined);
    assert.equal(softwareRelease.downloadUrl, undefined);
    assert.equal(new URL(softwareRelease.mainEntityOfPage).pathname, path);
    assert.deepEqual(
      breadcrumbs.itemListElement.map((item) => new URL(item.item).pathname),
      locale === "en" ? ["/", "/releases/", path] : ["/zh/", "/zh/releases/", path],
    );
  }
});

test("historical release entities keep version-specific immutable metadata", async () => {
  const { documents } = await prepareSite({ today: "2026-07-19" });
  for (const [id, version, datePublished] of [
    ["release-v0-6-2", "0.6.2", "2026-07-11"],
    ["release-v0-5-3", "0.5.3", "2026-07-08"],
  ]) {
    const releasePage = pages.find((page) => page.id === id);
    assert.ok(releasePage, `missing ${id}`);

    for (const locale of ["en", "zh-Hans"]) {
      const path = releasePage.paths[locale];
      const graph = jsonLD(documents.get(path))["@graph"];
      const article = graph.find((node) => node["@type"] === "TechArticle");
      const softwareRelease = graph.find((node) => node["@type"] === "SoftwareApplication");
      assert.equal(article.dateModified, "2026-07-19");
      assert.equal(softwareRelease["@id"], `https://github.com/magicnight/mac-mlx/releases/tag/v${version}#software-release`);
      assert.equal(softwareRelease.url, `https://github.com/magicnight/mac-mlx/releases/tag/v${version}`);
      assert.equal(softwareRelease.softwareVersion, version);
      assert.equal(softwareRelease.datePublished, datePublished);
      assert.equal(softwareRelease.dateModified, undefined);
      assert.equal(softwareRelease.downloadUrl, undefined);
      assert.equal(new URL(softwareRelease.mainEntityOfPage).pathname, path);
    }
  }
});
