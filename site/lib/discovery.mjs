import { escapeHTML } from "./localize.mjs";
import { siteURL, validateProject } from "./project-schema.mjs";

function escapeXML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function url(project, path) {
  return siteURL(project, path);
}

export function renderSitemap({ project, routes }) {
  validateProject(project);
  const entries = routes.flatMap((route) => ["en", "zh-Hans"].map((locale) => `  <url>
    <loc>${escapeXML(url(project, route.paths[locale]))}</loc>
    <lastmod>${escapeXML(route.lastVerified ?? project.lastVerified)}</lastmod>
    <xhtml:link rel="alternate" hreflang="en" href="${escapeXML(url(project, route.paths.en))}"/>
    <xhtml:link rel="alternate" hreflang="zh-Hans" href="${escapeXML(url(project, route.paths["zh-Hans"]))}"/>
    <xhtml:link rel="alternate" hreflang="x-default" href="${escapeXML(url(project, route.paths.en))}"/>
  </url>`));
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">
${entries.join("\n")}
</urlset>
`;
}

export function renderRobots(project) {
  validateProject(project);
  return `User-agent: *
Allow: /

User-agent: OAI-SearchBot
Allow: /

User-agent: GPTBot
Allow: /

Sitemap: ${siteURL(project, "/sitemap.xml")}
`;
}

export function renderNotFound({ project, locale }) {
  const zh = locale === "zh-Hans";
  const lang = zh ? "zh-CN" : "en";
  const title = zh ? "页面未找到 — macMLX" : "Page not found — macMLX";
  const heading = zh ? "页面未找到" : "Page not found";
  const body = zh ? "这个地址不存在，或页面已经移动。" : "This address does not exist, or the page has moved.";
  const label = zh ? "返回首页" : "Return home";
  const home = zh ? "/zh/" : "/";
  return `<!doctype html>
<html lang="${lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHTML(title)}</title>
  <meta name="robots" content="noindex,follow">
  <meta name="color-scheme" content="light dark">
  <style>html{font-family:ui-sans-serif,system-ui;background:#111311;color:#f3f1ea}body{min-height:100vh;margin:0;display:grid;place-items:center}.card{max-width:38rem;padding:4rem;border:1px solid #455047;border-radius:1.5rem;background:#191c19}p{color:#c9cdc7;line-height:1.7}a{color:#a9e88b}</style>
</head>
<body><main class="card"><p>404</p><h1>${escapeHTML(heading)}</h1><p>${escapeHTML(body)}</p><a href="${home}">${escapeHTML(label)}</a></main></body>
</html>
`;
}
