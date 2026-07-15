import { siteURL, validateProject } from "./project-schema.mjs";

const localeDirectory = Object.freeze({ en: "en", "zh-Hans": "zh" });

const labels = Object.freeze({
  en: Object.freeze({
    answer: "Direct answer",
    canonical: "Canonical",
    verified: "Last verified",
    facts: "Page facts",
    sources: "Sources",
    related: "Related pages",
    released: "Released",
    limited: "Limited",
    theoretical: "Theoretical",
    development: "In development",
    planned: "Planned",
  }),
  "zh-Hans": Object.freeze({
    answer: "直接回答",
    canonical: "规范网址",
    verified: "最后核验",
    facts: "页面事实",
    sources: "来源",
    related: "相关页面",
    released: "已发布",
    limited: "有限支持",
    theoretical: "理论支持",
    development: "开发中",
    planned: "规划中",
  }),
});

function unique(values) {
  return [...new Set(values)];
}

function pageFactIds(route, releases) {
  if (route.kind === "home") return ["platform-installation", "swift-in-process", "shared-core", "no-python-default"];
  const ids = [];
  for (const block of route.blocks ?? []) {
    ids.push(...(block.factIds ?? []));
    if (block.type === "release") {
      for (const releaseId of block.releaseIds ?? []) {
        const release = releases.find((item) => item.id === releaseId);
        if (release) ids.push(...release.shippedFactIds, ...release.limitationFactIds, ...release.developmentFactIds, ...release.plannedFactIds);
      }
    }
  }
  return unique(ids);
}

function routeSources(route, factItems, competitors, releases) {
  const urls = factItems.flatMap((fact) => fact.sourceUrls);
  for (const block of route.blocks ?? []) {
    for (const id of block.competitorIds ?? []) {
      const competitor = competitors.find((item) => item.id === id);
      if (competitor) urls.push(...competitor.officialSources);
    }
    for (const id of block.releaseIds ?? []) {
      const release = releases.find((item) => item.id === id);
      if (release) urls.push(...release.officialSources);
    }
  }
  return unique(urls);
}

function routeCopy(route, project, locale) {
  if (route.kind === "home") {
    return {
      title: project.locales[locale].title,
      description: project.locales[locale].description,
      directAnswer: project.locales[locale].structuredDescription,
    };
  }
  return route[locale];
}

function relatedRoutes(route, routes) {
  const byId = new Map(routes.map((item) => [item.id, item]));
  if (route.kind === "home") return routes.filter((item) => item.kind === "article").slice(0, 6);
  return route.relatedIds.map((id) => byId.get(id)).filter(Boolean);
}

function renderPageMarkdown({ route, locale, project, routes, facts, competitors, releases }) {
  const copy = routeCopy(route, project, locale);
  const text = labels[locale];
  const factById = new Map(facts.map((item) => [item.id, item]));
  const factItems = pageFactIds(route, releases).map((id) => factById.get(id)).filter(Boolean);
  const sources = routeSources(route, factItems, competitors, releases);
  const related = relatedRoutes(route, routes);
  const verified = route.lastVerified ?? project.lastVerified;
  const canonical = siteURL(project, route.paths[locale]);
  const separator = locale === "en" ? ": " : "：";
  const factLines = factItems.length === 0
    ? [`- ${locale === "en" ? "No separate governed facts are assigned to this index page." : "此索引页未单独分配受治理事实。"}`]
    : factItems.map((fact) => `- **${fact[locale].title}** — ${text[fact.supportTier ?? fact.status]}; ${locale === "en" ? "since" : "始于"} ${fact.sinceVersion}; ${text.verified.toLowerCase?.() ?? text.verified} ${fact.lastVerified}. ${fact[locale].summary} ${fact[locale].detail}`);
  const sourceLines = sources.length === 0
    ? [`- ${project.repositoryURL}`]
    : sources.map((url) => `- ${url}`);
  const relatedLines = related.map((item) => `- [${routeCopy(item, project, locale).title}](${siteURL(project, item.paths[locale])})`);

  return `# ${copy.title}

${copy.description}

## ${text.answer}

${copy.directAnswer || copy.description}

- ${text.canonical}${separator}${canonical}
- ${text.verified}${separator}${verified}

## ${text.facts}

${factLines.join("\n")}

## ${text.sources}

${sourceLines.join("\n")}

## ${text.related}

${relatedLines.join("\n")}
`;
}

export function renderMarkdownDocuments(context) {
  validateProject(context.project);
  const documents = new Map();
  for (const route of context.routes) {
    for (const locale of ["en", "zh-Hans"]) {
      const path = `content/${localeDirectory[locale]}/${route.id}.md`;
      documents.set(path, renderPageMarkdown({ ...context, route, locale }));
    }
  }
  return documents;
}

function pageIndex({ project, routes, locale }) {
  return routes.map((route) => {
    const copy = routeCopy(route, project, locale);
    return `- [${copy.title}](${siteURL(project, route.paths[locale])}) — ${copy.description}`;
  }).join("\n");
}

function shortIndex(context, locale) {
  const zh = locale === "zh-Hans";
  const separator = zh ? "：" : ": ";
  return `# macMLX

> ${context.project.locales[locale].structuredDescription}

## ${zh ? "当前状态" : "Current status"}

- ${zh ? "最新版本" : "Latest release"}${separator}v${context.project.currentVersion} (${context.project.releaseDate})
- ${zh ? "最后核验" : "Last verified"}${separator}${context.project.lastVerified}
- ${zh ? "源码" : "Source"}${separator}${context.project.repositoryURL}
- ${zh ? "下载" : "Download"}${separator}${context.project.downloadURL}
- ${zh ? "完整事实索引" : "Full fact index"}${separator}${siteURL(context.project, `${zh ? "/zh" : ""}/llms-full.txt`)}

## ${zh ? "页面" : "Pages"}

${pageIndex({ ...context, locale })}
`;
}

function fullIndex(context, locale) {
  const zh = locale === "zh-Hans";
  const separator = zh ? "：" : ": ";
  const facts = context.facts.map((fact) => `### ${fact.id}

- ${zh ? "状态" : "Status"}${separator}${fact.supportTier ?? fact.status}
- ${zh ? "始于版本" : "Since version"}${separator}${fact.sinceVersion}
- ${zh ? "最后核验" : "Last verified"}${separator}${fact.lastVerified}
- ${zh ? "标题" : "Title"}${separator}${fact[locale].title}
- ${zh ? "摘要" : "Summary"}${separator}${fact[locale].summary}
- ${zh ? "边界" : "Boundary"}${separator}${fact[locale].detail}
- ${zh ? "来源" : "Sources"}${zh ? "：" : ":"}
${fact.sourceUrls.map((url) => `  - ${url}`).join("\n")}`).join("\n\n");
  const pageURLs = context.routes.map((route) => `- ${siteURL(context.project, route.paths[locale])}`).join("\n");
  return `${shortIndex(context, locale)}
## ${zh ? "受治理事实" : "Governed facts"}

${facts}

## ${zh ? "规范页面列表" : "Canonical page list"}

${pageURLs}
`;
}

export function renderLLMSIndexes(context) {
  validateProject(context.project);
  return new Map([
    ["llms.txt", shortIndex(context, "en")],
    ["llms-full.txt", fullIndex(context, "en")],
    ["zh/llms.txt", shortIndex(context, "zh-Hans")],
    ["zh/llms-full.txt", fullIndex(context, "zh-Hans")],
  ]);
}
