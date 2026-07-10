import { escapeHTML } from "./localize.mjs";
import { breadcrumbItems } from "./breadcrumbs.mjs";
import { siteURL, validateProject } from "./project-schema.mjs";

const localeSlug = Object.freeze({ en: "en", "zh-Hans": "zh" });

function safeJSON(value) {
  return JSON.stringify(value, null, 2).replaceAll("<", "\\u003c");
}

function localizedTitle(project, route, page, locale) {
  if (route.kind === "home") return project.locales[locale].title;
  return `${page[locale].title} — macMLX`;
}

function localizedDescription(project, route, page, locale) {
  return route.kind === "home" ? project.locales[locale].description : page[locale].description;
}

function homeGraph(project, locale, canonical, description) {
  const websiteId = siteURL(project, "/#website");
  const softwareId = siteURL(project, "/#software");
  return [
    {
      "@type": "WebSite",
      "@id": websiteId,
      url: siteURL(project, "/"),
      name: "macMLX",
      description,
      inLanguage: locale,
    },
    {
      "@type": "SoftwareApplication",
      "@id": softwareId,
      name: "macMLX",
      url: canonical,
      description: project.locales[locale].structuredDescription,
      inLanguage: locale,
      isPartOf: { "@id": websiteId },
      operatingSystem: project.operatingSystem,
      applicationCategory: "DeveloperApplication",
      softwareVersion: project.currentVersion,
      dateModified: project.lastVerified,
      codeRepository: project.repositoryURL,
      downloadUrl: project.downloadURL,
      license: project.licenseURL,
      programmingLanguage: "Swift",
      featureList: locale === "en"
        ? ["Native SwiftUI app", "Swift command-line interface", "Compatible local HTTP API", "Swift in-process MLX inference"]
        : ["原生 SwiftUI 应用", "Swift 命令行界面", "兼容的本地 HTTP API", "Swift 进程内 MLX 推理"],
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
    },
  ];
}

function articleGraph(project, page, locale, canonical, description, faqs) {
  const graph = [
    {
      "@type": "TechArticle",
      "@id": `${canonical}#article`,
      headline: page[locale].title,
      description,
      inLanguage: locale,
      dateModified: page.lastVerified,
      mainEntityOfPage: canonical,
      isPartOf: { "@id": siteURL(project, "/#website") },
      about: { "@id": siteURL(project, "/#software") },
    },
    {
      "@type": "BreadcrumbList",
      "@id": `${canonical}#breadcrumb`,
      itemListElement: breadcrumbItems(page, locale).map((item, index) => ({ "@type": "ListItem", position: index + 1, name: item.name, item: siteURL(project, item.path) })),
    },
  ];
  if (page.id === "faq") {
    graph.push({
      "@type": "FAQPage",
      "@id": `${canonical}#faq`,
      mainEntity: faqs.map((item) => ({
        "@type": "Question",
        name: item[locale].question,
        acceptedAnswer: { "@type": "Answer", text: item[locale].answer },
      })),
    });
  }
  return graph;
}

export function markdownURLFor(project, route, locale) {
  return siteURL(project, `/content/${localeSlug[locale]}/${route.id}.md`);
}

export function socialImageURLFor(project, locale) {
  return siteURL(project, `/assets/social/og-${localeSlug[locale]}.png`);
}

export function renderMetadata({ project, route, locale, page = route, faqs = [] }) {
  validateProject(project);
  if (!route?.paths?.[locale]) throw new Error(`Missing route path for ${route?.id ?? "unknown"}/${locale}`);
  const title = localizedTitle(project, route, page, locale);
  const description = localizedDescription(project, route, page, locale);
  const canonical = siteURL(project, route.paths[locale]);
  const english = siteURL(project, route.paths.en);
  const chinese = siteURL(project, route.paths["zh-Hans"]);
  const image = socialImageURLFor(project, locale);
  const imageAlt = locale === "en" ? "macMLX native Swift inference for Apple Silicon" : "macMLX 在 Apple 芯片上的原生 Swift 推理";
  const graph = route.kind === "home"
    ? homeGraph(project, locale, canonical, description)
    : articleGraph(project, page, locale, canonical, description, faqs);

  return `<title>${escapeHTML(title)}</title>
  <meta name="description" content="${escapeHTML(description)}">
  <link rel="canonical" href="${escapeHTML(canonical)}">
  <link rel="alternate" hreflang="en" href="${escapeHTML(english)}">
  <link rel="alternate" hreflang="zh-Hans" href="${escapeHTML(chinese)}">
  <link rel="alternate" hreflang="x-default" href="${escapeHTML(english)}">
  <link rel="alternate" type="text/markdown" href="${escapeHTML(markdownURLFor(project, route, locale))}">
  <meta property="og:type" content="${route.kind === "home" ? "website" : "article"}">
  <meta property="og:site_name" content="macMLX">
  <meta property="og:url" content="${escapeHTML(canonical)}">
  <meta property="og:locale" content="${escapeHTML(project.locales[locale].ogLocale)}">
  <meta property="og:title" content="${escapeHTML(title)}">
  <meta property="og:description" content="${escapeHTML(description)}">
  <meta property="og:image" content="${escapeHTML(image)}">
  <meta property="og:image:type" content="image/png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="${escapeHTML(imageAlt)}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${escapeHTML(title)}">
  <meta name="twitter:description" content="${escapeHTML(description)}">
  <meta name="twitter:image" content="${escapeHTML(image)}">
  <script type="application/ld+json">${safeJSON({ "@context": "https://schema.org", "@graph": graph })}</script>`;
}
