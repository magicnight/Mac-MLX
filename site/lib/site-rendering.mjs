import { localizeBilingualElements, renderTokens } from "./localize.mjs";
import { renderMetadata } from "./metadata.mjs";
import { validateProject } from "./project-schema.mjs";

function homeTemplateValues(project, routes, locale) {
  const metadata = project.locales[locale];
  const counterpartLocale = locale === "en" ? "zh-Hans" : "en";
  const counterpart = project.locales[counterpartLocale];
  const prefix = locale === "en" ? "" : "/zh";
  const repositoryBase = new URL(`${project.repositoryURL}/`);
  return {
    apiLabel: "API",
    apiPath: `${prefix}/api-compatibility/`,
    architectureLabel: locale === "en" ? "Architecture" : "架构",
    architecturePath: `${prefix}/architecture/`,
    buildSourceURL: new URL("#building-from-source", project.repositoryURL).href,
    changelogURL: new URL("blob/main/CHANGELOG.md", repositoryBase).href,
    chineseLanguageClass: locale === "zh-Hans" ? "active" : "",
    copyFailure: metadata.copyFailure,
    copySuccess: metadata.copySuccess,
    counterpartHtmlLang: counterpart.htmlLang,
    counterpartLocale,
    currentVersion: project.currentVersion,
    downloadURL: project.downloadURL,
    englishLanguageClass: locale === "en" ? "active" : "",
    htmlLang: metadata.htmlLang,
    issuesURL: new URL("issues", repositoryBase).href,
    learnLabel: locale === "en" ? "Learn" : "了解",
    languageHref: routes[counterpartLocale],
    languageLinkLabel: metadata.languageLinkLabel,
    modelsLabel: locale === "en" ? "Models" : "模型",
    modelsPath: `${prefix}/models/`,
    primaryNavigationLabel: metadata.primaryNavigationLabel,
    compareLabel: locale === "en" ? "Compare" : "对比",
    comparePath: `${prefix}/compare/`,
    faqLabel: locale === "en" ? "FAQ" : "常见问题",
    faqPath: `${prefix}/faq/`,
    releasesLabel: locale === "en" ? "Releases" : "版本",
    releasesPath: `${prefix}/releases/`,
    releasesURL: new URL("releases", repositoryBase).href,
    repositoryLicenseURL: new URL("blob/main/LICENSE", repositoryBase).href,
    repositoryURL: project.repositoryURL,
    themeToggleLabel: metadata.themeToggleLabel,
  };
}

export function renderHomeTemplate({ template, project, routes, locale, metadataHTML }) {
  validateProject(project);
  if (locale !== "en" && locale !== "zh-Hans") throw new Error(`Unsupported locale: ${locale}`);
  const metadata = metadataHTML ?? renderMetadata({ project, route: { id: "home", kind: "home", paths: routes }, locale });
  if (typeof metadata !== "string" || metadata.trim() === "") throw new Error("Home metadata HTML is required");
  const localized = localizeBilingualElements(template, locale);
  const rendered = renderTokens(localized, homeTemplateValues(project, routes, locale));
  return `${rendered.replace("<!--site-metadata-->", metadata).trimEnd()}\n`;
}

export function renderSocialImage({ template, project }) {
  validateProject(project);
  return `${renderTokens(template, { currentVersion: project.currentVersion }).trimEnd()}\n`;
}
