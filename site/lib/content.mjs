export const contentStatuses = Object.freeze(["released", "development", "planned"]);

const locales = Object.freeze(["en", "zh-Hans"]);
const statuses = new Set(contentStatuses);
const isoDatePattern = /^\d{4}-\d{2}-\d{2}$/;
const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function requireNonEmptyString(value, label) {
  if (typeof value !== "string" || value.trim() === "") throw new Error(`${label} must be a non-empty string`);
}

function requireUniqueId(item, ids, label) {
  requireNonEmptyString(item?.id, `${label}.id`);
  if (!idPattern.test(item.id)) throw new Error(`invalid ${label} id: ${item.id}`);
  if (ids.has(item.id)) throw new Error(`duplicate ${label} id: ${item.id}`);
  ids.add(item.id);
}

function ageInDays(date, today) {
  if (!isoDatePattern.test(date ?? "") || Number.isNaN(Date.parse(`${date}T00:00:00Z`))) throw new Error(`invalid date: ${date}`);
  if (!isoDatePattern.test(today ?? "") || Number.isNaN(Date.parse(`${today}T00:00:00Z`))) throw new Error(`invalid validation date: ${today}`);
  return Math.floor((Date.parse(`${today}T00:00:00Z`) - Date.parse(`${date}T00:00:00Z`)) / 86_400_000);
}

function requireFreshDate(item, label, { today, maxAgeDays }) {
  const age = ageInDays(item.lastVerified, today);
  if (age < 0) throw new Error(`future ${label} verification date: ${item.id}`);
  if (age > maxAgeDays) throw new Error(`stale ${label}: ${item.id}`);
}

function requireHTTPSUrls(urls, label) {
  if (!Array.isArray(urls) || urls.length === 0) throw new Error(`${label} must have at least one source`);
  for (const value of urls) {
    let url;
    try {
      url = new URL(value);
    } catch {
      throw new Error(`invalid source URL for ${label}: ${value}`);
    }
    if (url.protocol !== "https:") throw new Error(`source URL must use HTTPS for ${label}: ${value}`);
  }
}

function pathWithin(pathname, root) {
  return pathname === root || pathname.startsWith(`${root}/`);
}

function isApprovedMacMLXSource(value) {
  const url = new URL(value);
  if (url.hostname === "github.com") return pathWithin(url.pathname, "/magicnight/mac-mlx");
  return url.hostname === "ml-explore.github.io" && pathWithin(url.pathname, "/mlx");
}

const competitorSourceRules = Object.freeze({
  ollama: (url) => (url.hostname === "github.com" && pathWithin(url.pathname, "/ollama/ollama")) || url.hostname === "ollama.com" || url.hostname === "docs.ollama.com",
  "lm-studio": (url) => url.hostname === "lmstudio.ai" || (url.hostname === "github.com" && pathWithin(url.pathname, "/lmstudio-ai/mlx-engine")),
  omlx: (url) => url.hostname === "github.com" && pathWithin(url.pathname, "/jundot/omlx"),
  swama: (url) => url.hostname === "github.com" && pathWithin(url.pathname, "/Trans-N-ai/swama"),
  swiftlm: (url) => url.hostname === "github.com" && pathWithin(url.pathname, "/SharpAI/SwiftLM"),
});

function requireNotFutureDate(date, today, label) {
  if (ageInDays(date, today) < 0) throw new Error(`future ${label}: ${date}`);
}

function requireLocaleRecord(item, label, fields) {
  for (const locale of locales) {
    if (typeof item[locale] !== "object" || item[locale] === null) throw new Error(`missing locale content: ${label}/${locale}`);
    for (const field of fields) requireNonEmptyString(item[locale][field], `${label}.${locale}.${field}`);
  }
}

function hasImmutableReleaseSource(urls) {
  return urls.some((value) => {
    const url = new URL(value);
    return url.hostname === "github.com" && (/\/releases\/tag\/v?\d/.test(url.pathname) || /\/blob\/v?\d+\.\d+\.\d+\//.test(url.pathname));
  });
}

export function validateFacts(facts, options) {
  if (!Array.isArray(facts) || facts.length === 0) throw new Error("facts must not be empty");
  const ids = new Set();
  for (const item of facts) {
    requireUniqueId(item, ids, "fact");
    if (!statuses.has(item.status)) throw new Error(`invalid fact status: ${item.id}`);
    requireNonEmptyString(item.sinceVersion, `fact.${item.id}.sinceVersion`);
    requireFreshDate(item, "fact", options);
    requireHTTPSUrls(item.sourceUrls, `fact.${item.id}`);
    for (const source of item.sourceUrls) if (!isApprovedMacMLXSource(source)) throw new Error(`unapproved macMLX source for ${item.id}: ${source}`);
    if (item.status === "released" && !hasImmutableReleaseSource(item.sourceUrls)) throw new Error(`immutable release source required: ${item.id}`);
    if (item.status === "development" && !item.sourceUrls.some((value) => {
      const url = new URL(value);
      return url.hostname === "github.com" && url.pathname.startsWith("/magicnight/mac-mlx/blob/main/");
    })) throw new Error(`development source must reference post-tag main: ${item.id}`);
    if (item.status === "planned" && !item.sourceUrls.some((url) => /\/docs\//.test(new URL(url).pathname))) throw new Error(`planned source must reference an approved design or spec: ${item.id}`);
    requireLocaleRecord(item, `fact.${item.id}`, ["title", "summary", "detail"]);
    if (!Array.isArray(item.pageIds)) throw new Error(`fact.pageIds must be an array: ${item.id}`);
    if (new Set(item.pageIds).size !== item.pageIds.length) throw new Error(`duplicate fact pageId: ${item.id}`);
  }
  return ids;
}

export function validateCompetitors(competitors, options) {
  if (!Array.isArray(competitors) || competitors.length === 0) throw new Error("competitors must not be empty");
  const ids = new Set();
  for (const item of competitors) {
    requireUniqueId(item, ids, "competitor");
    requireNonEmptyString(item.name, `competitor.${item.id}.name`);
    requireNonEmptyString(item.verifiedVersion, `competitor.${item.id}.verifiedVersion`);
    requireNotFutureDate(item.snapshotDate, options.today, "competitor snapshot");
    requireFreshDate(item, "competitor", options);
    requireHTTPSUrls(item.officialSources, `competitor.${item.id}`);
    const sourceRule = competitorSourceRules[item.id];
    if (sourceRule === undefined) throw new Error(`no official source ownership rule: ${item.id}`);
    for (const source of item.officialSources) if (!sourceRule(new URL(source))) throw new Error(`unapproved official source for ${item.id}: ${source}`);
    requireLocaleRecord(item, `competitor.${item.id}`, ["summary"]);
    for (const locale of locales) {
      if (!Array.isArray(item[locale].limitations) || item[locale].limitations.length === 0) throw new Error(`missing limitations: ${item.id}/${locale}`);
      item[locale].limitations.forEach((value, index) => requireNonEmptyString(value, `competitor.${item.id}.${locale}.limitations.${index}`));
      for (const key of ["platform", "runtime", "models", "interfaces", "focus"]) {
        requireNonEmptyString(item[locale].dimensions?.[key], `competitor.${item.id}.${locale}.dimensions.${key}`);
      }
    }
  }
  return ids;
}

export function validateFAQs(faqs, factIds) {
  if (!Array.isArray(faqs) || faqs.length !== 8) throw new Error("FAQ registry must contain exactly 8 entries");
  const ids = new Set();
  for (const item of faqs) {
    requireUniqueId(item, ids, "FAQ");
    requireLocaleRecord(item, `FAQ.${item.id}`, ["question", "answer"]);
    if (!Array.isArray(item.factIds) || item.factIds.length === 0) throw new Error(`FAQ must cite facts: ${item.id}`);
    for (const factId of item.factIds) if (!factIds.has(factId)) throw new Error(`unknown fact in FAQ ${item.id}: ${factId}`);
  }
  return ids;
}

export function validateReleases(releases, factIds, options) {
  if (!Array.isArray(releases) || releases.length === 0) throw new Error("releases must not be empty");
  const ids = new Set();
  for (const item of releases) {
    requireUniqueId(item, ids, "release");
    if (item.status !== "released") throw new Error(`invalid release status: ${item.id}`);
    if (!/^\d+\.\d+\.\d+$/.test(item.version ?? "")) throw new Error(`invalid release version: ${item.id}`);
    requireNotFutureDate(item.releaseDate, options.today, "release date");
    requireFreshDate(item, "release", options);
    requireHTTPSUrls(item.officialSources, `release.${item.id}`);
    for (const source of item.officialSources) if (!isApprovedMacMLXSource(source)) throw new Error(`unapproved macMLX release source: ${source}`);
    if (!hasImmutableReleaseSource(item.officialSources)) throw new Error(`immutable release source required: ${item.id}`);
    requireLocaleRecord(item, `release.${item.id}`, ["title", "summary", "compatibilityNotes", "upgradeNotes"]);
    for (const field of ["shippedFactIds", "limitationFactIds", "developmentFactIds", "plannedFactIds"]) {
      if (!Array.isArray(item[field])) throw new Error(`release.${item.id}.${field} must be an array`);
      if (field === "shippedFactIds" && item[field].length === 0) {
        throw new Error(`release.${item.id}.shippedFactIds must not be empty`);
      }
      if (new Set(item[field]).size !== item[field].length) {
        throw new Error(`release.${item.id}.${field} must not contain duplicates`);
      }
      for (const factId of item[field]) {
        if (!factIds.has(factId)) throw new Error(`unknown fact in release ${item.id}: ${factId}`);
      }
    }
    const factsById = options.factsById;
    if (!(factsById instanceof Map)) throw new Error("release validation requires factsById");
    for (const [field, expectedStatus] of [["shippedFactIds", "released"], ["limitationFactIds", "released"], ["developmentFactIds", "development"], ["plannedFactIds", "planned"]]) {
      for (const factId of item[field]) {
        if (factsById.get(factId)?.status !== expectedStatus) throw new Error(`release.${item.id}.${field} must contain only ${expectedStatus} facts: ${factId}`);
      }
    }
  }
  return ids;
}

function referencedIds(block) {
  return [
    ...(block.factIds ?? []),
    ...(block.faqIds ?? []),
    ...(block.competitorIds ?? []),
    ...(block.releaseIds ?? []),
  ];
}

export function validatePages(pages, references, { allowEmptyPages = false } = {}) {
  if (!Array.isArray(pages) || (!allowEmptyPages && pages.length === 0)) throw new Error("pages must not be empty");
  const ids = new Set();
  for (const item of pages) requireUniqueId(item, ids, "page");
  for (const item of pages) {
    requireLocaleRecord(item, `page.${item.id}`, ["title", "description", "directAnswer", "eyebrow"]);
    requireFreshDate(item, "page", references.options);
    if (!Array.isArray(item.relatedIds) || item.relatedIds.length < 2) throw new Error(`page must have at least two relatedIds: ${item.id}`);
    for (const id of item.relatedIds) if (!ids.has(id)) throw new Error(`unknown related page in ${item.id}: ${id}`);
    if (!Array.isArray(item.blocks) || item.blocks.length === 0) throw new Error(`page blocks must not be empty: ${item.id}`);
    for (const block of item.blocks) {
      if (!["paragraph", "illustration", "facts", "code", "table", "faq", "comparison", "release", "sources", "related"].includes(block.type)) throw new Error(`unsupported content block: ${block.type}`);
      if (block.type === "sources" && block.sources !== undefined) throw new Error(`free-form sources are not allowed: ${item.id}`);
      if ((block.factIds ?? []).some((id) => !references.factIds.has(id))) throw new Error(`unknown fact in page ${item.id}`);
      if ((block.faqIds ?? []).some((id) => !references.faqIds.has(id))) throw new Error(`unknown FAQ in page ${item.id}`);
      if ((block.competitorIds ?? []).some((id) => !references.competitorIds.has(id))) throw new Error(`unknown competitor in page ${item.id}`);
      if ((block.releaseIds ?? []).some((id) => !references.releaseIds.has(id))) throw new Error(`unknown release in page ${item.id}`);
      referencedIds(block);
    }
  }
  return ids;
}

export function validateComparisonProfile(profile, factIds) {
  const dimensions = ["platform", "runtime", "models", "interfaces", "focus"];
  if (typeof profile !== "object" || profile === null) throw new Error("macMLX comparison profile is required");
  for (const locale of locales) {
    if (typeof profile[locale] !== "object" || profile[locale] === null) throw new Error(`missing macMLX comparison profile locale: ${locale}`);
    if (Object.keys(profile[locale]).sort().join(",") !== [...dimensions].sort().join(",")) throw new Error(`invalid macMLX comparison dimensions: ${locale}`);
    for (const dimension of dimensions) {
      const cell = profile[locale][dimension];
      requireNonEmptyString(cell?.text, `macMLX comparison ${locale}.${dimension}.text`);
      if (!Array.isArray(cell.sourceFactIds) || cell.sourceFactIds.length === 0) throw new Error(`macMLX comparison cell must cite facts: ${locale}.${dimension}`);
      for (const factId of cell.sourceFactIds) if (!factIds.has(factId)) throw new Error(`unknown macMLX comparison fact: ${locale}.${dimension}/${factId}`);
    }
  }
}

export function validateFactPageReferences(facts, pages) {
  const directUsage = new Map(facts.map((fact) => [fact.id, []]));
  for (const page of pages) {
    for (const block of page.blocks.filter((item) => item.type === "facts")) {
      for (const factId of block.factIds) directUsage.get(factId)?.push(page.id);
    }
  }
  for (const fact of facts) {
    const declared = [...fact.pageIds].sort();
    const actual = directUsage.get(fact.id).sort();
    if (declared.join(",") !== actual.join(",")) throw new Error(`fact/page direct reference mismatch: ${fact.id}; declared=${declared.join(",")}; actual=${actual.join(",")}`);
  }
}

export function validateReleaseIdentity(project, releases) {
  const current = releases.find((item) => item.status === "released" && item.version === project.currentVersion);
  if (current === undefined) throw new Error(`project.currentVersion has no matching release: ${project.currentVersion}`);
  if (current.releaseDate !== project.releaseDate) throw new Error(`project.releaseDate does not match release registry: ${project.releaseDate}`);
}

export function validateContentHub({ facts, competitors, faqs, releases, pages, macmlxComparisonProfile }, options) {
  const factIds = validateFacts(facts, options);
  const competitorIds = validateCompetitors(competitors, options);
  const faqIds = validateFAQs(faqs, factIds);
  const factsById = new Map(facts.map((fact) => [fact.id, fact]));
  const releaseIds = validateReleases(releases, factIds, { ...options, factsById });
  const pageIds = validatePages(pages, { factIds, competitorIds, faqIds, releaseIds, options }, options);
  validateComparisonProfile(macmlxComparisonProfile, factIds);
  if (!options.allowEmptyPages) {
    for (const fact of facts) for (const pageId of fact.pageIds) if (!pageIds.has(pageId)) throw new Error(`unknown pageId in fact ${fact.id}: ${pageId}`);
    validateFactPageReferences(facts, pages);
  }
  return { factIds, competitorIds, faqIds, releaseIds, pageIds };
}
