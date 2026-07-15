import { escapeHTML } from "./localize.mjs";

export const supportedContentBlocks = Object.freeze([
  "paragraph", "illustration", "facts", "code", "table", "faq", "comparison", "release", "sources", "related",
]);

const statusLabels = Object.freeze({
  en: Object.freeze({ released: "Released", development: "In development", planned: "Planned" }),
  "zh-Hans": Object.freeze({ released: "已发布", development: "开发中", planned: "规划中" }),
});

function localized(value, locale, label) {
  if (typeof value === "string") return value;
  const result = value?.[locale];
  if (typeof result !== "string" || result.trim() === "") throw new Error(`missing localized ${label}: ${locale}`);
  return result;
}

function heading(block, context) {
  return `<h2>${escapeHTML(localized(block.heading, context.locale, `${block.type} heading`))}</h2>`;
}

function requireItem(map, id, label) {
  const item = map?.[id];
  if (item === undefined) throw new Error(`unknown ${label}: ${id}`);
  return item;
}

function renderParagraph(block, context) {
  return `<p class="article-paragraph">${escapeHTML(localized(block.text, context.locale, "paragraph"))}</p>`;
}

function renderIllustration(block, context) {
  const alt = localized(block.alt, context.locale, "illustration alt");
  const caption = localized(block.caption, context.locale, "illustration caption");
  if (!/^\/assets\/images\/[a-z0-9/_-]+\.webp$/.test(block.src ?? "")) throw new Error(`invalid illustration source: ${block.src}`);
  if (!Number.isInteger(block.width) || block.width < 1 || !Number.isInteger(block.height) || block.height < 1) throw new Error("illustration dimensions must be positive integers");
  return `<figure class="article-illustration"><img src="${escapeHTML(block.src)}" width="${block.width}" height="${block.height}" alt="${escapeHTML(alt)}" loading="lazy" decoding="async"><figcaption>${escapeHTML(caption)}</figcaption></figure>`;
}

function factCard(fact, context) {
  const copy = fact[context.locale];
  if (copy === undefined) throw new Error(`missing locale content for fact: ${fact.id}/${context.locale}`);
  const status = statusLabels[context.locale]?.[fact.status];
  if (status === undefined) throw new Error(`unknown fact status: ${fact.status}`);
  return `<article class="fact-card" data-status="${escapeHTML(fact.status)}"><div class="fact-card__meta"><span class="status-label">${escapeHTML(status)}</span><span>${escapeHTML(fact.sinceVersion)}</span></div><h3>${escapeHTML(copy.title)}</h3><p>${escapeHTML(copy.summary)}</p><p class="fact-detail">${escapeHTML(copy.detail)}</p><p class="fact-verified"><span>${context.locale === "en" ? "Verified" : "核验"}</span> <time datetime="${escapeHTML(fact.lastVerified)}">${escapeHTML(fact.lastVerified)}</time></p></article>`;
}

function renderFacts(block, context) {
  const cards = block.factIds.map((id) => factCard(requireItem(context.factsById, id, "fact"), context)).join("");
  return `<section class="content-section">${heading(block, context)}<div class="fact-cards">${cards}</div></section>`;
}

function renderCode(block, context) {
  const language = block.language ?? "text";
  if (!/^[a-z0-9-]+$/.test(language)) throw new Error(`invalid code language: ${language}`);
  const intro = block.intro === undefined ? "" : `<p>${escapeHTML(localized(block.intro, context.locale, "code intro"))}</p>`;
  return `<section class="content-section">${heading(block, context)}${intro}<pre class="article-code"><code class="language-${escapeHTML(language)}">${escapeHTML(block.code)}</code></pre></section>`;
}

function renderTable(block, context) {
  const headers = block.headers?.[context.locale];
  const rows = block.rows?.[context.locale];
  if (!Array.isArray(headers) || headers.length < 2 || !Array.isArray(rows) || rows.length === 0) throw new Error(`invalid localized table: ${context.locale}`);
  const tableRows = rows.map((row) => {
    if (!Array.isArray(row) || row.length !== headers.length) throw new Error("table row width must match headers");
    return `<tr>${row.map((cell, index) => index === 0 ? `<th scope="row">${escapeHTML(cell)}</th>` : `<td>${escapeHTML(cell)}</td>`).join("")}</tr>`;
  }).join("");
  return `<section class="content-section">${heading(block, context)}<div class="table-wrap"><table class="article-table"><caption>${escapeHTML(localized(block.caption, context.locale, "table caption"))}</caption><thead><tr>${headers.map((cell) => `<th scope="col">${escapeHTML(cell)}</th>`).join("")}</tr></thead><tbody>${tableRows}</tbody></table></div></section>`;
}

function renderFAQ(block, context) {
  const items = block.faqIds.map((id) => {
    const copy = requireItem(context.faqsById, id, "FAQ")[context.locale];
    if (copy === undefined) throw new Error(`missing FAQ locale: ${id}/${context.locale}`);
    return `<details><summary>${escapeHTML(copy.question)}</summary><p>${escapeHTML(copy.answer)}</p></details>`;
  }).join("");
  return `<section class="content-section">${heading(block, context)}<div class="faq-list">${items}</div></section>`;
}

const comparisonLabels = Object.freeze({
  en: Object.freeze({ caption: "Dated factual comparison", dimension: "Dimension", platform: "Platform", runtime: "Core runtime", models: "Model workflow", interfaces: "Interfaces", focus: "Factual focus / audience", verified: "Snapshot" }),
  "zh-Hans": Object.freeze({ caption: "带日期的事实对比", dimension: "维度", platform: "平台", runtime: "核心运行时", models: "模型工作流", interfaces: "接口", focus: "事实重点 / 受众", verified: "快照" }),
});

function renderComparison(block, context) {
  const labels = comparisonLabels[context.locale];
  const competitors = block.competitorIds.map((id) => requireItem(context.competitorsById, id, "competitor"));
  const headers = ["macMLX", ...competitors.map((item) => item.name)];
  const macmlx = context.macmlxComparisonProfile?.[context.locale];
  if (macmlx === undefined) throw new Error(`macMLX comparison profile is required: ${context.locale}`);
  const rows = ["platform", "runtime", "models", "interfaces", "focus"].map((key) => {
    const cell = macmlx[key];
    if (typeof cell?.text !== "string" || !Array.isArray(cell.sourceFactIds) || cell.sourceFactIds.length === 0) throw new Error(`invalid macMLX comparison profile cell: ${context.locale}.${key}`);
    for (const factId of cell.sourceFactIds) requireItem(context.factsById, factId, "macMLX comparison fact");
    return `<tr><th scope="row">${escapeHTML(labels[key])}</th><td>${escapeHTML(cell.text)}</td>${competitors.map((item) => `<td>${escapeHTML(item[context.locale].dimensions[key])}</td>`).join("")}</tr>`;
  }).join("");
  const notes = competitors.map((item) => `<li><a href="${escapeHTML(item.officialSources[0])}">${escapeHTML(item.name)} ${escapeHTML(item.verifiedVersion)}</a> · <time datetime="${escapeHTML(item.snapshotDate)}">${escapeHTML(item.snapshotDate)}</time></li>`).join("");
  const limitationsHeading = context.locale === "en" ? "Documented limitations" : "已记录限制";
  const limitations = competitors.map((item) => `<article><h4>${escapeHTML(item.name)}</h4><ul>${item[context.locale].limitations.map((value) => `<li>${escapeHTML(value)}</li>`).join("")}</ul></article>`).join("");
  return `<section class="content-section">${heading(block, context)}<div class="table-wrap"><table class="comparison-table"><caption>${escapeHTML(labels.caption)}</caption><thead><tr><th scope="col">${escapeHTML(labels.dimension)}</th>${headers.map((name) => `<th scope="col">${escapeHTML(name)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div><div class="comparison-limitations"><h3>${escapeHTML(limitationsHeading)}</h3>${limitations}</div><div class="comparison-snapshots"><h3>${escapeHTML(labels.verified)}</h3><ul>${notes}</ul></div></section>`;
}

function renderRelease(block, context) {
  const sections = [];
  for (const id of block.releaseIds) {
    const item = requireItem(context.releasesById, id, "release");
    const copy = item[context.locale];
    const categories = [
      [context.locale === "en" ? "Shipped" : "已交付", item.shippedFactIds],
      [context.locale === "en" ? "Current limitations" : "当前限制", item.limitationFactIds],
      [context.locale === "en" ? "Development after the tag" : "标签后的开发工作", item.developmentFactIds],
      [context.locale === "en" ? "Planned" : "规划中", item.plannedFactIds],
    ];
    const groups = categories
      .filter(([, ids]) => ids.length > 0)
      .map(([label, ids]) => `<section class="release-status-group"><h3>${escapeHTML(label)}</h3><div class="fact-cards">${ids.map((factId) => factCard(requireItem(context.factsById, factId, "fact"), context)).join("")}</div></section>`)
      .join("");
    const notesHeading = context.locale === "en" ? "Compatibility and upgrade notes" : "兼容性与升级说明";
    const compatibilityLabel = context.locale === "en" ? "Compatibility" : "兼容性";
    const upgradeLabel = context.locale === "en" ? "Upgrade" : "升级";
    const releaseNotes = `<section class="release-notes"><h3>${notesHeading}</h3><dl><div><dt>${compatibilityLabel}</dt><dd>${escapeHTML(copy.compatibilityNotes)}</dd></div><div><dt>${upgradeLabel}</dt><dd>${escapeHTML(copy.upgradeNotes)}</dd></div></dl></section>`;
    sections.push(`<article class="release-entry"><header><h3>${escapeHTML(copy.title)}</h3><p>${escapeHTML(copy.summary)}</p><p><time datetime="${escapeHTML(item.releaseDate)}">${escapeHTML(item.releaseDate)}</time></p></header>${releaseNotes}${groups}</article>`);
  }
  return `<section class="content-section">${heading(block, context)}${sections.join("")}</section>`;
}

function renderSources(block, context) {
  if (block.sources !== undefined) throw new Error("free-form sources are not allowed");
  const sources = [];
  for (const id of block.factIds ?? []) {
    const item = requireItem(context.factsById, id, "fact");
    item.sourceUrls.forEach((url) => sources.push({ label: { en: item.en.title, "zh-Hans": item["zh-Hans"].title }, url }));
  }
  for (const id of block.competitorIds ?? []) {
    const item = requireItem(context.competitorsById, id, "competitor");
    item.officialSources.forEach((url) => sources.push({ label: `${item.name} · ${new URL(url).hostname}`, url }));
  }
  for (const id of block.releaseIds ?? []) {
    const item = requireItem(context.releasesById, id, "release");
    item.officialSources.forEach((url) => sources.push({ label: `${item[context.locale].title} · ${new URL(url).hostname}`, url }));
  }
  const unique = [...new Map(sources.map((source) => [source.url, source])).values()];
  const list = unique.map((source) => `<li><a href="${escapeHTML(source.url)}">${escapeHTML(localized(source.label, context.locale, "source label"))}</a></li>`).join("");
  return `<section class="content-section sources">${heading(block, context)}<ol>${list}</ol></section>`;
}

function renderRelated(block, context) {
  const links = block.relatedIds.map((id) => {
    const item = requireItem(context.pagesById, id, "related page");
    const copy = item[context.locale];
    return `<a href="${escapeHTML(item.paths[context.locale])}"><strong>${escapeHTML(copy.title)}</strong><span>${escapeHTML(copy.description)}</span></a>`;
  }).join("");
  const aria = context.locale === "en" ? "Related macMLX pages" : "macMLX 相关页面";
  return `<nav class="related-pages" aria-label="${aria}">${heading(block, context)}<div>${links}</div></nav>`;
}

export function renderContentBlocks(blocks, context) {
  if (!Array.isArray(blocks)) throw new Error("content blocks must be an array");
  return blocks.map((block) => {
    if (!supportedContentBlocks.includes(block.type)) throw new Error(`unsupported content block: ${block.type}`);
    if (block.type === "paragraph") return renderParagraph(block, context);
    if (block.type === "illustration") return renderIllustration(block, context);
    if (block.type === "facts") return renderFacts(block, context);
    if (block.type === "code") return renderCode(block, context);
    if (block.type === "table") return renderTable(block, context);
    if (block.type === "faq") return renderFAQ(block, context);
    if (block.type === "comparison") return renderComparison(block, context);
    if (block.type === "release") return renderRelease(block, context);
    if (block.type === "sources") return renderSources(block, context);
    return renderRelated(block, context);
  }).join("\n");
}
