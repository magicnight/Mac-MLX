const tokenPattern = /{{\s*([A-Za-z][A-Za-z0-9_.-]*)\s*}}/g;

export function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function renderTokens(template, values) {
  const tokens = [...template.matchAll(tokenPattern)].map((match) => match[1]);
  const tokenSet = new Set(tokens);

  for (const token of tokenSet) {
    if (!Object.hasOwn(values, token)) {
      throw new Error(`Missing template token: ${token}`);
    }
  }

  for (const key of Object.keys(values)) {
    if (!tokenSet.has(key)) {
      throw new Error(`Unused template token: ${key}`);
    }
  }

  const rendered = template.replace(tokenPattern, (_match, token) => escapeHTML(values[token]));
  if (/{{|}}/.test(rendered)) {
    throw new Error("Unresolved or invalid template token");
  }
  return rendered;
}

function sourceOpeningTags(template) {
  const tags = [];
  let cursor = 0;

  while ((cursor = template.indexOf("<", cursor)) !== -1) {
    const marker = template[cursor + 1];
    if (marker === "/" || marker === "!" || marker === "?") {
      cursor += 2;
      continue;
    }

    let quote = null;
    let end = cursor + 1;
    for (; end < template.length; end += 1) {
      const character = template[end];
      if (quote !== null) {
        if (character === quote) quote = null;
      } else if (character === '"' || character === "'") {
        quote = character;
      } else if (character === ">") {
        break;
      }
    }

    if (end >= template.length) throw new Error("Unclosed HTML opening tag");
    const opening = template.slice(cursor, end + 1);
    const tagName = opening.match(/^<([A-Za-z][\w:-]*)/)?.[1];
    if (tagName !== undefined) tags.push({ start: cursor, end: end + 1, opening, tagName });
    cursor = end + 1;
  }
  return tags;
}

function matchingClosingTag(template, tagName, contentStart) {
  let depth = 1;
  let cursor = contentStart;
  const normalizedName = tagName.toLowerCase();

  while ((cursor = template.indexOf("<", cursor)) !== -1) {
    if (template.startsWith("<!--", cursor)) {
      const commentEnd = template.indexOf("-->", cursor + 4);
      if (commentEnd === -1) throw new Error("Unclosed HTML comment");
      cursor = commentEnd + 3;
      continue;
    }

    let quote = null;
    let end = cursor + 1;
    for (; end < template.length; end += 1) {
      const character = template[end];
      if (quote !== null) {
        if (character === quote) quote = null;
      } else if (character === '"' || character === "'") {
        quote = character;
      } else if (character === ">") {
        break;
      }
    }
    if (end >= template.length) throw new Error("Unclosed HTML tag");

    const tag = template.slice(cursor, end + 1);
    const closingName = tag.match(/^<\/\s*([A-Za-z][\w:-]*)/)?.[1]?.toLowerCase();
    const openingName = tag.match(/^<\s*([A-Za-z][\w:-]*)/)?.[1]?.toLowerCase();
    if (closingName === normalizedName) {
      depth -= 1;
      if (depth === 0) return { start: cursor, end: end + 1 };
    } else if (openingName === normalizedName && !/\/\s*>$/.test(tag)) {
      depth += 1;
    }
    cursor = end + 1;
  }
  return null;
}

export function localizeBilingualElements(template, locale) {
  const attribute = locale === "en" ? "data-en" : locale === "zh-Hans" ? "data-zh" : null;
  if (attribute === null) throw new Error(`Unsupported locale: ${locale}`);

  const bilingualTags = sourceOpeningTags(template).filter(({ opening }) => /\bdata-(?:en|zh)="/.test(opening));
  for (const { opening } of bilingualTags) {
    const englishValue = opening.match(/\bdata-en="([^"]*)"/)?.[1];
    const chineseValue = opening.match(/\bdata-zh="([^"]*)"/)?.[1];
    if (englishValue === undefined) throw new Error("Bilingual element is missing data-en");
    if (chineseValue === undefined) throw new Error("Bilingual element is missing data-zh");
    if (englishValue.trim() === "") throw new Error("Bilingual element has an empty data-en value");
    if (chineseValue.trim() === "") throw new Error("Bilingual element has an empty data-zh value");
  }

  let rendered = template;
  for (const { start, end, opening, tagName } of bilingualTags.reverse()) {
    const localizedContent = opening.match(new RegExp(`\\b${attribute}="([^"]*)"`))?.[1];
    if (localizedContent === undefined) {
      throw new Error(`Bilingual element is missing ${attribute}`);
    }
    const closingTag = matchingClosingTag(rendered, tagName, end);
    if (closingTag === null) throw new Error(`Bilingual <${tagName}> element is missing its closing tag`);
    const closing = rendered.slice(closingTag.start, closingTag.end);
    const cleanOpening = opening.replace(/\s+data-(?:en|zh)="[^"]*"/g, "");
    rendered = `${rendered.slice(0, start)}${cleanOpening}${localizedContent}${closing}${rendered.slice(closingTag.end)}`;
  }

  if (/\bdata-(?:en|zh)=/.test(rendered)) {
    throw new Error("Unable to localize every bilingual element");
  }
  return rendered;
}
