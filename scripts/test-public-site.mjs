import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getAttribute(tag, name) {
  return tag.match(new RegExp(`\\b${escapeRegExp(name)}="([^"]*)"`))?.[1];
}

function countClassToken(source, token) {
  return [...source.matchAll(/\bclass="([^"]*)"/g)].filter((match) =>
    match[1].split(/\s+/).includes(token),
  ).length;
}

function extractMediaBlock(source, query) {
  const opening = new RegExp(`@media\\s+${escapeRegExp(query)}\\s*\\{`, "g");
  const match = opening.exec(source);
  assert.ok(match, `missing @media ${query}`);

  const contentStart = opening.lastIndex;
  let depth = 1;
  for (let index = contentStart; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    if (source[index] !== "}") continue;

    depth -= 1;
    if (depth === 0) return source.slice(contentStart, index);
  }

  assert.fail(`unclosed @media ${query}`);
}

function extractRule(source, selector) {
  const match = source.match(new RegExp(`${escapeRegExp(selector)}\\s*\\{([^}]*)\\}`));
  assert.ok(match, `missing ${selector} rule`);
  return match[1];
}

function extractProperty(rule, property) {
  return rule.match(new RegExp(`${escapeRegExp(property)}\\s*:\\s*([^;]+);`))?.[1].trim();
}

function parseHexColor(value) {
  const match = value.match(/^#([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i);
  assert.ok(match, `invalid hex color: ${value}`);
  return match.slice(1).map((channel) => Number.parseInt(channel, 16));
}

function relativeLuminance(color) {
  const linear = color.map((channel) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return (0.2126 * linear[0]) + (0.7152 * linear[1]) + (0.0722 * linear[2]);
}

function contrastRatio(foreground, background) {
  const foregroundLuminance = relativeLuminance(foreground);
  const backgroundLuminance = relativeLuminance(background);
  const lighter = Math.max(foregroundLuminance, backgroundLuminance);
  const darker = Math.min(foregroundLuminance, backgroundLuminance);
  return (lighter + 0.05) / (darker + 0.05);
}

const root = new URL("../", import.meta.url);
const html = await readFile(new URL("public/index.html", root), "utf8");
const zhHTML = await readFile(new URL("public/zh/index.html", root), "utf8");
const homeTemplate = await readFile(new URL("site/templates/home.html", root), "utf8");
const css = await readFile(new URL("public/assets/css/main.css", root), "utf8");
const js = await readFile(new URL("public/assets/js/main.js", root), "utf8");
const llms = await readFile(new URL("public/llms.txt", root), "utf8");
const ogImage = await readFile(new URL("public/assets/og-image.svg", root), "utf8");

assert.match(
  html,
  /<link rel="stylesheet" href="\/assets\/css\/main\.css\?v=4">\s*<noscript>\s*<link rel="stylesheet" href="\/assets\/css\/no-js\.css\?v=2026071004">\s*<\/noscript>/,
  "the no-JavaScript stylesheet should load after the main stylesheet",
);

const noJSCSSURL = new URL("public/assets/css/no-js.css", root);
const noJSCSSMetadata = await stat(noJSCSSURL);
const noJSCSS = await readFile(noJSCSSURL, "utf8");
assert.ok(noJSCSSMetadata.isFile(), "the no-JavaScript stylesheet should be a file");
assert.ok(noJSCSSMetadata.size > 0, "the no-JavaScript stylesheet should not be empty");

const noJSRevealCSS = extractRule(noJSCSS, ".reveal");
assert.match(noJSRevealCSS, /opacity:\s*1\s*;/);
assert.match(noJSRevealCSS, /transform:\s*none\s*;/);
assert.match(extractRule(noJSCSS, ".engine-story"), /display:\s*block\s*;/);
const noJSStepsCSS = extractRule(noJSCSS, ".engine-steps");
assert.match(noJSStepsCSS, /display:\s*grid\s*;/);
assert.match(noJSStepsCSS, /gap:\s*48px\s*;/);
const noJSStepCSS = extractRule(noJSCSS, ".engine-step");
assert.match(noJSStepCSS, /min-height:\s*auto\s*;/);
assert.match(noJSStepCSS, /opacity:\s*1\s*;/);
assert.match(noJSStepCSS, /transform:\s*none\s*;/);
assert.match(noJSStepCSS, /clip-path:\s*none\s*;/);
assert.match(noJSStepCSS, /transition:\s*none\s*;/);
const noJSMobileFigureCSS = extractRule(noJSCSS, ".engine-mobile-figure");
assert.match(noJSMobileFigureCSS, /aspect-ratio:\s*16\s*\/\s*9\s*;/);
assert.match(noJSMobileFigureCSS, /margin:\s*28px 0 0\s*;/);
assert.match(noJSMobileFigureCSS, /display:\s*block\s*;/);
assert.match(noJSMobileFigureCSS, /border:\s*1px solid var\(--line-strong\)\s*;/);
assert.match(noJSMobileFigureCSS, /border-radius:\s*15px\s*;/);
assert.match(noJSMobileFigureCSS, /overflow:\s*hidden\s*;/);
assert.match(noJSMobileFigureCSS, /opacity:\s*1\s*;/);
assert.match(noJSMobileFigureCSS, /transform:\s*none\s*;/);
assert.match(noJSMobileFigureCSS, /clip-path:\s*none\s*;/);
assert.match(noJSMobileFigureCSS, /transition:\s*none\s*;/);
assert.match(extractRule(noJSCSS, ".engine-stage"), /display:\s*none\s*;/);

const inactiveEngineStepCSS = extractRule(css, ".engine-step");
const inactiveOpacityMatch = inactiveEngineStepCSS.match(/opacity:\s*([\d.]+)\s*;/);
assert.ok(inactiveOpacityMatch, "inactive engine steps should define an opacity");
const inactiveOpacity = Number(inactiveOpacityMatch[1]);
assert.ok(inactiveOpacity >= 0.72, "inactive engine steps should remain readable");
const activeOpacityMatch = extractRule(css, ".engine-step.is-active").match(/opacity:\s*([\d.]+)\s*;/);
assert.ok(activeOpacityMatch, "active engine steps should define an opacity");
assert.equal(Number(activeOpacityMatch[1]), 1, "active engine steps should remain fully opaque");

const engineSectionCSS = extractRule(css, ".engine-section");
assert.equal(extractProperty(engineSectionCSS, "color"), "var(--ink)", "the engine section should follow the active text theme");
assert.equal(extractProperty(engineSectionCSS, "background"), "var(--paper-deep)", "the engine section should follow the active surface theme");
assert.match(engineSectionCSS, /border-top:\s*1px solid var\(--line\)\s*;/, "the engine divider should follow the active theme");
assert.equal(extractProperty(extractRule(css, ".engine-meta"), "color"), "var(--ink)", "engine metadata should stay readable in both themes");
assert.equal(extractProperty(extractRule(css, ".engine-step p"), "color"), "var(--ink)", "engine copy should stay readable in both themes");
assert.equal(extractProperty(extractRule(css, ".engine-visuals"), "background"), "var(--paper)", "the engine stage should follow the active theme");

for (const [theme, backgroundHex, textHex] of [
  ["light", "#e8e4da", "#131613"],
  ["dark", "#191c19", "#f1f0e9"],
]) {
  const engineBackground = parseHexColor(backgroundHex);
  const engineText = parseHexColor(textHex);
  const compositedEngineText = engineText.map((channel, index) =>
    (inactiveOpacity * channel) + ((1 - inactiveOpacity) * engineBackground[index]),
  );
  const engineTextContrast = contrastRatio(compositedEngineText, engineBackground);
  assert.ok(
    engineTextContrast >= 4.5,
    `${theme} inactive engine text contrast should be at least 4.5:1, got ${engineTextContrast.toFixed(2)}:1`,
  );
}

const bilingualNodes = homeTemplate.match(/<[^>]+data-en="[^"]+"[^>]+data-zh="[^"]+"[^>]*>/g) ?? [];
assert.equal(bilingualNodes.length, 97, "the tracked source should preserve all 97 approved bilingual nodes");
assert.equal(homeTemplate.match(/data-en=/g)?.length, homeTemplate.match(/data-zh=/g)?.length, "English and Chinese source copy counts should match");

for (const node of bilingualNodes) {
  assert.match(node, /data-en="[^"]+"/, "every bilingual node needs English copy");
  assert.match(node, /data-zh="[^"]+"/, "every bilingual node needs Chinese copy");
}

for (const [locale, localizedHTML] of [["en", html], ["zh-Hans", zhHTML]]) {
  assert.doesNotMatch(localizedHTML, /\bdata-(?:en|zh|lang-option)=/, `${locale} output should not include runtime translation data`);
  assert.doesNotMatch(localizedHTML, /<body[^>]+data-lang=/, `${locale} output should not include runtime language state`);
  assert.doesNotMatch(localizedHTML, /(?:src|href)="assets\//, `${locale} assets should be root-absolute`);
  assert.equal(countClassToken(localizedHTML, "engine-step"), 5, `${locale} engine story needs five chapters`);
  assert.equal(countClassToken(localizedHTML, "engine-mobile-figure"), 5, `${locale} no-JavaScript story needs five chapter images`);
  assert.equal(localizedHTML.match(/data-engine-panel=/g)?.length, 5, `${locale} desktop stage needs five visual panels`);
}

for (const section of ["why", "features", "progress", "quickstart"]) {
  assert.match(html, new RegExp(`id="${section}"`), `missing #${section} section`);
}

const featuresPosition = html.indexOf('id="features"');
const enginePosition = html.indexOf('id="engine"');
const progressPosition = html.indexOf('id="progress"');

assert.ok(enginePosition > featuresPosition, "#engine should follow #features");
assert.ok(enginePosition < progressPosition, "#engine should precede #progress");
assert.equal(countClassToken(html, "engine-step"), 5, "engine story needs five chapters");
assert.equal(countClassToken(html, "engine-mobile-figure"), 5, "the no-JavaScript story needs five chapter images");
assert.equal(countClassToken(html, "engine-status"), 5, "every engine chapter needs a status");
assert.equal(countClassToken(html, "engine-tags"), 5, "every engine chapter needs tags");
assert.equal(html.match(/data-engine-panel=/g)?.length, 5, "desktop stage needs five visual panels");
assert.equal(html.match(/data-engine-progress=/g)?.length, 5, "progress rail needs five segments");

const engineAssets = [
  "mac-silicon-foundation.webp",
  "moe-routing.webp",
  "paged-kv-memory.webp",
  "adaptive-runtime.webp",
  "generation-controls.webp",
];

for (const asset of engineAssets) {
  const assetURL = new URL(`public/assets/images/engine/${asset}`, root);
  const metadata = await stat(assetURL);
  assert.ok(metadata.isFile(), `engine image should be a file: ${asset}`);
  assert.ok(metadata.size > 0, `empty engine image: ${asset}`);
  assert.match(
    html,
    new RegExp(escapeRegExp(`/assets/images/engine/${asset}`)),
    `unreferenced engine image: ${asset}`,
  );
}

const engineImageTags = (html.match(/<img\b[^>]*>/g) ?? []).filter((tag) =>
  getAttribute(tag, "class")?.split(/\s+/).includes("engine-image"),
);
assert.equal(engineImageTags.length, 10, "engine story needs ten image instances");
for (const tag of engineImageTags) {
  assert.equal(getAttribute(tag, "width"), "1600", "engine images should declare a 1600px width");
  assert.equal(getAttribute(tag, "height"), "900", "engine images should declare a 900px height");
}

for (const asset of engineAssets) {
  const assetPath = `/assets/images/engine/${asset}`;
  const assetTags = engineImageTags.filter((tag) => getAttribute(tag, "src") === assetPath);
  assert.equal(assetTags.length, 2, `engine image should appear in desktop and mobile views: ${asset}`);

  assert.ok(
    assetTags.every((tag) => getAttribute(tag, "loading") === "lazy" && getAttribute(tag, "fetchpriority") === undefined),
    `${asset} should remain below-fold lazy content without high fetch priority`,
  );
}

const mobileCSS = extractMediaBlock(css, "(max-width: 720px)");
const tabletCSS = extractMediaBlock(css, "(max-width: 1020px)");
const reducedMotionCSS = extractMediaBlock(css, "(prefers-reduced-motion: reduce)");
assert.match(css, /\.engine-stage \{[^}]*position: sticky/);
assert.match(css, /\.engine-stage \{[^}]*height:\s*calc\(100vh - 118px\);[^}]*height:\s*calc\(100dvh - 118px\);/);
assert.match(css, /\.engine-stage \{[^}]*min-height:\s*min\(620px, calc\(100dvh - 118px\)\);/);
assert.match(tabletCSS, /\.engine-story\s*\{[^}]*grid-template-columns:\s*minmax\(0, \.86fr\) minmax\(0, 1\.14fr\)\s*;/);
assert.doesNotMatch(tabletCSS, /grid-template-columns:[^;]*(?:280px|420px)/);
assert.match(tabletCSS, /\.engine-stage\s*\{[^}]*min-height:\s*min\(520px, calc\(100dvh - 118px\)\)\s*;/);
assert.match(mobileCSS, /\.engine-stage\s*\{[^}]*\bdisplay:\s*none\s*;/);
assert.match(css, /\.engine-story\.is-enhanced \.engine-mobile-figure/);
assert.match(css, /\.engine-story\.is-static\s*\{[^}]*display:\s*block\s*;/);
assert.match(css, /\.engine-story\.is-static \.engine-steps\s*\{[^}]*display:\s*grid\s*;[^}]*gap:\s*\d+px\s*;/);
assert.match(css, /\.engine-story\.is-static \.engine-step\s*\{[^}]*min-height:\s*auto\s*;[^}]*opacity:\s*1\s*;[^}]*transform:\s*none\s*;/);
assert.match(css, /\.engine-story\.is-static \.engine-mobile-figure\s*\{[^}]*aspect-ratio:\s*16\s*\/\s*9\s*;[^}]*display:\s*block\s*;[^}]*border:\s*1px solid var\(--line-strong\)\s*;[^}]*border-radius:\s*15px\s*;[^}]*opacity:\s*1\s*;/);
assert.match(css, /\.engine-story\.is-static \.engine-stage\s*\{[^}]*display:\s*none\s*;/);
assert.match(reducedMotionCSS, /\.engine-visual\b/);
assert.match(reducedMotionCSS, /\.engine-stage\s*\{[^}]*display:\s*none\s*;/);
assert.match(reducedMotionCSS, /\.engine-story\s*\{[^}]*display:\s*block\s*;/);
assert.match(reducedMotionCSS, /\.engine-steps\s*\{[^}]*display:\s*grid\s*;[^}]*gap:\s*\d+px\s*;/);
assert.match(reducedMotionCSS, /\.engine-step\s*\{[^}]*min-height:\s*auto\s*;/);
assert.match(reducedMotionCSS, /\.engine-mobile-figure\s*\{[^}]*aspect-ratio:\s*16\s*\/\s*9\s*;[^}]*display:\s*block\s*;[^}]*border:\s*1px solid var\(--line-strong\)\s*;[^}]*border-radius:\s*15px\s*;/);
assert.match(reducedMotionCSS, /\.engine-mobile-figure[^}]*opacity:\s*1\s*;[^}]*transform:\s*none\s*;[^}]*transition:\s*none\s*;/);
assert.match(js, /function initialiseEngineStory\(\)/);
assert.match(js, /story\.dataset\.engineStep/);
assert.match(js, /story\.classList\.add\("is-enhanced"\)/);
assert.match(js, /if \(!\("IntersectionObserver" in window\)\) \{[\s\S]*?story\.classList\.add\("is-static"\)/);
assert.match(js, /activateClosestStep\(\);\s*steps\.find\([\s\S]*?story\.dataset\.engineStep[\s\S]*?\)\?\.classList\.add\("is-visible"\)/);

for (const match of html.matchAll(/href="#([^"]+)"/g)) {
  assert.match(html, new RegExp(`id="${match[1]}"`), `broken internal link: #${match[1]}`);
}

for (const localizedHTML of [html, zhHTML]) {
  for (const match of localizedHTML.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)) {
    assert.doesNotThrow(() => JSON.parse(match[1]), "structured data should contain valid JSON");
  }
}

assert.match(html, /Swift-native in-process engine/);
assert.doesNotMatch(html, /Swift 原生进程内引擎/);
assert.match(zhHTML, /Swift 原生进程内引擎/);
assert.doesNotMatch(zhHTML, /Swift-native in-process engine/);
assert.match(html, /Current release[\s\S]*?<strong>v0\.6\.2<\/strong>/);
assert.match(zhHTML, /当前版本[\s\S]*?<strong>v0\.6\.2<\/strong>/);
assert.match(html, /Agent and API tool loops with structured output controls/);
assert.match(html, /Continuous batching, LCP reuse, and speculative decoding runtime/);
assert.match(html, /Track G distinguishes tested results from theoretical estimates/);
assert.match(html, /v0\.6\.1 hardening and model-family templates/);
assert.match(zhHTML, /智能体与 API 工具循环以及结构化输出控制/);
assert.match(zhHTML, /连续批处理、LCP 复用与投机解码运行时/);
assert.match(zhHTML, /Track G 明确区分实测结果与理论估算/);
assert.match(zhHTML, /v0\.6\.1 加固与模型家族模板/);
assert.match(llms, /Latest release: v0\.6\.2/);
assert.match(ogImage, /v0\.6\.2/);

const combined = `${html}\n${zhHTML}\n${llms}\n${ogImage}`;
for (const staleClaim of [
  /v0\.3\.7/,
  /only native GUI/i,
  /唯一原生 GUI/,
  /Starting v0\.4\.1, macMLX will/,
]) {
  assert.doesNotMatch(combined, staleClaim, `stale content found: ${staleClaim}`);
}

assert.doesNotMatch(`${html}\n${llms}`, /brew install/, "unfinished Homebrew installation should not be advertised");
assert.match(css, /\.features-section \{ color: var\(--ink\); background: var\(--paper-deep\); \}/);
assert.match(css, /\.site-footer \{ color: var\(--ink\); background: var\(--paper-deep\); \}/);
assert.doesNotMatch(css, /--panel(?:-soft|-text|-muted)?:/, "section colors should follow the active page theme");
assert.match(css, /\.progress-section \{ background: var\(--paper\); border-top: 1px solid var\(--line\); \}/);
assert.match(css, /\.feature-card::before/);
assert.match(css, /\.mini-code \{[^}]+background: var\(--demo\)/);
assert.match(homeTemplate, /data-zh="下载模型，<br>开始对话。"/);
assert.match(zhHTML, /下载模型，<br>开始对话。/);
assert.doesNotMatch(homeTemplate, /data-zh="[^"]*token[^"]*"/i, "Chinese display copy should not mix in the English token label");
assert.match(
  homeTemplate,
  /data-zh="你的 Mac。<br>你的模型。<br><em>一个原生<br>引擎。<\/em>"/,
  "Chinese hero display copy should use a balanced explicit line break",
);
assert.match(html, /href="\/assets\/css\/main\.css\?v=\d+"/);
assert.match(html, /type="module" src="\/assets\/js\/main\.js\?v=\d+"/);

assert.match(html, /id="lang-toggle" href="\/zh\/"[^>]+aria-label=/);
assert.match(zhHTML, /id="lang-toggle" href="\/"[^>]+aria-label=/);
assert.match(html, /id="theme-toggle"[^>]+aria-label=/);
assert.match(css, /prefers-reduced-motion/);
assert.doesNotMatch(js, /setLanguage|setTranslatedContent|data-lang-option|dataset\.zh/);
assert.match(js, /legacyLanguageDestination\(window\.location\.href\)/);

for (const localizedHTML of [html, zhHTML]) {
  assert.match(localizedHTML, /hreflang="en" href="https:\/\/macmlx\.app\/">/);
  assert.match(localizedHTML, /hreflang="zh-Hans" href="https:\/\/macmlx\.app\/zh\/">/);
  assert.match(localizedHTML, /hreflang="x-default" href="https:\/\/macmlx\.app\/">/);
}

console.log(`public site checks passed (${bilingualNodes.length} bilingual nodes)`);
