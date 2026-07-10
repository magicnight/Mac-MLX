import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { access, readFile, readdir } from "node:fs/promises";
import test, { before } from "node:test";
import { fileURLToPath } from "node:url";

import { project } from "../content/project.mjs";
import { assetPaths, contentHubCopiedAssetPaths } from "../content/assets.mjs";
import { validateProject } from "../lib/project-schema.mjs";
import { renderHomeTemplate, renderSocialImage } from "../lib/site-rendering.mjs";

const repositoryRoot = new URL("../../", import.meta.url);
const buildScript = new URL("scripts/build-public-site.mjs", repositoryRoot);
const englishURL = new URL("public/index.html", repositoryRoot);
const chineseURL = new URL("public/zh/index.html", repositoryRoot);
const templateURL = new URL("site/templates/home.html", repositoryRoot);

let english;
let chinese;
let template;
let mainCSS;
let browserJS;
let socialImageTemplate;

function cssBlock(source, selector) {
  const match = source.match(new RegExp(`${selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*\\{([^}]*)\\}`));
  assert.ok(match, `missing CSS block: ${selector}`);
  return match[1];
}

function cssVariables(block) {
  return Object.fromEntries([...block.matchAll(/--([\w-]+):\s*(#[\da-f]{6})\s*;/gi)].map((match) => [match[1], match[2]]));
}

function hexChannels(value) {
  return [1, 3, 5].map((index) => Number.parseInt(value.slice(index, index + 2), 16));
}

function relativeLuminance(channels) {
  const linear = channels.map((channel) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return (0.2126 * linear[0]) + (0.7152 * linear[1]) + (0.0722 * linear[2]);
}

function contrastRatio(foreground, background) {
  const foregroundLuminance = relativeLuminance(foreground);
  const backgroundLuminance = relativeLuminance(background);
  return (Math.max(foregroundLuminance, backgroundLuminance) + 0.05)
    / (Math.min(foregroundLuminance, backgroundLuminance) + 0.05);
}

async function relativeFiles(directory, prefix = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) files.push(...await relativeFiles(new URL(`${entry.name}/`, directory), relativePath));
    else if (entry.isFile()) files.push(relativePath);
  }
  return files.sort();
}

before(async () => {
  execFileSync(process.execPath, [fileURLToPath(buildScript)], { cwd: repositoryRoot, stdio: "pipe" });
  [english, chinese, template, mainCSS, browserJS, socialImageTemplate] = await Promise.all([
    readFile(englishURL, "utf8"),
    readFile(chineseURL, "utf8"),
    readFile(templateURL, "utf8"),
    readFile(new URL("site/assets/css/main.css", repositoryRoot), "utf8"),
    readFile(new URL("site/assets/js/main.js", repositoryRoot), "utf8"),
    readFile(new URL("site/templates/og-image.svg", repositoryRoot), "utf8"),
  ]);
});

test("project registry keeps supported facts and locale metadata together", () => {
  assert.equal(project.origin, "https://macmlx.app");
  assert.equal(project.repositoryURL, "https://github.com/magicnight/mac-mlx");
  assert.equal(project.downloadURL, "https://github.com/magicnight/mac-mlx/releases/latest");
  assert.equal(project.currentVersion, "0.5.3");
  assert.deepEqual(Object.keys(project.locales), ["en", "zh-Hans"]);
  for (const locale of Object.values(project.locales)) {
    assert.ok(locale.title);
    assert.ok(locale.description);
  }
});

test("project schema rejects missing display metadata and runtime copy fields", () => {
  const missingTitle = structuredClone(project);
  delete missingTitle.locales.en.title;
  assert.throws(() => validateProject(missingTitle), /Missing project\.locales\.en key: title/);

  const missingCopyLabel = structuredClone(project);
  delete missingCopyLabel.locales["zh-Hans"].copySuccess;
  assert.throws(() => validateProject(missingCopyLabel), /Missing project\.locales\.zh-Hans key: copySuccess/);
});

test("project schema rejects unknown fields and undefined locale values", () => {
  const unknownField = structuredClone(project);
  unknownField.locales.en.extra = "not supported";
  assert.throws(() => validateProject(unknownField), /Unknown project\.locales\.en key: extra/);

  const undefinedField = structuredClone(project);
  undefinedField.locales.en.copyFailure = undefined;
  assert.throws(() => validateProject(undefinedField), /project\.locales\.en\.copyFailure must be a non-empty string/);
});

test("project schema rejects hostile or untrusted URL registry values", () => {
  const hostileValues = [
    ["origin", " https://macmlx.app", /project\.origin must not contain surrounding whitespace/],
    ["origin", "javascript:alert(1)", /project\.origin must use HTTPS/],
    ["origin", "https://user:pass@macmlx.app", /project\.origin must not contain credentials/],
    ["origin", "https://macmlx.app/path", /project\.origin must be origin-only/],
    ["origin", "https://macmlx.app?x=1", /project\.origin must be origin-only/],
    ["origin", "https://macmlx.app#x", /project\.origin must be origin-only/],
    ["origin", 'https://macmlx.app/" onload="alert(1)', /project\.origin must be a valid URL/],
    ["repositoryURL", "https://github.com/attacker/mac-mlx", /project\.repositoryURL must equal the trusted repository URL/],
    ["repositoryURL", "https://user@github.com/magicnight/mac-mlx", /project\.repositoryURL must not contain credentials/],
    ["downloadURL", "https://github.com/magicnight/mac-mlx/issues", /project\.downloadURL must stay under the repository releases path/],
    ["downloadURL", "https://evil.example/magicnight/mac-mlx/releases/latest", /project\.downloadURL must stay under the repository releases path/],
    ["licenseURL", "https://evil.example/LICENSE-2.0", /project\.licenseURL must equal the trusted license URL/],
  ];
  for (const [field, value, expected] of hostileValues) {
    const candidate = structuredClone(project);
    candidate[field] = value;
    assert.throws(() => validateProject(candidate), expected, `${field}: ${value}`);
  }
});

test("source template preserves all 97 approved bilingual nodes", () => {
  assert.equal(template.match(/data-en=/g)?.length, 97);
  assert.equal(template.match(/data-zh=/g)?.length, 97);
});

test("reveal content stays visible until JavaScript establishes readiness", () => {
  assert.match(mainCSS, /\.reveal\s*\{[^}]*opacity:\s*1\s*;[^}]*transform:\s*none\s*;/);
  assert.doesNotMatch(mainCSS, /(?:^|\n)\.reveal\s*\{[^}]*opacity:\s*0\s*;/);
  assert.match(mainCSS, /html\.reveal-ready \.reveal\s*\{[^}]*opacity:\s*0\s*;/);
  assert.match(browserJS, /nodes\.forEach\(\(node\) => observer\.observe\(node\)\);\s*document\.documentElement\.classList\.add\("reveal-ready"\)/);
});

test("engine status text reaches 4.5:1 contrast in light and dark themes", () => {
  const statusRule = cssBlock(mainCSS, ".engine-status");
  const textVariable = statusRule.match(/color:\s*var\(--([\w-]+)\)/)?.[1];
  const backgroundMix = statusRule.match(/background:\s*color-mix\(in srgb, var\(--([\w-]+)\)\s+(\d+)%, transparent\)/);
  const surfaceVariable = cssBlock(mainCSS, ".engine-section").match(/background:\s*var\(--([\w-]+)\)/)?.[1];
  assert.ok(textVariable && backgroundMix && surfaceVariable, "engine status colors should use theme variables");

  for (const [theme, variables] of [
    ["light", cssVariables(cssBlock(mainCSS, ":root"))],
    ["dark", cssVariables(cssBlock(mainCSS, 'html[data-theme="dark"]'))],
  ]) {
    const text = hexChannels(variables[textVariable]);
    const tint = hexChannels(variables[backgroundMix[1]]);
    const surface = hexChannels(variables[surfaceVariable]);
    const alpha = Number(backgroundMix[2]) / 100;
    const background = surface.map((channel, index) => (alpha * tint[index]) + ((1 - alpha) * channel));
    const ratio = contrastRatio(text, background);
    assert.ok(ratio >= 4.5, `${theme} engine status contrast is ${ratio.toFixed(2)}:1`);
  }
});

test("build emits independent locale documents without runtime bilingual attributes", () => {
  assert.match(english, /<html lang="en">/);
  assert.match(chinese, /<html lang="zh-CN">/);
  assert.match(english, /Swift-native in-process engine/);
  assert.match(chinese, /Swift 原生进程内引擎/);
  for (const html of [english, chinese]) {
    assert.doesNotMatch(html, /\bdata-(?:en|zh|lang-option)=/);
    assert.doesNotMatch(html, /<body[^>]+data-lang=/);
    assert.doesNotMatch(html, /{{|}}/);
    assert.doesNotMatch(html, /\bundefined\b/);
  }
});

test("locale pages use self canonicals and reciprocal alternates", () => {
  assert.match(english, /<link rel="canonical" href="https:\/\/macmlx\.app\/">/);
  assert.match(chinese, /<link rel="canonical" href="https:\/\/macmlx\.app\/zh\/">/);

  for (const html of [english, chinese]) {
    assert.match(html, /hreflang="en" href="https:\/\/macmlx\.app\/">/);
    assert.match(html, /hreflang="zh-Hans" href="https:\/\/macmlx\.app\/zh\/">/);
    assert.match(html, /hreflang="x-default" href="https:\/\/macmlx\.app\/">/);
  }
  assert.match(english, /id="lang-toggle" href="\/zh\/"/);
  assert.match(chinese, /id="lang-toggle" href="\/"/);
});

test("metadata is locale-specific and JSON-LD remains truthful", () => {
  assert.match(english, new RegExp(`<title>${project.locales.en.title}</title>`));
  assert.match(chinese, new RegExp(`<title>${project.locales["zh-Hans"].title}</title>`));
  assert.notEqual(project.locales.en.description, project.locales["zh-Hans"].description);

  for (const [html, locale] of [[english, "en"], [chinese, "zh-Hans"]]) {
    const jsonSource = html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)?.[1];
    assert.ok(jsonSource, `${locale} page should include JSON-LD`);
    const json = JSON.parse(jsonSource);
    const software = json["@graph"].find((node) => node["@type"] === "SoftwareApplication");
    assert.equal(software.softwareVersion, project.currentVersion);
    assert.equal(software.inLanguage, locale);
    assert.equal(software.codeRepository, project.repositoryURL);
    assert.equal(software.downloadUrl, project.downloadURL);
  }
});

test("home and social rendering derive visible facts from a supplied registry", () => {
  const modifiedProject = structuredClone(project);
  modifiedProject.currentVersion = "9.8.7";
  modifiedProject.locales.en.title = "macMLX registry-driven title";

  const renderedHome = renderHomeTemplate({
    template,
    project: modifiedProject,
    routes: { en: "/", "zh-Hans": "/zh/" },
    locale: "en",
  });
  const renderedSocialImage = renderSocialImage({ template: socialImageTemplate, project: modifiedProject });

  assert.match(renderedHome, /<b>v9\.8\.7<\/b>/);
  assert.match(renderedHome, /available in v9\.8\.7/);
  assert.match(renderedHome, /<title>macMLX registry-driven title<\/title>/);
  assert.match(renderedHome, /href="https:\/\/github\.com\/magicnight\/mac-mlx\/releases\/latest"/);
  assert.match(renderedHome, /href="https:\/\/github\.com\/magicnight\/mac-mlx"/);
  assert.match(renderedHome, /"softwareVersion": "9\.8\.7"/);
  assert.match(renderedSocialImage, />v9\.8\.7 · available<\/text>/);
  assert.doesNotMatch(template, /0\.5\.3|https:\/\/github\.com\/magicnight\/mac-mlx/);
  assert.doesNotMatch(socialImageTemplate, /0\.5\.3/);
});

test("all local page assets are root-absolute and exist", async () => {
  for (const html of [english, chinese]) {
    assert.doesNotMatch(html, /(?:src|href)="assets\//);
    const assetPaths = [...html.matchAll(/(?:src|href)="(\/assets\/[^"?#]+)/g)].map((match) => match[1]);
    assert.ok(assetPaths.length >= 13, "expected styles, script, images, and social image references");
    for (const assetPath of new Set(assetPaths)) {
      await access(new URL(`public${assetPath}`, repositoryRoot));
    }
  }
});

test("generated public assets exactly match the complete explicit manifest", async () => {
  const published = await relativeFiles(new URL("public/assets/", repositoryRoot));
  assert.deepEqual(published, [...assetPaths].sort());
  assert.deepEqual(published.filter((path) => path.startsWith("images/generated/")), [...contentHubCopiedAssetPaths].sort());
});

test("repeated builds produce byte-identical locale documents", async () => {
  execFileSync(process.execPath, [fileURLToPath(buildScript)], { cwd: repositoryRoot, stdio: "pipe" });
  assert.equal(await readFile(englishURL, "utf8"), english);
  assert.equal(await readFile(chineseURL, "utf8"), chinese);
});
