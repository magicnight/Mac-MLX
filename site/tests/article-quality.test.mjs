import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const css = await readFile(new URL("../assets/css/main.css", import.meta.url), "utf8");
const home = await readFile(new URL("../templates/home.html", import.meta.url), "utf8");

function mediaBlock(query) {
  const marker = `@media ${query}`;
  const start = css.indexOf(marker);
  assert.ok(start >= 0, `missing ${marker}`);
  const opening = css.indexOf("{", start);
  let depth = 1;
  for (let index = opening + 1; index < css.length; index += 1) {
    if (css[index] === "{") depth += 1;
    if (css[index] === "}") depth -= 1;
    if (depth === 0) return css.slice(opening + 1, index);
  }
  assert.fail(`unclosed ${marker}`);
}

function rgb(hex) {
  return [1, 3, 5].map((index) => Number.parseInt(hex.slice(index, index + 2), 16) / 255).map((value) => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
}

function contrast(left, right) {
  const lum = (value) => { const c = rgb(value); return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]; };
  const [a, b] = [lum(left), lum(right)].sort((x, y) => y - x);
  return (a + .05) / (b + .05);
}

test("mobile article tables preserve semantic headers inside horizontal scrollers", () => {
  assert.match(css, /\.table-wrap \{[^}]*overflow-x: auto;/);
  const mobile = mediaBlock("(max-width: 720px)");
  assert.doesNotMatch(mobile, /\.article-table[^}]*display:\s*block/);
  assert.doesNotMatch(mobile, /\.comparison-table[^}]*display:\s*block/);
  assert.doesNotMatch(mobile, /(?:\.article-table|\.comparison-table) thead[^}]*clip:/);
});

test("article small text and links meet 4.5 to 1 in both themes and surfaces", () => {
  const themes = [
    { faint: "#5f665e", blue: "#0057c7", paper: "#f3f1ea", deep: "#e8e4da" },
    { faint: "#a3aaa1", blue: "#75adff", paper: "#111311", deep: "#191c19" },
  ];
  for (const theme of themes) for (const foreground of [theme.faint, theme.blue]) for (const background of [theme.paper, theme.deep]) {
    assert.ok(contrast(foreground, background) >= 4.5, `${foreground} on ${background}`);
  }
  for (const value of ["#5f665e", "#0057c7", "#a3aaa1"]) assert.match(css, new RegExp(value, "i"));
});

test("compact primary navigation remains present and keyboard-scrollable below 1020px", () => {
  const tablet = mediaBlock("(max-width: 1020px)");
  assert.doesNotMatch(tablet, /\.nav-links\s*\{[^}]*display:\s*none/);
  assert.match(tablet, /\.nav-links\s*\{[^}]*overflow-x:\s*auto/);
  assert.match(tablet, /\.nav-links\s*\{[^}]*white-space:\s*nowrap/);
});

test("below-fold engine duplicates never request high fetch priority", () => {
  assert.equal(home.match(/fetchpriority="high"/g)?.length ?? 0, 0);
});
