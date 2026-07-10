import assert from "node:assert/strict";
import test from "node:test";

import {
  escapeHTML,
  localizeBilingualElements,
  renderTokens,
} from "../lib/localize.mjs";

test("escapeHTML escapes text before inserting it into HTML", () => {
  assert.equal(
    escapeHTML(`<script data-name="x">Tom & Jerry's</script>`),
    "&lt;script data-name=&quot;x&quot;&gt;Tom &amp; Jerry&#39;s&lt;/script&gt;",
  );
});

test("renderTokens escapes every supplied token", () => {
  assert.equal(
    renderTokens("<title>{{title}}</title>", { title: "A & <B>" }),
    "<title>A &amp; &lt;B&gt;</title>",
  );
});

test("renderTokens rejects missing tokens", () => {
  assert.throws(
    () => renderTokens("{{title}} — {{description}}", { title: "macMLX" }),
    /Missing template token: description/,
  );
});

test("renderTokens rejects unused values", () => {
  assert.throws(
    () => renderTokens("{{title}}", { title: "macMLX", description: "unused" }),
    /Unused template token: description/,
  );
});

test("localizeBilingualElements selects source markup and removes runtime scaffolding", () => {
  const source = '<h1 class="hero" data-en="One <em>engine</em>" data-zh="一个<em>引擎</em>">One <em>engine</em></h1>';

  assert.equal(
    localizeBilingualElements(source, "zh-Hans"),
    '<h1 class="hero">一个<em>引擎</em></h1>',
  );
});

test("localizeBilingualElements pairs nested same-name elements by depth", () => {
  const source = '<div class="outer" data-en="Outer" data-zh="外层"><div data-en="Inner" data-zh="内层">Inner</div></div>';

  assert.equal(
    localizeBilingualElements(source, "zh-Hans"),
    '<div class="outer">外层</div>',
  );
});

test("localizeBilingualElements rejects incomplete locale pairs", () => {
  assert.throws(
    () => localizeBilingualElements('<p data-en="English">English</p>', "en"),
    /Bilingual element is missing data-zh/,
  );
});

test("localizeBilingualElements rejects empty or whitespace-only locale values", () => {
  for (const source of [
    '<p data-en="" data-zh="中文">English</p>',
    '<p data-en="English" data-zh="   ">English</p>',
  ]) {
    assert.throws(
      () => localizeBilingualElements(source, "en"),
      /Bilingual element has an empty data-(?:en|zh) value/,
    );
  }
});

test("localizeBilingualElements rejects unsupported locales", () => {
  assert.throws(
    () => localizeBilingualElements('<p data-en="English" data-zh="中文">English</p>', "fr"),
    /Unsupported locale: fr/,
  );
});
