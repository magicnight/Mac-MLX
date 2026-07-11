import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const workflow = await readFile(new URL("../../.github/workflows/ci.yml", import.meta.url), "utf8");

test("CI website job verifies every generated-site surface", () => {
  assert.match(workflow, /^  website:\n/m);
  assert.match(workflow, /website:[\s\S]*?runs-on: macos-26/);
  assert.match(workflow, /node scripts\/build-public-site\.mjs/);
  assert.match(workflow, /node scripts\/validate-social-cards\.mjs/);
  assert.match(workflow, /node --test site\/tests\/\*\.test\.mjs/);
  assert.match(workflow, /node scripts\/crawl-public-site\.mjs/);
  assert.match(workflow, /node scripts\/test-public-site\.mjs/);
  assert.match(workflow, /node --check scripts\/build-public-site\.mjs/);
  assert.match(workflow, /node --check scripts\/crawl-public-site\.mjs/);
  assert.match(workflow, /node --check scripts\/render-brand-icons\.mjs/);
  assert.match(workflow, /node --check scripts\/render-social-cards\.mjs/);
  assert.match(workflow, /node --check scripts\/validate-social-cards\.mjs/);
  assert.match(workflow, /node --check scripts\/verify-cloudflare-deploy\.mjs/);
  assert.match(workflow, /node --check site\/lib\/install-manifest\.mjs/);
  assert.match(workflow, /node --check site\/lib\/png-source-digest\.mjs/);
  assert.match(workflow, /node --check public\/assets\/js\/main\.js/);
  assert.match(workflow, /command -v xmllint/);
  assert.match(workflow, /xmllint --noout site\/assets\/brand\/macmlx-mark\.svg site\/assets\/brand\/favicon\.svg public\/assets\/og-image\.svg public\/sitemap\.xml/);
});
