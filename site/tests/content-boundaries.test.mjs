import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { project } from "../content/project.mjs";

const home = await readFile(new URL("../templates/home.html", import.meta.url), "utf8");

test("home copy preserves the audited VLM and Python boundaries", () => {
  assert.doesNotMatch(home, /16 (?:vision|种视觉)/i);
  assert.match(home, /14 detected VLM families|检测到的 14 个 VLM 家族/);
  assert.doesNotMatch(home, /<strong>0<\/strong><span[^>]*>Python runtime<\/span>/);
  assert.match(project.locales.en.twitterDescription, /default inference path/i);
  assert.match(project.locales["zh-Hans"].twitterDescription, /默认推理路径/);
});

test("home exposes the knowledge hub without removing section anchors", () => {
  assert.match(home, /href="{{architecturePath}}">{{learnLabel}}<\/a>/);
  for (const anchor of ["#why", "#features", "#engine", "#progress", "#quickstart"]) assert.match(home, new RegExp(`href="${anchor}"`));
});
