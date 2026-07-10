import assert from "node:assert/strict";
import test from "node:test";

import { outputFileForPath, validateCanonicalPath } from "../lib/routes.mjs";
import { homeRoutes, routes } from "../routes.mjs";

test("validateCanonicalPath accepts normalized directory routes", () => {
  assert.equal(validateCanonicalPath("/"), "/");
  assert.equal(validateCanonicalPath("/zh/"), "/zh/");
});

for (const invalidPath of ["zh/", "/zh", "//zh/", "/zh/?lang=zh", "/zh/#top", "/../zh/"]) {
  test(`validateCanonicalPath rejects ${invalidPath}`, () => {
    assert.throws(() => validateCanonicalPath(invalidPath), /Invalid canonical path/);
  });
}

test("outputFileForPath maps canonical routes to deterministic HTML files", () => {
  assert.equal(outputFileForPath("/"), "index.html");
  assert.equal(outputFileForPath("/zh/"), "zh/index.html");
});

test("home routes declare exact English and Simplified Chinese counterparts", () => {
  assert.deepEqual(homeRoutes, { en: "/", "zh-Hans": "/zh/" });
  assert.deepEqual(routes[0], { id: "home", kind: "home", paths: homeRoutes });
  assert.equal(routes.length, 13);
  assert.equal(routes.filter((route) => route.kind === "article").length, 12);
});
