import assert from "node:assert/strict";
import test from "node:test";

import { legacyLanguageDestination } from "../assets/js/legacy-language.mjs";

test("legacy Chinese query migrates to the static Chinese route", () => {
  assert.equal(
    legacyLanguageDestination("https://macmlx.app/?lang=zh#engine"),
    "/zh/#engine",
  );
});

test("legacy English query on Chinese route migrates to the English root", () => {
  assert.equal(
    legacyLanguageDestination("https://macmlx.app/zh/?lang=en&source=old#quickstart"),
    "/#quickstart",
  );
});

test("legacy migration removes queries while preserving hashes", () => {
  assert.equal(
    legacyLanguageDestination("https://macmlx.app/?campaign=old&lang=en#top"),
    "/#top",
  );
});

test("unrelated URLs are not redirected", () => {
  assert.equal(legacyLanguageDestination("https://macmlx.app/#engine"), null);
  assert.equal(legacyLanguageDestination("https://macmlx.app/?lang=fr#engine"), null);
});

test("supported legacy language queries do not redirect unrelated paths", () => {
  assert.equal(legacyLanguageDestination("https://macmlx.app/docs/?lang=zh#engine"), null);
  assert.equal(legacyLanguageDestination("https://macmlx.app/download?lang=en#top"), null);
});
