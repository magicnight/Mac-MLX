import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { verifyWWWRedirect } from "../../scripts/verify-www-redirect.mjs";
import worker, { redirectDestination } from "../cloudflare/www-redirect.mjs";

const repositoryRoot = new URL("../../", import.meta.url);

test("www requests permanently redirect to the equivalent apex URL", async () => {
  const response = await worker.fetch(
    new Request("https://www.macmlx.app/zh/?q=mlx%20swift&lang=zh"),
  );

  assert.equal(response.status, 308);
  assert.equal(
    response.headers.get("location"),
    "https://macmlx.app/zh/?q=mlx%20swift&lang=zh",
  );
  assert.equal(await response.text(), "");
});

test("the incoming host can never change the apex destination", () => {
  assert.equal(
    redirectDestination(
      "https://attacker.example/models/?next=https%3A%2F%2Fevil.example",
    ),
    "https://macmlx.app/models/?next=https%3A%2F%2Fevil.example",
  );
});

test("root requests redirect to the apex root", () => {
  assert.equal(redirectDestination("https://www.macmlx.app/"), "https://macmlx.app/");
});

test("encoded Unicode paths stay encoded", () => {
  assert.equal(
    redirectDestination("https://www.macmlx.app/%E6%A8%A1%E5%9E%8B/%E9%80%89%E6%8B%A9/"),
    "https://macmlx.app/%E6%A8%A1%E5%9E%8B/%E9%80%89%E6%8B%A9/",
  );
});

test("POST requests receive a method-preserving permanent redirect", async () => {
  const response = await worker.fetch(
    new Request("https://www.macmlx.app/v1/models?source=www", {
      method: "POST",
      body: "payload",
    }),
  );

  assert.equal(response.status, 308);
  assert.equal(
    response.headers.get("location"),
    "https://macmlx.app/v1/models?source=www",
  );
});

test("the Wrangler configuration owns only the exact www custom domain", async () => {
  const config = JSON.parse(
    await readFile(new URL("wrangler.www.jsonc", repositoryRoot), "utf8"),
  );

  assert.equal(config.name, "macmlx-www-redirect");
  assert.equal(config.main, "site/cloudflare/www-redirect.mjs");
  assert.equal(config.compatibility_date, "2026-07-11");
  assert.equal(config.workers_dev, false);
  assert.equal(config.preview_urls, false);
  assert.deepEqual(config.routes, [
    { pattern: "www.macmlx.app", custom_domain: true },
  ]);
  assert.equal(config.assets, undefined);
});

const expectedRedirects = new Map([
  ["https://www.macmlx.app/", "https://macmlx.app/"],
  ["https://www.macmlx.app/zh/", "https://macmlx.app/zh/"],
  [
    "https://www.macmlx.app/models/choosing-a-model/?q=mlx%20swift",
    "https://macmlx.app/models/choosing-a-model/?q=mlx%20swift",
  ],
]);

function responseFor(location, status = 308) {
  return new Response(null, { status, headers: { Location: location } });
}

test("the verifier checks all production cases without following redirects", async () => {
  const requests = [];
  const fetchImpl = async (url, init) => {
    requests.push([url, init]);
    return responseFor(expectedRedirects.get(url));
  };

  const result = await verifyWWWRedirect({ fetchImpl });

  assert.equal(result.checked, expectedRedirects.size);
  assert.deepEqual(
    requests.map(([url]) => url),
    [...expectedRedirects.keys()],
  );
  for (const [, init] of requests) {
    assert.equal(init.redirect, "manual");
    assert.ok(init.signal instanceof AbortSignal);
  }
});

for (const [name, failingResponse, expectedMessage] of [
  ["status 200", responseFor("https://macmlx.app/", 200), "expected status 308, received 200"],
  ["a cross-origin destination", responseFor("https://evil.example/"), "expected Location https://macmlx.app/"],
  ["a changed path", responseFor("https://macmlx.app/changed/"), "expected Location https://macmlx.app/"],
  ["a dropped query", responseFor("https://macmlx.app/models/choosing-a-model/"), "?q=mlx%20swift"],
]) {
  test(`the verifier rejects ${name}`, async () => {
    const failingURL = name === "a dropped query"
      ? "https://www.macmlx.app/models/choosing-a-model/?q=mlx%20swift"
      : "https://www.macmlx.app/";
    const fetchImpl = async (url) => {
      if (url === failingURL) return failingResponse;
      return responseFor(expectedRedirects.get(url));
    };

    await assert.rejects(
      verifyWWWRedirect({ fetchImpl }),
      (error) => {
        assert.ok(error instanceof AggregateError);
        assert.match(error.message, /www redirect verification failed/i);
        assert.match(error.errors.map(String).join("\n"), new RegExp(expectedMessage.replace(/[.?+^$[\]\\(){}|-]/g, "\\$&")));
        return true;
      },
    );
  });
}
