import assert from "node:assert/strict";
import test from "node:test";

import worker, { redirectDestination } from "../cloudflare/www-redirect.mjs";

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
