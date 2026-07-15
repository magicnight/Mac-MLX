import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { digestPreparedSite } from "../lib/determinism.mjs";
import { prepareSite } from "../../scripts/build-public-site.mjs";

const root = new URL("../../", import.meta.url);
const trackedManifestSource = await readFile(new URL("site/assets/brand/site.webmanifest", root), "utf8");
const trackedManifest = JSON.parse(trackedManifestSource);
const trackedBrandPNGs = Object.freeze({
  "/assets/brand/apple-touch-icon.png": await readFile(new URL("site/assets/brand/apple-touch-icon.png", root)),
  "/assets/brand/icon-192.png": await readFile(new URL("site/assets/brand/icon-192.png", root)),
  "/assets/brand/icon-512.png": await readFile(new URL("site/assets/brand/icon-512.png", root)),
});

async function deploymentConfig() {
  return JSON.parse(await readFile(new URL("wrangler.jsonc", root), "utf8"));
}

async function policySource(name) {
  return readFile(new URL(`site/cloudflare/${name}`, root), "utf8");
}

function parseHeaderRules(source) {
  const rules = [];
  let current;
  for (const [index, line] of source.split("\n").entries()) {
    assert.ok(line.length <= 2_000, `_headers line ${index + 1} exceeds 2,000 characters`);
    if (!line.trim() || line.startsWith("#")) continue;
    if (!line.startsWith(" ")) {
      assert.match(line, /^(?:\/|https:\/\/)/, `invalid _headers pattern on line ${index + 1}`);
      current = { pattern: line, headers: new Map() };
      rules.push(current);
      continue;
    }
    assert.ok(current, `header appears before a pattern on line ${index + 1}`);
    assert.match(line, /^  [!A-Za-z0-9-]+(?:: .*)?$/, `invalid _headers syntax on line ${index + 1}`);
    const match = line.match(/^  ([A-Za-z0-9-]+): (.+)$/);
    assert.ok(match, `header must have a value on line ${index + 1}`);
    assert.equal(current.headers.has(match[1].toLowerCase()), false, `${match[1]} is duplicated in ${current.pattern}`);
    current.headers.set(match[1].toLowerCase(), match[2]);
  }
  assert.ok(rules.length <= 100, `_headers has ${rules.length} rules; Cloudflare allows 100`);
  return rules;
}

function ruleMap(rules) {
  return new Map(rules.map((rule) => [rule.pattern, rule.headers]));
}

const assets = Object.freeze({
  directory: "./public",
  html_handling: "force-trailing-slash",
  not_found_handling: "404-page",
});

test("production is a scriptless static-assets deployment on the apex custom domain", async () => {
  const config = await deploymentConfig();
  assert.equal(config.name, "macmlx-site");
  assert.equal(config.compatibility_date, "2026-07-10");
  assert.equal(Object.hasOwn(config, "main"), false);
  assert.deepEqual(config.assets, assets);
  assert.equal(config.workers_dev, false);
  assert.equal(config.preview_urls, false);
  assert.deepEqual(config.routes, [{ pattern: "macmlx.app", custom_domain: true }]);
  for (const forbidden of ["r2_buckets", "kv_namespaces", "d1_databases", "services"]) {
    assert.equal(Object.hasOwn(config, forbidden), false, `${forbidden} must remain absent`);
  }
});

test("staging stays isolated on workers.dev with the complete static routing policy", async () => {
  const config = await deploymentConfig();
  assert.deepEqual(config.env?.staging, {
    name: "macmlx-site-staging",
    workers_dev: true,
    preview_urls: true,
    routes: [],
    assets,
  });
});

test("header policy stays within Cloudflare syntax and rule limits", async () => {
  const source = await policySource("_headers");
  const rules = parseHeaderRules(source);
  assert.equal(new Set(rules.map((rule) => rule.pattern)).size, rules.length, "header patterns must be unique");
});

test("global headers harden every response without an untested CSP", async () => {
  const source = await policySource("_headers");
  const headers = ruleMap(parseHeaderRules(source)).get("/*");
  assert.deepEqual(Object.fromEntries(headers), {
    "x-content-type-options": "nosniff",
    "referrer-policy": "strict-origin-when-cross-origin",
    "x-frame-options": "DENY",
    "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
  });
  assert.doesNotMatch(source, /content-security-policy/i);
});

test("documents revalidate while stable CSS, JavaScript, and images use bounded caching", async () => {
  const source = await policySource("_headers");
  const rules = ruleMap(parseHeaderRules(source));
  for (const pattern of ["/", "/*/", "/*.html", "/*.md", "/*.txt", "/*.xml", "/*.json"]) {
    assert.equal(rules.get(pattern)?.get("cache-control"), "public, max-age=0, must-revalidate", pattern);
  }
  for (const pattern of ["/*.css", "/*.js", "/*.mjs", "/*.webmanifest"]) {
    assert.equal(rules.get(pattern)?.get("cache-control"), "public, max-age=3600, stale-while-revalidate=86400", pattern);
  }
  for (const pattern of ["/*.webp", "/*.png", "/*.svg"]) {
    assert.equal(rules.get(pattern)?.get("cache-control"), "public, max-age=604800, stale-while-revalidate=86400", pattern);
  }
  assert.doesNotMatch(source, /immutable/i);
});

test("all workers.dev staging and preview hosts are explicitly noindex", async () => {
  const rules = ruleMap(parseHeaderRules(await policySource("_headers")));
  assert.equal(
    rules.get("https://:version.:subdomain.workers.dev/*")?.get("x-robots-tag"),
    "noindex, nofollow",
  );
});

test("redirect policy contains only the two canonical index migrations", async () => {
  const redirects = (await policySource("_redirects")).trim().split(/\n+/);
  assert.deepEqual(redirects, [
    "/index.html / 301",
    "/zh/index.html /zh/ 301",
  ]);
  assert.ok(redirects.every((line) => line.length <= 1_000));
  assert.ok(redirects.length <= 2_100);
  assert.doesNotMatch(redirects.join("\n"), /\?|:|\*/);
});

test("build preparation carries deployment policies into deterministic output", async () => {
  const prepared = await prepareSite({ today: "2026-07-15" });
  assert.deepEqual([...prepared.deploymentFiles.keys()], ["_headers", "_redirects"]);
  const changed = {
    ...prepared,
    deploymentFiles: new Map(prepared.deploymentFiles).set("_headers", `${prepared.deploymentFiles.get("_headers")}\n# changed\n`),
  };
  assert.notEqual(digestPreparedSite(prepared), digestPreparedSite(changed));
});

test("generated deployment policies are byte-identical to their tracked sources", async () => {
  for (const name of ["_headers", "_redirects"]) {
    assert.equal(
      await readFile(new URL(`public/${name}`, root), "utf8"),
      await policySource(name),
      name,
    );
  }
  const sitemap = await readFile(new URL("public/sitemap.xml", root), "utf8");
  assert.doesNotMatch(sitemap, /_headers|_redirects/);
});

function pngHeader(width = 1200, height = 630) {
  const bytes = Buffer.alloc(24);
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(bytes);
  bytes.writeUInt32BE(13, 8);
  bytes.write("IHDR", 12, "ascii");
  bytes.writeUInt32BE(width, 16);
  bytes.writeUInt32BE(height, 20);
  return bytes;
}

function mockDeploymentFetch({ workersDevNoindex = true, omitFrameHeader = false, brokenAsset = false, redirectLocations = {}, mimeOverrides = {}, cacheOverrides = {}, manifestOverride, manifestIconBodies = {}, missingIcon } = {}) {
  const calls = [];
  const security = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "strict-origin-when-cross-origin",
    ...(omitFrameHeader ? {} : { "x-frame-options": "DENY" }),
    "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
  };
  const documentHeaders = {
    ...security,
    "content-type": "text/html; charset=UTF-8",
    "cache-control": "public, max-age=0, must-revalidate",
    ...(workersDevNoindex ? { "x-robots-tag": "noindex, nofollow" } : {}),
  };
  const canonicalPaths = new Set(["/", "/zh/", "/architecture/", "/zh/architecture/"]);
  const responseHeaders = (path, headers) => ({
    ...headers,
    ...(mimeOverrides[path] ? { "content-type": mimeOverrides[path] } : {}),
    ...(cacheOverrides[path] ? { "cache-control": cacheOverrides[path] } : {}),
  });

  const fetchImpl = async (input, init = {}) => {
    const url = new URL(input);
    const method = init.method ?? "GET";
    calls.push({ url: url.href, method, redirect: init.redirect, signal: init.signal });
    if (method === "HEAD" && url.pathname === "/index.html") return new Response(null, { status: 301, headers: { location: redirectLocations.index ?? "/" } });
    if (method === "HEAD" && url.pathname === "/zh/index.html") return new Response(null, { status: 301, headers: { location: redirectLocations.zh ?? "/zh/" } });
    if (url.pathname === "/__macmlx-deployment-verification__/") {
      return new Response('<meta name="robots" content="noindex,follow">', { status: 404, headers: responseHeaders(url.pathname, documentHeaders) });
    }
    if (canonicalPaths.has(url.pathname)) {
      const assetsHTML = url.pathname === "/"
        ? '<link rel="stylesheet" href="/assets/css/main.css?v=3"><link rel="manifest" href="/assets/brand/site.webmanifest"><script src="/assets/js/main.js?v=3"></script><img src="/assets/images/engine/adaptive-runtime.webp" alt=""><img src="/assets/og-image.svg" alt="">'
        : "";
      return new Response(`<link rel="canonical" href="https://macmlx.app${url.pathname}">${assetsHTML}`, { status: 200, headers: responseHeaders(url.pathname, documentHeaders) });
    }
    if (url.pathname === "/robots.txt") {
      return new Response("User-agent: *\nAllow: /\nSitemap: https://macmlx.app/sitemap.xml\n", { status: 200, headers: responseHeaders(url.pathname, { ...documentHeaders, "content-type": "text/plain" }) });
    }
    if (url.pathname === "/sitemap.xml") {
      return new Response("<urlset><url><loc>https://macmlx.app/</loc></url><url><loc>https://macmlx.app/zh/</loc></url></urlset>", { status: 200, headers: responseHeaders(url.pathname, { ...documentHeaders, "content-type": "application/xml" }) });
    }
    if (url.pathname === "/llms.txt" || url.pathname === "/zh/llms.txt") {
      return new Response("# macMLX\n", { status: 200, headers: responseHeaders(url.pathname, { ...documentHeaders, "content-type": "text/plain" }) });
    }
    if (url.pathname === "/assets/social/og-en.png") {
      return new Response(pngHeader(), { status: 200, headers: responseHeaders(url.pathname, { ...security, "content-type": "image/png", "cache-control": "public, max-age=604800, stale-while-revalidate=86400" }) });
    }
    if (url.pathname === "/assets/brand/site.webmanifest" && method === "GET") {
      const body = manifestOverride === undefined ? trackedManifestSource : manifestOverride;
      return new Response(body, { status: 200, headers: responseHeaders(url.pathname, { ...security, "content-type": "application/manifest+json", "cache-control": "public, max-age=3600, stale-while-revalidate=86400" }) });
    }
    if (method === "GET" && ["/assets/brand/icon-192.png", "/assets/brand/icon-512.png"].includes(url.pathname)) {
      if (url.pathname === missingIcon) return new Response("missing", { status: 404, headers: responseHeaders(url.pathname, { "content-type": "text/plain", "cache-control": "public, max-age=0, must-revalidate" }) });
      const body = manifestIconBodies[url.pathname] ?? trackedBrandPNGs[url.pathname];
      return new Response(body, { status: 200, headers: responseHeaders(url.pathname, { ...security, "content-type": "image/png", "cache-control": "public, max-age=604800, stale-while-revalidate=86400" }) });
    }
    if (method === "HEAD" && url.pathname.startsWith("/assets/")) {
      if (brokenAsset && url.pathname.endsWith("main.js")) return new Response(null, { status: 404 });
      const image = /\.(?:webp|png|svg)$/.test(url.pathname);
      const contentType = url.pathname.endsWith(".css") ? "text/css"
        : url.pathname.endsWith(".js") ? "text/javascript"
          : url.pathname.endsWith(".webmanifest") ? "application/manifest+json"
          : url.pathname.endsWith(".svg") ? "image/svg+xml"
            : "image/webp";
      return new Response(null, { status: 200, headers: responseHeaders(url.pathname, { ...security, "content-type": contentType, "cache-control": image ? "public, max-age=604800, stale-while-revalidate=86400" : "public, max-age=3600, stale-while-revalidate=86400" }) });
    }
    return new Response("not mocked", { status: 500 });
  };
  return { fetchImpl, calls };
}

test("remote verifier checks production routes, redirects, discovery output, PNG dimensions, and primary assets", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  const { fetchImpl, calls } = mockDeploymentFetch({ workersDevNoindex: false });
  const result = await verifyCloudflareDeployment("https://macmlx.app/", { fetchImpl });
  assert.ok(result.checks >= 20);
  assert.ok(calls.some((call) => call.method === "GET"));
  assert.ok(calls.some((call) => call.method === "HEAD"));
  assert.ok(calls.every((call) => call.signal instanceof AbortSignal));
  assert.ok(calls.some((call) => call.url.includes("/assets/js/main.js?v=3")));
  assert.ok(calls.some((call) => call.url.endsWith("/index.html") && call.redirect === "manual"));
  assert.ok(calls.some((call) => call.method === "GET" && call.url.endsWith("/assets/brand/site.webmanifest")));
  assert.ok(calls.some((call) => call.method === "GET" && call.url.endsWith("/assets/brand/icon-192.png")));
  assert.ok(calls.some((call) => call.method === "GET" && call.url.endsWith("/assets/brand/icon-512.png")));
});

test("remote verifier strictly validates the install manifest and its same-origin PNG icons", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  const manifestCases = [
    [{ manifestOverride: "not json" }, /site\.webmanifest.*JSON/i],
    ...[null, false, 0, ""].map((value) => [{ manifestOverride: JSON.stringify(value) }, /site\.webmanifest.*exact tracked install contract/i]),
    [{ manifestOverride: JSON.stringify({ ...trackedManifest, extra: true }) }, /exact tracked install contract/i],
    [{ manifestOverride: JSON.stringify({ ...trackedManifest, theme_color: "#ffffff" }) }, /exact tracked install contract/i],
    [{ manifestOverride: JSON.stringify({ ...trackedManifest, icons: [...trackedManifest.icons, trackedManifest.icons[0]] }) }, /exact tracked install contract/i],
    [{ manifestOverride: JSON.stringify({ ...trackedManifest, icons: [{ ...trackedManifest.icons[0], src: "https://macmlx.app/assets/brand/icon-192.png" }, trackedManifest.icons[1]] }) }, /exact tracked install contract/i],
    [{ manifestOverride: JSON.stringify({ ...trackedManifest, icons: [trackedManifest.icons[0], { ...trackedManifest.icons[1], src: "https://evil.example/icon-512.png" }] }) }, /exact tracked install contract/i],
    [{ manifestOverride: JSON.stringify({ ...trackedManifest, icons: [{ ...trackedManifest.icons[0], src: "/assets/brand/../brand/icon-192.png" }, trackedManifest.icons[1]] }) }, /exact tracked install contract/i],
    [{ manifestOverride: JSON.stringify({ ...trackedManifest, icons: [{ ...trackedManifest.icons[0], extra: true }, trackedManifest.icons[1]] }) }, /exact tracked install contract/i],
  ];
  for (const [options, expected] of manifestCases) {
    const { fetchImpl } = mockDeploymentFetch({ workersDevNoindex: false, ...options });
    await assert.rejects(verifyCloudflareDeployment("https://macmlx.app/", { fetchImpl }), expected);
  }

  const iconCases = [
    [{ manifestIconBodies: { "/assets/brand/icon-192.png": trackedBrandPNGs["/assets/brand/apple-touch-icon.png"] } }, /icon-192\.png.*must be 192x192/i],
    [{ mimeOverrides: { "/assets/brand/icon-512.png": "image/webp" } }, /icon-512\.png.*image\/png/i],
    [{ cacheOverrides: { "/assets/brand/icon-192.png": "public, max-age=0, must-revalidate" } }, /icon-192\.png.*Cache-Control.*max-age=604800/i],
    [{ missingIcon: "/assets/brand/icon-512.png" }, /icon-512\.png.*expected 200.*404/i],
  ];
  for (const [options, expected] of iconCases) {
    const { fetchImpl } = mockDeploymentFetch({ workersDevNoindex: false, ...options });
    await assert.rejects(verifyCloudflareDeployment("https://macmlx.app/", { fetchImpl }), expected);
  }
});

test("remote verifier rejects truncated, corrupt, incomplete, and trailing brand PNG bodies", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  const valid = trackedBrandPNGs["/assets/brand/icon-192.png"];
  const chunks = [];
  for (let offset = 8; offset < valid.length;) {
    const length = valid.readUInt32BE(offset);
    const end = offset + 12 + length;
    chunks.push({ type: valid.toString("ascii", offset + 4, offset + 8), bytes: valid.subarray(offset, end) });
    offset = end;
  }
  const without = (type) => Buffer.concat([valid.subarray(0, 8), ...chunks.filter((chunk) => chunk.type !== type).map((chunk) => chunk.bytes)]);
  const badCRC = Buffer.from(valid);
  badCRC[20] ^= 1;
  const cases = [
    [valid.subarray(0, 24), /truncated PNG chunk/i],
    [badCRC, /CRC mismatch/i],
    [without("IDAT"), /invalid terminal IEND|missing required PNG chunks/i],
    [without("IEND"), /missing required PNG chunks/i],
    [without("tEXt"), /exactly one source digest/i],
    [Buffer.concat([valid, Buffer.from("trailing")]), /terminal IEND|trailing/i],
  ];
  for (const [body, expected] of cases) {
    const { fetchImpl } = mockDeploymentFetch({ workersDevNoindex: false, manifestIconBodies: { "/assets/brand/icon-192.png": body } });
    await assert.rejects(verifyCloudflareDeployment("https://macmlx.app/", { fetchImpl }), expected);
  }
});

test("remote verifier accepts only the exact production origin or macmlx staging host family", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  for (const baseURL of [
    "http://macmlx.app/",
    "https://user@macmlx.app/",
    "https://macmlx.app",
    "https://macmlx.app:443/",
    "https://macmlx.app/path/",
    "https://macmlx.app/?preview=1",
    "https://macmlx.app/#preview",
    "https://localhost/",
    "https://127.0.0.1/",
    "https://10.0.0.1/",
    "https://172.16.0.1/",
    "https://192.168.0.1/",
    "https://[::1]/",
    "https://example.com/",
    "https://other.example.workers.dev/",
    "https://macmlx-site-staging.example.workers.dev.evil.com/",
    "https://macmlx-site-staging.evil.example.workers.dev/",
    "https://notmacmlx-site-staging.example.workers.dev/",
  ]) {
    let fetched = false;
    await assert.rejects(
      verifyCloudflareDeployment(baseURL, { fetchImpl: async () => { fetched = true; throw new Error("must not fetch"); } }),
      /deployment base URL|deployment target/i,
      baseURL,
    );
    assert.equal(fetched, false, baseURL);
  }

  const preview = mockDeploymentFetch();
  await assert.doesNotReject(
    verifyCloudflareDeployment("https://a1b2c3-macmlx-site-staging.example.workers.dev/", { fetchImpl: preview.fetchImpl }),
  );
});

test("remote verifier enforces workers.dev noindex on staging and preview hosts", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  const passing = mockDeploymentFetch();
  await assert.doesNotReject(verifyCloudflareDeployment("https://macmlx-site-staging.example.workers.dev/", { fetchImpl: passing.fetchImpl }));

  const failing = mockDeploymentFetch({ workersDevNoindex: false });
  await assert.rejects(
    verifyCloudflareDeployment("https://macmlx-site-staging.example.workers.dev/", { fetchImpl: failing.fetchImpl }),
    /X-Robots-Tag.*noindex.*nofollow/i,
  );
});

test("remote verifier rejects redirects that escape the verified origin or add query and fragment state", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  for (const redirectLocations of [
    { index: "https://macmlx.app.evil.example/" },
    { index: "https://evil.example/" },
    { index: "/?redirected=1" },
    { zh: "/zh/#redirected" },
  ]) {
    const { fetchImpl } = mockDeploymentFetch({ workersDevNoindex: false, redirectLocations });
    await assert.rejects(
      verifyCloudflareDeployment("https://macmlx.app/", { fetchImpl }),
      /redirect.*expected exact same-origin/i,
    );
  }
});

test("remote verifier reports all actionable header and broken-asset failures", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  const { fetchImpl } = mockDeploymentFetch({ workersDevNoindex: false, omitFrameHeader: true, brokenAsset: true });
  await assert.rejects(
    verifyCloudflareDeployment("https://macmlx.app/", { fetchImpl }),
    (error) => /X-Frame-Options/.test(error.message) && /main\.js/.test(error.message),
  );
});

test("remote verifier rejects wrong-but-present MIME types and cache classes", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  const cases = [
    [{ mimeOverrides: { "/": "application/octet-stream" } }, /GET \/.*Content-Type.*text\/html/is],
    [{ mimeOverrides: { "/robots.txt": "text/html" } }, /robots\.txt.*Content-Type.*text\/plain/is],
    [{ mimeOverrides: { "/sitemap.xml": "text/plain" } }, /sitemap\.xml.*Content-Type.*xml/is],
    [{ mimeOverrides: { "/assets/css/main.css": "application/octet-stream" } }, /main\.css.*Content-Type.*text\/css/is],
    [{ mimeOverrides: { "/assets/js/main.js": "text/plain" } }, /main\.js.*Content-Type.*javascript/is],
    [{ mimeOverrides: { "/assets/brand/site.webmanifest": "application/octet-stream" } }, /site\.webmanifest.*Content-Type.*application\/manifest\+json/is],
    [{ mimeOverrides: { "/assets/images/engine/adaptive-runtime.webp": "image/png" } }, /adaptive-runtime\.webp.*Content-Type.*image\/webp/is],
    [{ mimeOverrides: { "/assets/og-image.svg": "image/png" } }, /og-image\.svg.*Content-Type.*image\/svg\+xml/is],
    [{ mimeOverrides: { "/assets/social/og-en.png": "image/webp" } }, /og-en\.png.*image\/png/is],
    [{ cacheOverrides: { "/robots.txt": "public, max-age=60" } }, /robots\.txt.*Cache-Control.*max-age=0.*must-revalidate/is],
    [{ cacheOverrides: { "/assets/js/main.js": "public, max-age=0, must-revalidate" } }, /main\.js.*Cache-Control.*max-age=3600/is],
    [{ cacheOverrides: { "/assets/brand/site.webmanifest": "public, max-age=0, must-revalidate" } }, /site\.webmanifest.*Cache-Control.*max-age=3600/is],
  ];
  for (const [options, expected] of cases) {
    const { fetchImpl } = mockDeploymentFetch({ workersDevNoindex: false, ...options });
    await assert.rejects(verifyCloudflareDeployment("https://macmlx.app/", { fetchImpl }), expected);
  }
});

test("remote verifier times out every fetch through an injected deterministic AbortSignal", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  let signalCount = 0;
  const timeoutSignalFactory = () => {
    signalCount += 1;
    const controller = new AbortController();
    controller.abort(new DOMException("timed out", "TimeoutError"));
    return controller.signal;
  };
  const fetchImpl = async (_input, { signal }) => {
    assert.equal(signal.aborted, true);
    throw signal.reason;
  };
  await assert.rejects(
    verifyCloudflareDeployment("https://macmlx.app/", { fetchImpl, timeoutMs: 2_500, timeoutSignalFactory }),
    /request timed out after 2500 ms/i,
  );
  assert.ok(signalCount >= 10);
});

test("remote verifier rejects declared and streamed bodies beyond the configured byte cap", async () => {
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  for (const oversizedResponse of [
    () => new Response("small", { status: 200, headers: { "content-type": "text/plain", "content-length": "1025", "cache-control": "public, max-age=0, must-revalidate" } }),
    () => new Response("x".repeat(1025), { status: 200, headers: { "content-type": "text/plain", "cache-control": "public, max-age=0, must-revalidate" } }),
  ]) {
    const base = mockDeploymentFetch({ workersDevNoindex: false });
    const fetchImpl = async (input, init) => new URL(input).pathname === "/robots.txt" ? oversizedResponse() : base.fetchImpl(input, init);
    await assert.rejects(
      verifyCloudflareDeployment("https://macmlx.app/", { fetchImpl, maxBodyBytes: 1_024 }),
      /robots\.txt.*exceeds 1024 bytes/is,
    );
  }
});

test("deployment runbook pins the installed Wrangler location and safe staging-to-production sequence", async () => {
  const readme = await readFile(new URL("site/README.md", root), "utf8");
  assert.doesNotMatch(readme, /\/Users\/|kevin|engine-scroll-story/i);
  assert.match(readme, /SITE_ROOT=\/absolute\/path\/to\/macmlx-site-checkout/);
  assert.match(readme, /WRANGLER_PACKAGE_DIR=\/absolute\/path\/to\/directory-containing-wrangler-package/);
  assert.match(readme, /cd "\$WRANGLER_PACKAGE_DIR"/);
  assert.match(readme, /bun wrangler deploy --dry-run --config "\$SITE_ROOT\/wrangler\.jsonc" --env staging/);
  assert.match(readme, /bun wrangler deploy --config "\$SITE_ROOT\/wrangler\.jsonc" --env staging/);
  const stagingVerifyCommand = readme.match(/node "\$SITE_ROOT\/scripts\/verify-cloudflare-deploy\.mjs" "(https:\/\/macmlx-site-staging\.<your-workers-subdomain>\.workers\.dev\/)"/);
  assert.ok(stagingVerifyCommand, "runbook must pass an exact root URL with its trailing slash");
  const { verifyCloudflareDeployment } = await import("../../scripts/verify-cloudflare-deploy.mjs");
  const stagingURL = stagingVerifyCommand[1].replace("<your-workers-subdomain>", "account");
  const stagingMock = mockDeploymentFetch({ workersDevNoindex: true });
  await assert.doesNotReject(verifyCloudflareDeployment(stagingURL, { fetchImpl: stagingMock.fetchImpl }));
  assert.match(readme, /bun wrangler versions list --config "\$SITE_ROOT\/wrangler\.jsonc"/);
  assert.match(readme, /PREVIOUS_VERSION_ID="<recorded-known-good-version-id>"/);
  assert.match(readme, /bun wrangler deploy --dry-run --config "\$SITE_ROOT\/wrangler\.jsonc"\nbun wrangler deploy --config "\$SITE_ROOT\/wrangler\.jsonc"/);
  assert.match(readme, /node "\$SITE_ROOT\/scripts\/verify-cloudflare-deploy\.mjs" https:\/\/macmlx\.app\//);
  assert.match(readme, /bun wrangler rollback "\$PREVIOUS_VERSION_ID" --config "\$SITE_ROOT\/wrangler\.jsonc"[\s\S]*?node "\$SITE_ROOT\/scripts\/verify-cloudflare-deploy\.mjs" https:\/\/macmlx\.app\//);
  assert.match(readme, /CSP.+deferred.+browser-tested/is);
  assert.match(readme, /custom_domain.+(?:provision|use).+Cloudflare DNS.+certificate/is);
  assert.match(readme, /conflict.+CNAME.+Worker.+Pages.+origin/is);
  assert.match(readme, /captures all paths.+exact host/is);
  assert.match(readme, /preflight.+inventory.+approved migration/is);
  assert.match(readme, /WAF.+zone TLS.+HSTS.+out of scope/is);
});
