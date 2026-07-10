import assert from "node:assert/strict";
import test from "node:test";

import { copiedAssetPaths } from "../content/assets.mjs";
import { outputFileForPath } from "../lib/routes.mjs";
import { routes } from "../routes.mjs";
import { prepareSite } from "../../scripts/build-public-site.mjs";
import { inspectPublicTree } from "../../scripts/crawl-public-site.mjs";

async function validTree() {
  const prepared = await prepareSite({ today: "2026-07-10" });
  const textByPath = new Map();
  for (const [path, html] of prepared.documents) textByPath.set(outputFileForPath(path), html);
  for (const pair of [...prepared.markdownDocuments, ...prepared.discoveryFiles, ...prepared.deploymentFiles]) textByPath.set(...pair);
  textByPath.set("assets/og-image.svg", prepared.socialImage);
  const files = new Set([...textByPath.keys(), ...copiedAssetPaths.map((path) => `assets/${path}`), "assets/social/og-en.png", "assets/social/og-zh.png"]);
  return { files, textByPath };
}

test("crawler accepts the complete deterministic generated inventory without network access", async () => {
  const tree = await validTree();
  assert.deepEqual(inspectPublicTree({ ...tree, routes }), []);
});

test("crawler requires deployment policies without treating them as indexable pages", async () => {
  const tree = await validTree();
  tree.files.delete("_headers");
  tree.files.delete("_redirects");
  const problems = inspectPublicTree({ ...tree, routes });
  assert.ok(problems.includes("missing required file: _headers"));
  assert.ok(problems.includes("missing required file: _redirects"));

  const valid = await validTree();
  valid.textByPath.set("sitemap.xml", `${valid.textByPath.get("sitemap.xml")}\n<loc>https://macmlx.app/_headers</loc>`);
  assert.ok(inspectPublicTree({ ...valid, routes }).some((problem) => problem.includes("deployment policy exposed as page")));
});

test("crawler reports missing social cards, broken locales, bad URLs, unresolved tokens, and temp artifacts", async () => {
  const tree = await validTree();
  tree.files.delete("assets/social/og-en.png");
  tree.files.delete("zh/architecture/index.html");
  tree.files.add("assets/js/main.js.map");
  tree.textByPath.set("content/en/home.md", `${tree.textByPath.get("content/en/home.md")}\n{{BROKEN}}`);
  tree.textByPath.set("index.html", tree.textByPath.get("index.html").replace('href="#top"', 'href="javascript:alert(1)"').replace('href="/architecture/"', 'href="/architecture/?lang=zh"'));
  const problems = inspectPublicTree({ ...tree, routes });
  assert.ok(problems.some((problem) => problem.includes("missing required file: assets/social/og-en.png")));
  assert.ok(problems.some((problem) => problem.includes("missing locale counterpart: zh/architecture/index.html")));
  assert.ok(problems.some((problem) => problem.includes("noncanonical language query")));
  assert.ok(problems.some((problem) => problem.includes("unsafe URL scheme")));
  assert.ok(problems.some((problem) => problem.includes("unresolved token")));
  assert.ok(problems.some((problem) => problem.includes("unexpected generated artifact")));
});

test("crawler detects duplicate canonicals and syntactically invalid external links", async () => {
  const tree = await validTree();
  tree.files.add("copy.html");
  tree.textByPath.set("copy.html", tree.textByPath.get("index.html").replace('href="https://github.com/magicnight/mac-mlx"', 'href="https://["'));
  const problems = inspectPublicTree({ ...tree, routes });
  assert.ok(problems.some((problem) => problem.includes("duplicate canonical")));
  assert.ok(problems.some((problem) => problem.includes("invalid external URL")));
});

test("crawler validates external URLs in generated Markdown and text without network access", async () => {
  const tree = await validTree();
  tree.textByPath.set("content/en/home.md", `${tree.textByPath.get("content/en/home.md")}\n- https://[`);
  const problems = inspectPublicTree({ ...tree, routes });
  assert.ok(problems.some((problem) => problem.includes("invalid external URL: content/en/home.md -> https://[")));
});

test("crawler rejects spaced and dangling unresolved template residue in generated text", async () => {
  const tree = await validTree();
  tree.textByPath.set("content/en/home.md", `${tree.textByPath.get("content/en/home.md")}\n{{ BROKEN }}`);
  tree.textByPath.set("llms.txt", `${tree.textByPath.get("llms.txt")}\n}}`);
  const problems = inspectPublicTree({ ...tree, routes });
  assert.ok(problems.some((problem) => problem === "unresolved token: content/en/home.md"));
  assert.ok(problems.some((problem) => problem === "unresolved token: llms.txt"));
});

test("crawler rejects unsafe Markdown destinations while preserving local and HTTPS links", async () => {
  const unsafeDestinations = [
    "javascript:alert(1)",
    "data:text/html,unsafe",
    "file:///tmp/unsafe",
    "//evil.example/unsafe",
    "http://evil.example/unsafe",
    "mailto:unsafe@example.com",
  ];
  for (const destination of unsafeDestinations) {
    const tree = await validTree();
    tree.textByPath.set("content/en/home.md", `${tree.textByPath.get("content/en/home.md")}\n[x](${destination})`);
    const problems = inspectPublicTree({ ...tree, routes });
    assert.ok(
      problems.some((problem) => problem.includes(`unsafe Markdown destination: content/en/home.md -> ${destination}`)),
      `expected unsafe destination rejection for ${destination}`,
    );
  }

  const valid = await validTree();
  valid.textByPath.set("content/en/home.md", `${valid.textByPath.get("content/en/home.md")}\n[home](/)\n[relative](architecture.md)\n[source](https://github.com/magicnight/mac-mlx)`);
  assert.deepEqual(inspectPublicTree({ ...valid, routes }), []);
});

test("crawler resolves relative HTML and Markdown destinations against their containing files", async () => {
  const missing = ["missing/", "../missing/", "missing.md"];
  for (const destination of missing) {
    const tree = await validTree();
    tree.textByPath.set("content/en/home.md", `${tree.textByPath.get("content/en/home.md")}\n[x](${destination})`);
    const problems = inspectPublicTree({ ...tree, routes });
    assert.ok(problems.some((problem) => problem.includes("missing local reference: content/en/home.md")), destination);
  }

  const htmlTree = await validTree();
  htmlTree.textByPath.set("index.html", htmlTree.textByPath.get("index.html").replace('href="#top"', 'href="missing/"'));
  assert.ok(inspectPublicTree({ ...htmlTree, routes }).some((problem) => problem.includes("missing local reference: index.html -> missing/index.html")));

  const valid = await validTree();
  valid.textByPath.set("content/en/home.md", `${valid.textByPath.get("content/en/home.md")}\n[peer](architecture.md)\n[root](/architecture/?view=all#top)`);
  valid.textByPath.set("index.html", valid.textByPath.get("index.html").replace('href="#top"', 'href="architecture/?view=all#top"'));
  assert.deepEqual(inspectPublicTree({ ...valid, routes }), []);
});

test("crawler decodes entities and browser whitespace before rejecting unsafe HTML destinations", async () => {
  const unsafe = [
    " &#x6a;avascript&colon;alert(1)",
    "java&#x73;cript&#58;alert(1)",
    "data&colon;text/html,unsafe",
    "file&colon;&#47;&#47;&#47;tmp/unsafe",
    "http&colon;&#47;&#47;evil.example/unsafe",
    "mailto&colon;unsafe@example.com",
    "&#47;&#47;evil.example/unsafe",
  ];
  for (const destination of unsafe) {
    const tree = await validTree();
    tree.textByPath.set("index.html", tree.textByPath.get("index.html").replace('href="#top"', `href="${destination}"`));
    assert.ok(
      inspectPublicTree({ ...tree, routes }).some((problem) => problem.includes("unsafe URL scheme: index.html")),
      destination,
    );
  }
});
