import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { outputFileForPath } from "../site/lib/routes.mjs";
import { routes as siteRoutes } from "../site/routes.mjs";

const root = fileURLToPath(new URL("../public/", import.meta.url));
const textExtensions = new Set([".html", ".md", ".txt", ".xml", ".svg", ".css", ".js", ".mjs"]);
const deploymentPolicyFiles = new Set(["_headers", "_redirects"]);

function extension(path) {
  const index = path.lastIndexOf(".");
  return index === -1 ? "" : path.slice(index);
}

function requiredFiles(routes) {
  const files = new Set([
    "llms.txt", "llms-full.txt", "zh/llms.txt", "zh/llms-full.txt",
    "robots.txt", "sitemap.xml", "404.html", "zh/404.html",
    "_headers", "_redirects",
    "assets/social/og-en.png", "assets/social/og-zh.png",
  ]);
  for (const route of routes) {
    for (const path of Object.values(route.paths)) files.add(outputFileForPath(path));
    files.add(`content/en/${route.id}.md`);
    files.add(`content/zh/${route.id}.md`);
  }
  return files;
}

function decodeHTMLReference(value) {
  const named = { amp: "&", quot: '"', apos: "'", lt: "<", gt: ">", colon: ":", sol: "/", tab: "\t", newline: "\n" };
  const decoded = value.replace(/&(?:#(\d+)|#x([\da-f]+)|([a-z]+));/gi, (entity, decimal, hexadecimal, name) => {
    if (decimal !== undefined || hexadecimal !== undefined) {
      const codePoint = Number.parseInt(decimal ?? hexadecimal, decimal === undefined ? 16 : 10);
      try {
        return String.fromCodePoint(codePoint);
      } catch {
        return entity;
      }
    }
    return named[name.toLowerCase()] ?? entity;
  });
  return decoded.replace(/[\t\n\r]/g, "").replace(/^[\u0000-\u0020]+|[\u0000-\u0020]+$/g, "");
}

function localFileFor(reference, sourcePath) {
  let resolved;
  try {
    resolved = new URL(reference, new URL(sourcePath, "https://generated.invalid/"));
  } catch {
    return null;
  }
  if (resolved.origin !== "https://generated.invalid") return null;
  let pathname;
  try {
    pathname = decodeURIComponent(resolved.pathname);
  } catch {
    return null;
  }
  const withoutRoot = pathname.replace(/^\//, "");
  return pathname.endsWith("/") ? `${withoutRoot}index.html` : withoutRoot;
}

function markdownDestinations(source) {
  const destinations = [];
  for (let opening = source.indexOf("]("); opening !== -1; opening = source.indexOf("](", opening + 2)) {
    const start = opening + 2;
    let depth = 1;
    let closing = start;
    for (; closing < source.length; closing += 1) {
      const character = source[closing];
      if (character === "\\") {
        closing += 1;
        continue;
      }
      if (character === "(") depth += 1;
      if (character !== ")") continue;
      depth -= 1;
      if (depth === 0) break;
    }
    if (depth !== 0) continue;
    const content = source.slice(start, closing).trim();
    if (!content) continue;
    if (content.startsWith("<")) {
      const angleClose = content.indexOf(">");
      if (angleClose > 1) destinations.push(content.slice(1, angleClose));
      continue;
    }
    let nested = 0;
    let end = 0;
    for (; end < content.length; end += 1) {
      const character = content[end];
      if (/\s/.test(character) && nested === 0) break;
      if (character === "(") nested += 1;
      if (character === ")" && nested > 0) nested -= 1;
    }
    destinations.push(content.slice(0, end));
  }
  return destinations;
}

function validateMarkdownDestination({ destination, path, files, problems }) {
  destination = decodeHTMLReference(destination);
  if (destination.startsWith("#")) return;
  if (destination.startsWith("//")) {
    problems.push(`unsafe Markdown destination: ${path} -> ${destination}`);
    return;
  }
  if (destination.startsWith("/")) {
    const localPath = localFileFor(destination, path);
    if (!files.has(localPath)) problems.push(`missing local reference: ${path} -> ${localPath}`);
    return;
  }
  const scheme = destination.match(/^([a-z][a-z0-9+.-]*):/i)?.[1]?.toLowerCase();
  if (scheme === undefined) {
    const localPath = localFileFor(destination, path);
    if (localPath === null || !files.has(localPath)) problems.push(`missing local reference: ${path} -> ${localPath ?? destination}`);
    return;
  }
  if (scheme !== "https") {
    problems.push(`unsafe Markdown destination: ${path} -> ${destination}`);
    return;
  }
  try {
    new URL(destination);
  } catch {
    problems.push(`invalid external URL: ${path} -> ${destination}`);
  }
}

export function inspectPublicTree({ files, textByPath, routes = siteRoutes }) {
  const problems = [];
  const expected = requiredFiles(routes);
  for (const path of expected) if (!files.has(path)) {
    const localeRoute = path.endsWith("/index.html") || path === "index.html";
    problems.push(`${localeRoute ? "missing locale counterpart" : "missing required file"}: ${path}`);
  }

  for (const path of files) {
    if (/\.map$|(?:^|\/)(?:\.DS_Store|[^/]+\.tmp|[^/]+~|\.public-(?:build|backup)-)/.test(path)) {
      problems.push(`unexpected generated artifact: ${path}`);
    }
  }

  const canonicalOwners = new Map();
  for (const [path, source] of textByPath) {
    if (/\{\{|\}\}/.test(source)) problems.push(`unresolved token: ${path}`);
    if (path.endsWith(".md") || path.endsWith(".txt")) {
      for (const destination of markdownDestinations(source)) validateMarkdownDestination({ destination, path, files, problems });
    }
    if (!path.endsWith(".html") && !deploymentPolicyFiles.has(path)) {
      for (const match of source.matchAll(/https?:\/\/[^\s<>"']+/g)) {
        const reference = match[0].replace(/[),.;]+$/, "");
        try {
          new URL(reference);
        } catch {
          problems.push(`invalid external URL: ${path} -> ${reference}`);
        }
      }
    }
    if (path === "sitemap.xml" && /(?:^|[/>])_(?:headers|redirects)(?:[<\s]|$)/.test(source)) {
      problems.push(`deployment policy exposed as page: ${path}`);
    }
    if (!path.endsWith(".html")) continue;

    const canonicals = [...source.matchAll(/<link rel="canonical" href="([^"]+)">/g)].map((match) => match[1]);
    if (!path.endsWith("404.html") && canonicals.length !== 1) problems.push(`expected one canonical: ${path}`);
    for (const canonical of canonicals) {
      const owner = canonicalOwners.get(canonical);
      if (owner) problems.push(`duplicate canonical ${canonical}: ${owner}, ${path}`);
      else canonicalOwners.set(canonical, path);
      if (canonical.includes("?lang=")) problems.push(`noncanonical language query: ${path} -> ${canonical}`);
    }

    for (const match of source.matchAll(/(?:href|src)="([^"]+)"/g)) {
      const reference = decodeHTMLReference(match[1]);
      const scheme = reference.match(/^([a-z][a-z0-9+.-]*):/i)?.[1]?.toLowerCase();
      if (reference.startsWith("//") || (scheme !== undefined && scheme !== "https")) {
        problems.push(`unsafe URL scheme: ${path} -> ${reference}`);
        continue;
      }
      if (reference.includes("?lang=")) problems.push(`noncanonical language query: ${path} -> ${reference}`);
      if (reference.startsWith("#")) continue;
      if (scheme === undefined) {
        const localPath = localFileFor(reference, path);
        if (!files.has(localPath)) problems.push(`missing local reference: ${path} -> ${localPath}`);
        continue;
      }
      if (scheme === "https") {
        try {
          new URL(reference);
        } catch {
          problems.push(`invalid external URL: ${path} -> ${reference}`);
        }
      }
    }
  }

  return [...new Set(problems)].sort();
}

async function readTree(directory, prefix = "") {
  const files = new Set();
  const textByPath = new Map();
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const path = prefix ? `${prefix}/${entry.name}` : entry.name;
    const absolute = join(directory, entry.name);
    if (entry.isDirectory()) {
      const child = await readTree(absolute, path);
      child.files.forEach((file) => files.add(file));
      child.textByPath.forEach((content, file) => textByPath.set(file, content));
    } else if (entry.isFile()) {
      files.add(path);
      if (textExtensions.has(extension(path)) || deploymentPolicyFiles.has(path)) textByPath.set(path, await readFile(absolute, "utf8"));
    }
  }
  return { files, textByPath };
}

export async function crawlPublicSite(publicRoot = root) {
  const tree = await readTree(publicRoot);
  const problems = inspectPublicTree({ ...tree, routes: siteRoutes });
  if (problems.length > 0) throw new Error(`Public crawl failed:\n- ${problems.join("\n- ")}`);
  return { fileCount: tree.files.size, htmlCount: [...tree.files].filter((path) => path.endsWith(".html")).length };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const result = await crawlPublicSite();
  console.log(`Crawled ${result.fileCount} generated files (${result.htmlCount} HTML documents)`);
}
