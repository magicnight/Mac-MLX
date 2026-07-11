import { fileURLToPath } from "node:url";

const productionOrigin = "https://macmlx.app";
const documentCache = "public, max-age=0, must-revalidate";
const scriptCache = "public, max-age=3600, stale-while-revalidate=86400";
const imageCache = "public, max-age=604800, stale-while-revalidate=86400";
const securityHeaders = Object.freeze({
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "X-Frame-Options": "DENY",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Resource-Policy": "same-origin",
});

function deploymentBase(value) {
  if (typeof value !== "string" || !value) throw new Error(`Invalid deployment base URL: ${value}`);
  const authority = value.match(/^[a-z][a-z0-9+.-]*:\/\/([^/?#]*)/i)?.[1] ?? "";
  const hostPort = authority.slice(authority.lastIndexOf("@") + 1);
  const hasExplicitPort = hostPort.startsWith("[") ? hostPort.includes("]:") : hostPort.includes(":");
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`Invalid deployment base URL: ${value}`);
  }
  if (url.protocol !== "https:") throw new Error(`Deployment base URL must use HTTPS: ${value}`);
  if (hasExplicitPort) throw new Error(`Deployment base URL must not include an explicit port: ${value}`);
  if (url.username || url.password || url.search || url.hash) throw new Error(`Deployment base URL must not include credentials, a query, or a fragment: ${value}`);
  if (!value.endsWith("/")) throw new Error(`Deployment base URL must include the exact root-path slash: ${value}`);
  if (url.pathname !== "/") throw new Error(`Deployment base URL must use the exact root path: ${value}`);
  const labels = url.hostname.split(".");
  const dnsLabel = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/;
  const first = labels[0] ?? "";
  const stagingName = "macmlx-site-staging";
  const previewPrefix = first.endsWith(`-${stagingName}`) ? first.slice(0, -stagingName.length - 1) : "";
  const stagingTarget = labels.length === 4
    && labels[2] === "workers"
    && labels[3] === "dev"
    && dnsLabel.test(labels[1])
    && (first === stagingName || (dnsLabel.test(previewPrefix) && first.length <= 63));
  if (url.hostname !== "macmlx.app" && !stagingTarget) {
    throw new Error(`Deployment target is not the macmlx.app production origin or approved staging host: ${value}`);
  }
  return url;
}

function headerValue(response, name) {
  return response.headers.get(name)?.trim() ?? "";
}

function mediaType(response) {
  return headerValue(response, "content-type").split(";", 1)[0].trim().toLowerCase();
}

function hasDocumentCache(response) {
  const directives = headerValue(response, "cache-control").toLowerCase().split(",").map((item) => item.trim());
  return directives.includes("max-age=0") && directives.includes("must-revalidate");
}

function canonicalFrom(html) {
  return html.match(/<link\s+rel="canonical"\s+href="([^"]+)"/i)?.[1];
}

function primaryAssets(html) {
  return [...new Set([...html.matchAll(/(?:src|href)="(\/assets\/[^"#]+)(?:#[^"]*)?"/g)].map((match) => match[1]))];
}

function pngDimensions(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.length < 24 || !signature.every((byte, index) => bytes[index] === byte)) throw new Error("body is not a PNG");
  if (String.fromCharCode(...bytes.slice(12, 16)) !== "IHDR") throw new Error("PNG is missing its leading IHDR chunk");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { width: view.getUint32(16), height: view.getUint32(20) };
}

function defaultTimeoutSignal(timeoutMs) {
  if (typeof AbortSignal.timeout === "function") return AbortSignal.timeout(timeoutMs);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException("timed out", "TimeoutError")), timeoutMs);
  timer.unref?.();
  return controller.signal;
}

async function boundedBody(response, maxBodyBytes) {
  const declaredLength = headerValue(response, "content-length");
  if (/^\d+$/.test(declaredLength) && Number(declaredLength) > maxBodyBytes) {
    await response.body?.cancel();
    throw new Error(`declared Content-Length ${declaredLength} exceeds ${maxBodyBytes} bytes`);
  }
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > maxBodyBytes) {
        await reader.cancel();
        throw new Error(`streamed body exceeds ${maxBodyBytes} bytes`);
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const body = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

export async function verifyCloudflareDeployment(baseURL, {
  fetchImpl = globalThis.fetch,
  timeoutMs = 10_000,
  maxBodyBytes = 2 * 1024 * 1024,
  timeoutSignalFactory = defaultTimeoutSignal,
} = {}) {
  if (typeof fetchImpl !== "function") throw new Error("A fetch implementation is required");
  if (!Number.isInteger(timeoutMs) || timeoutMs <= 0) throw new Error("timeoutMs must be a positive integer");
  if (!Number.isInteger(maxBodyBytes) || maxBodyBytes <= 0) throw new Error("maxBodyBytes must be a positive integer");
  if (typeof timeoutSignalFactory !== "function") throw new Error("timeoutSignalFactory must be a function");
  const base = deploymentBase(baseURL);
  const failures = [];
  let checks = 0;

  function check(label, assertion) {
    checks += 1;
    try {
      assertion();
    } catch (error) {
      failures.push(`${label}: ${error.message}`);
    }
  }

  async function request(path, init, label) {
    const signal = timeoutSignalFactory(timeoutMs);
    if (!(signal instanceof AbortSignal)) {
      failures.push(`${label}: timeoutSignalFactory did not return an AbortSignal`);
      return null;
    }
    try {
      return await fetchImpl(new URL(path, base), { ...init, signal });
    } catch (error) {
      if (signal.aborted && (signal.reason?.name === "TimeoutError" || error?.name === "TimeoutError")) {
        failures.push(`${label}: request timed out after ${timeoutMs} ms`);
      } else {
        failures.push(`${label}: request failed: ${error.message}`);
      }
      return null;
    }
  }

  async function textResponse(path, label) {
    const response = await request(path, { method: "GET", redirect: "manual" }, label);
    if (!response) return null;
    let body = "";
    try {
      body = new TextDecoder().decode(await boundedBody(response, maxBodyBytes));
    } catch (error) {
      failures.push(`${label}: response body could not be read: ${error.message}`);
    }
    return { response, body };
  }

  const routeResults = new Map();
  for (const path of ["/", "/zh/", "/architecture/", "/zh/architecture/"]) {
    const result = await textResponse(path, `GET ${path}`);
    routeResults.set(path, result);
    if (!result) continue;
    check(`GET ${path} status`, () => {
      if (result.response.status !== 200) throw new Error(`expected 200, received ${result.response.status}`);
    });
    check(`GET ${path} canonical`, () => {
      const expected = `${productionOrigin}${path}`;
      const actual = canonicalFrom(result.body);
      if (actual !== expected) throw new Error(`expected ${expected}, received ${actual ?? "none"}`);
    });
    check(`GET ${path} Content-Type`, () => {
      const actual = mediaType(result.response);
      if (actual !== "text/html") throw new Error(`expected text/html, received ${actual || "none"}`);
    });
    check(`GET ${path} Cache-Control`, () => {
      if (!hasDocumentCache(result.response)) throw new Error(`expected max-age=0 and must-revalidate, received ${headerValue(result.response, "cache-control") || "none"}`);
    });
  }

  const root = routeResults.get("/");
  if (root) {
    for (const [name, expected] of Object.entries(securityHeaders)) {
      check(`GET / ${name}`, () => {
        const actual = headerValue(root.response, name);
        if (actual !== expected) throw new Error(`expected ${expected}, received ${actual || "none"}`);
      });
    }
    check("GET / Cache-Control", () => {
      const actual = headerValue(root.response, "cache-control");
      if (actual !== documentCache) throw new Error(`expected ${documentCache}, received ${actual || "none"}`);
    });
    if (base.hostname.endsWith(".workers.dev")) {
      check("GET / X-Robots-Tag", () => {
        const actual = headerValue(root.response, "x-robots-tag");
        if (!/\bnoindex\b/i.test(actual) || !/\bnofollow\b/i.test(actual)) throw new Error(`expected noindex and nofollow, received ${actual || "none"}`);
      });
    }
  }

  for (const [path, destination] of [["/index.html", "/"], ["/zh/index.html", "/zh/"]]) {
    const response = await request(path, { method: "HEAD", redirect: "manual" }, `HEAD ${path}`);
    if (!response) continue;
    check(`HEAD ${path} redirect`, () => {
      const location = headerValue(response, "location");
      const resolved = location ? new URL(location, base) : null;
      const exactDestination = resolved
        && resolved.origin === base.origin
        && resolved.pathname === destination
        && !resolved.search
        && !resolved.hash;
      if (response.status !== 301 || !exactDestination) {
        throw new Error(`expected exact same-origin 301 to ${destination} without query or fragment, received ${response.status} to ${location || "none"}`);
      }
    });
  }

  const notFound = await textResponse("/__macmlx-deployment-verification__/", "GET missing route");
  if (notFound) {
    check("missing route status", () => {
      if (notFound.response.status !== 404) throw new Error(`expected 404, received ${notFound.response.status}`);
    });
    check("missing route robots", () => {
      if (!/<meta\s+name="robots"\s+content="[^"]*noindex/i.test(notFound.body)) throw new Error("expected a noindex robots meta tag");
    });
    check("missing route Content-Type", () => {
      const actual = mediaType(notFound.response);
      if (actual !== "text/html") throw new Error(`expected text/html, received ${actual || "none"}`);
    });
    check("missing route Cache-Control", () => {
      if (!hasDocumentCache(notFound.response)) throw new Error(`expected max-age=0 and must-revalidate, received ${headerValue(notFound.response, "cache-control") || "none"}`);
    });
  }

  const discoveryChecks = [
    ["/robots.txt", "text/plain", (body) => /User-agent:\s*\*/i.test(body) && body.includes(`${productionOrigin}/sitemap.xml`), "crawler policy and production sitemap"],
    ["/sitemap.xml", "xml", (body) => body.includes(`<loc>${productionOrigin}/</loc>`) && body.includes(`<loc>${productionOrigin}/zh/</loc>`), "English and Chinese canonical URLs"],
    ["/llms.txt", "text/plain", (body) => /macMLX/i.test(body), "macMLX content"],
    ["/zh/llms.txt", "text/plain", (body) => /macMLX/i.test(body), "localized macMLX content"],
  ];
  for (const [path, expectedType, validate, expected] of discoveryChecks) {
    const result = await textResponse(path, `GET ${path}`);
    if (!result) continue;
    check(`GET ${path}`, () => {
      if (result.response.status !== 200) throw new Error(`expected 200, received ${result.response.status}`);
      if (!validate(result.body)) throw new Error(`expected ${expected}`);
    });
    check(`GET ${path} Content-Type`, () => {
      const actual = mediaType(result.response);
      const matches = expectedType === "xml" ? /^(?:application|text)\/xml$/.test(actual) : actual === expectedType;
      if (!matches) throw new Error(`expected ${expectedType}, received ${actual || "none"}`);
    });
    check(`GET ${path} Cache-Control`, () => {
      if (!hasDocumentCache(result.response)) throw new Error(`expected max-age=0 and must-revalidate, received ${headerValue(result.response, "cache-control") || "none"}`);
    });
  }

  const manifestPath = "/assets/brand/site.webmanifest";
  const manifestResult = await textResponse(manifestPath, `GET ${manifestPath}`);
  let manifestIcons = [];
  if (manifestResult) {
    check(`GET ${manifestPath}`, () => {
      if (manifestResult.response.status !== 200) throw new Error(`expected 200, received ${manifestResult.response.status}`);
    });
    check(`GET ${manifestPath} Content-Type`, () => {
      const actual = mediaType(manifestResult.response);
      if (actual !== "application/manifest+json") throw new Error(`expected application/manifest+json, received ${actual || "none"}`);
    });
    check(`GET ${manifestPath} Cache-Control`, () => {
      const actual = headerValue(manifestResult.response, "cache-control");
      if (actual !== scriptCache) throw new Error(`expected ${scriptCache}, received ${actual || "none"}`);
    });
    let manifest;
    try {
      manifest = JSON.parse(manifestResult.body);
    } catch (error) {
      failures.push(`GET ${manifestPath} JSON: ${error.message}`);
    }
    if (manifest) {
      const requiredIcons = [[192, "/assets/brand/icon-192.png"], [512, "/assets/brand/icon-512.png"]];
      check(`GET ${manifestPath} required install structure`, () => {
        const validRoot = manifest && typeof manifest === "object" && !Array.isArray(manifest)
          && manifest.name === "macMLX" && manifest.short_name === "macMLX"
          && manifest.start_url === "/" && manifest.display === "standalone" && Array.isArray(manifest.icons);
        if (!validRoot) throw new Error("expected macMLX name, short_name, root start_url, standalone display, and icons array");
        manifestIcons = requiredIcons.map(([size, path]) => {
          const icon = manifest.icons.find((entry) => entry?.sizes === `${size}x${size}`);
          if (!icon || icon.type !== "image/png") throw new Error(`expected ${size}x${size} image/png icon`);
          const resolved = new URL(icon.src, base);
          const safe = resolved.origin === base.origin && resolved.pathname === path && !resolved.search && !resolved.hash;
          if (!safe) throw new Error(`expected same-origin ${path}, received ${icon.src ?? "none"}`);
          return { size, path };
        });
      });
    }
  }

  for (const { size, path } of manifestIcons) {
    const response = await request(path, { method: "GET", redirect: "manual" }, `GET ${path}`);
    if (!response) continue;
    let body;
    try {
      const bounded = await boundedBody(response, maxBodyBytes);
      body = bounded.buffer.slice(bounded.byteOffset, bounded.byteOffset + bounded.byteLength);
    } catch (error) {
      failures.push(`GET ${path}: response body could not be read: ${error.message}`);
    }
    check(`GET ${path}`, () => {
      if (response.status !== 200) throw new Error(`expected 200, received ${response.status}`);
      if (mediaType(response) !== "image/png") throw new Error(`expected image/png, received ${mediaType(response) || "none"}`);
      const cache = headerValue(response, "cache-control");
      if (cache !== imageCache) throw new Error(`expected Cache-Control ${imageCache}, received ${cache || "none"}`);
      if (!body) throw new Error("PNG body is unavailable");
      const dimensions = pngDimensions(body);
      if (dimensions.width !== size || dimensions.height !== size) throw new Error(`expected ${size}x${size}, received ${dimensions.width}x${dimensions.height}`);
    });
  }

  const socialPath = "/assets/social/og-en.png";
  const socialResponse = await request(socialPath, { method: "GET", redirect: "manual" }, `GET ${socialPath}`);
  if (socialResponse) {
    let socialBody;
    try {
      const bounded = await boundedBody(socialResponse, maxBodyBytes);
      socialBody = bounded.buffer.slice(bounded.byteOffset, bounded.byteOffset + bounded.byteLength);
    } catch (error) {
      failures.push(`GET ${socialPath}: response body could not be read: ${error.message}`);
    }
    check(`GET ${socialPath}`, () => {
      if (socialResponse.status !== 200) throw new Error(`expected 200, received ${socialResponse.status}`);
      const contentType = headerValue(socialResponse, "content-type");
      if (!/^image\/png\b/i.test(contentType)) throw new Error(`expected image/png, received ${contentType || "none"}`);
      if (!socialBody) throw new Error("PNG body is unavailable");
      const dimensions = pngDimensions(socialBody);
      if (dimensions.width !== 1200 || dimensions.height !== 630) throw new Error(`expected 1200x630, received ${dimensions.width}x${dimensions.height}`);
      const cache = headerValue(socialResponse, "cache-control");
      if (cache !== imageCache) throw new Error(`expected ${imageCache}, received ${cache || "none"}`);
    });
  }

  for (const asset of primaryAssets(root?.body ?? "")) {
    const response = await request(asset, { method: "HEAD", redirect: "manual" }, `HEAD ${asset}`);
    if (!response) continue;
    check(`HEAD ${asset}`, () => {
      if (response.status !== 200) throw new Error(`expected 200, received ${response.status}`);
      const expectedCache = /\.(?:webp|png|svg)(?:\?|$)/i.test(asset) ? imageCache : scriptCache;
      const actualCache = headerValue(response, "cache-control");
      if (actualCache !== expectedCache) throw new Error(`expected Cache-Control ${expectedCache}, received ${actualCache || "none"}`);
    });
    check(`HEAD ${asset} Content-Type`, () => {
      const path = new URL(asset, base).pathname;
      const expected = path.endsWith(".css") ? "text/css"
        : path.endsWith(".js") || path.endsWith(".mjs") ? "javascript"
          : path.endsWith(".webmanifest") ? "application/manifest+json"
          : path.endsWith(".svg") ? "image/svg+xml"
            : path.endsWith(".png") ? "image/png"
              : path.endsWith(".webp") ? "image/webp"
                : "";
      const actual = mediaType(response);
      const matches = expected === "javascript" ? /^(?:application|text)\/javascript$/.test(actual) : actual === expected;
      if (!expected || !matches) throw new Error(`expected ${expected || "a known asset MIME type"}, received ${actual || "none"}`);
    });
  }

  if (failures.length > 0) {
    const error = new Error(`Cloudflare deployment verification failed for ${base.origin}:\n- ${failures.join("\n- ")}`);
    error.failures = failures;
    throw error;
  }
  return { baseURL: base.origin, checks };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const baseURL = process.argv[2];
  if (!baseURL) {
    console.error("Usage: node scripts/verify-cloudflare-deploy.mjs <base-url>");
    process.exitCode = 2;
  } else {
    try {
      const result = await verifyCloudflareDeployment(baseURL);
      console.log(`Verified ${result.checks} deployment checks against ${result.baseURL}`);
    } catch (error) {
      console.error(error.message);
      process.exitCode = 1;
    }
  }
}
