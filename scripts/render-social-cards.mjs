import { randomUUID } from "node:crypto";
import { mkdir, rename, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { project } from "../site/content/project.mjs";
import { embedSocialSourceDigest, renderSocialCardSVG, socialCardCaptures, socialCardSourceDigest, validateSocialPNG } from "../site/lib/social-card.mjs";

const require = createRequire(import.meta.url);
const repositoryRoot = fileURLToPath(new URL("../", import.meta.url));

function loadSharp() {
  try {
    return require("sharp");
  } catch (projectError) {
    const moduleDirectory = process.env.MACMLX_NODE_MODULES;
    if (moduleDirectory) {
      try {
        return require(require.resolve("sharp", { paths: [resolve(moduleDirectory)] }));
      } catch (runtimeError) {
        throw new Error(`Unable to load Sharp from MACMLX_NODE_MODULES=${moduleDirectory}. Point it at a node_modules directory containing Sharp.`, { cause: runtimeError });
      }
    }
    throw new Error("Sharp is required only to refresh social cards. Provide it through normal module resolution or set MACMLX_NODE_MODULES to a node_modules directory containing Sharp; the normal site build uses the tracked PNGs and does not require Sharp.", { cause: projectError });
  }
}

export async function renderSocialCards({ outputDirectory = join(repositoryRoot, "site/assets/social"), render } = {}) {
  const parent = dirname(outputDirectory);
  const name = basename(outputDirectory);
  const unique = `${process.pid}-${randomUUID()}`;
  const stagingDirectory = join(parent, `.${name}.stage-${unique}`);
  const backupDirectory = join(parent, `.${name}.backup-${unique}`);
  const lockDirectory = join(parent, `.${name}.render.lock`);
  await mkdir(parent, { recursive: true });
  try {
    await mkdir(lockDirectory);
  } catch (error) {
    if (error.code === "EEXIST") throw new Error(`Another social-card render is already running for ${outputDirectory}`, { cause: error });
    throw error;
  }

  let backedUp = false;
  let published = false;
  try {
    await mkdir(stagingDirectory);
    let renderCard = render;
    if (renderCard === undefined) {
      const sharp = loadSharp();
      renderCard = async ({ svg }) => sharp(svg, { density: 72 })
        .resize(1200, 630, { fit: "fill" })
        .png({ adaptiveFiltering: false, compressionLevel: 9, effort: 10, palette: false })
        .toBuffer();
    }
    for (const capture of socialCardCaptures) {
      const svg = Buffer.from(renderSocialCardSVG({ project, locale: capture.locale }));
      const expectedSourceDigest = socialCardSourceDigest({ project, locale: capture.locale });
      const png = embedSocialSourceDigest(await renderCard({ capture, svg }), expectedSourceDigest);
      validateSocialPNG(png, capture.source, { expectedSourceDigest });
      await writeFile(join(stagingDirectory, basename(capture.source)), png);
    }

    try {
      await rename(outputDirectory, backupDirectory);
      backedUp = true;
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    try {
      await rename(stagingDirectory, outputDirectory);
      published = true;
    } catch (error) {
      if (backedUp) {
        await rename(backupDirectory, outputDirectory);
        backedUp = false;
      }
      throw error;
    }
    if (backedUp) {
      await rm(backupDirectory, { recursive: true, force: true });
      backedUp = false;
    }
    console.log("Rendered 2 deterministic social PNGs at 1200x630");
  } finally {
    if (!published && backedUp) {
      try {
        await rename(backupDirectory, outputDirectory);
        backedUp = false;
      } catch {
        // The original error remains primary; an existing output is never removed here.
      }
    }
    await rm(stagingDirectory, { recursive: true, force: true });
    if (!backedUp) await rm(backupDirectory, { recursive: true, force: true });
    await rm(lockDirectory, { recursive: true, force: true });
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await renderSocialCards();
