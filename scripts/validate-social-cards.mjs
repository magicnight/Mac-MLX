import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { socialCardCaptures, validateSocialPNG } from "../site/lib/social-card.mjs";

const repositoryRoot = new URL("../", import.meta.url);

export async function validateTrackedSocialCards() {
  for (const capture of socialCardCaptures) {
    const png = await readFile(new URL(capture.source, repositoryRoot));
    validateSocialPNG(png, capture.source);
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await validateTrackedSocialCards();
  console.log("Validated 2 tracked social PNGs at 1200x630");
}
