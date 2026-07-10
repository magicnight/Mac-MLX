import { createHash } from "node:crypto";

export function digestPreparedSite(prepared, staticEntries = []) {
  const entries = [
    ...[...prepared.documents].map(([path, content]) => [`html:${path}`, content]),
    ...[...prepared.markdownDocuments].map(([path, content]) => [`markdown:${path}`, content]),
    ...[...prepared.discoveryFiles].map(([path, content]) => [`discovery:${path}`, content]),
    ...[...prepared.deploymentFiles].map(([path, content]) => [`deployment:${path}`, content]),
    ["generated:assets/og-image.svg", prepared.socialImage],
    ...staticEntries.map(([path, content]) => [`static:${path}`, content]),
  ].sort(([left], [right]) => left.localeCompare(right));
  const hash = createHash("sha256");
  for (const [path, content] of entries) {
    hash.update(path);
    hash.update("\0");
    hash.update(content);
    hash.update("\0");
  }
  return hash.digest("hex");
}
