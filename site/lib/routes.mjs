export function validateCanonicalPath(path) {
  if (
    typeof path !== "string"
    || !/^\/(?:[a-z0-9]+(?:-[a-z0-9]+)*\/)*$/.test(path)
    || path.includes("//")
    || path.includes("/../")
  ) {
    throw new Error(`Invalid canonical path: ${path}`);
  }
  return path;
}

export function outputFileForPath(path) {
  const canonicalPath = validateCanonicalPath(path);
  return `${canonicalPath.slice(1)}index.html`;
}
