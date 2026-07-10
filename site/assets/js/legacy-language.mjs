export function legacyLanguageDestination(input) {
  const url = input instanceof URL ? input : new URL(input);
  if (url.pathname !== "/" && url.pathname !== "/zh/") return null;
  const language = url.searchParams.get("lang");
  if (language !== "en" && language !== "zh") return null;

  const canonicalPath = language === "zh" ? "/zh/" : "/";
  return `${canonicalPath}${url.hash}`;
}
