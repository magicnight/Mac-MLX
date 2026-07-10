const projectKeys = [
  "origin",
  "repositoryURL",
  "downloadURL",
  "currentVersion",
  "releaseDate",
  "lastVerified",
  "licenseURL",
  "operatingSystem",
  "locales",
];

const localeKeys = [
  "htmlLang",
  "ogLocale",
  "title",
  "description",
  "socialDescription",
  "twitterDescription",
  "structuredDescription",
  "primaryNavigationLabel",
  "themeToggleLabel",
  "languageLinkLabel",
  "copySuccess",
  "copyFailure",
];

function validateRecord(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function validateExactKeys(value, requiredKeys, label) {
  validateRecord(value, label);
  for (const key of requiredKeys) {
    if (!Object.hasOwn(value, key)) throw new Error(`Missing ${label} key: ${key}`);
  }
  for (const key of Object.keys(value)) {
    if (!requiredKeys.includes(key)) throw new Error(`Unknown ${label} key: ${key}`);
  }
}

function validateString(value, label) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${label} must be a non-empty string`);
  }
}

const trustedRepositoryURL = "https://github.com/magicnight/mac-mlx";
const trustedLicenseURL = "https://www.apache.org/licenses/LICENSE-2.0";

function validateURL(value, label) {
  validateString(value, label);
  if (value !== value.trim()) throw new Error(`${label} must not contain surrounding whitespace`);
  if (/[\u0000-\u0020\u007f"'<>]/.test(value)) throw new Error(`${label} must be a valid URL`);
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${label} must be a valid URL`);
  }
  if (parsed.protocol !== "https:") throw new Error(`${label} must use HTTPS`);
  if (parsed.username || parsed.password) throw new Error(`${label} must not contain credentials`);
  return parsed;
}

function validateProjectURLs(value) {
  const origin = validateURL(value.origin, "project.origin");
  if (origin.href !== `${origin.origin}/`) throw new Error("project.origin must be origin-only");

  validateURL(value.repositoryURL, "project.repositoryURL");
  if (value.repositoryURL !== trustedRepositoryURL) throw new Error("project.repositoryURL must equal the trusted repository URL");

  const download = validateURL(value.downloadURL, "project.downloadURL");
  const repository = new URL(trustedRepositoryURL);
  const releasePrefix = `${repository.pathname}/releases/`;
  if (download.origin !== repository.origin || !download.pathname.startsWith(releasePrefix) || download.pathname === releasePrefix || download.search || download.hash) {
    throw new Error("project.downloadURL must stay under the repository releases path");
  }

  validateURL(value.licenseURL, "project.licenseURL");
  if (value.licenseURL !== trustedLicenseURL) throw new Error("project.licenseURL must equal the trusted license URL");
}

export function validateProject(value) {
  validateExactKeys(value, projectKeys, "project");
  for (const key of projectKeys.filter((key) => key !== "locales")) {
    validateString(value[key], `project.${key}`);
  }
  validateProjectURLs(value);

  validateExactKeys(value.locales, ["en", "zh-Hans"], "project.locales");
  for (const locale of ["en", "zh-Hans"]) {
    const metadata = value.locales[locale];
    validateExactKeys(metadata, localeKeys, `project.locales.${locale}`);
    for (const key of localeKeys) {
      validateString(metadata[key], `project.locales.${locale}.${key}`);
    }
  }
  return value;
}

export function siteURL(project, path) {
  return new URL(path, project.origin).href;
}
