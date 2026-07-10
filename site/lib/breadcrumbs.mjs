const copy = Object.freeze({
  en: Object.freeze({ home: "Home", models: "Models", compare: "Compare", releases: "Releases" }),
  "zh-Hans": Object.freeze({ home: "首页", models: "模型", compare: "对比", releases: "版本" }),
});

export function breadcrumbItems(page, locale) {
  const labels = copy[locale];
  if (!labels) throw new Error(`Unsupported breadcrumb locale: ${locale}`);
  const homePath = locale === "en" ? "/" : "/zh/";
  const segments = page.paths.en.split("/").filter(Boolean);
  const items = [{ name: labels.home, path: homePath }];
  if (segments.length > 1) {
    const parent = segments[0];
    const parentName = labels[parent];
    if (!parentName) throw new Error(`Unknown breadcrumb parent: ${parent}`);
    items.push({ name: parentName, path: locale === "en" ? `/${parent}/` : `/zh/${parent}/` });
  }
  items.push({ name: page[locale].title, path: page.paths[locale] });
  return items;
}
