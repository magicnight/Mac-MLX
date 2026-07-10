# Bilingual SEO and GEO Content Hub Design

**Date:** 2026-07-10

**Status:** Awaiting written specification approval

**Surface:** `public/` marketing website and its build-time source files

## Purpose

Turn the macMLX marketing site into a bilingual, citation-ready product and technical knowledge hub without weakening the existing high-impact landing page. English and Simplified Chinese must receive equal information depth, independent crawlable URLs, equivalent metadata, and the same factual boundaries.

The work should improve conventional search discovery and generative-answer discovery by making macMLX claims easy to crawl, understand, verify, and cite. It must not promise rankings, rich results, or inclusion in any answer engine.

## Confirmed Product Decisions

- English and Simplified Chinese receive equal priority.
- English uses `/`; Chinese uses `/zh/`.
- The project will build a complete content hub, not only optimize the home page.
- Competitor content is neutral, versioned, evidence-backed, and sourced from official material.
- Version-sensitive content is reviewed manually with every macMLX release and displays its last verified date.
- The implementation uses a hybrid static generator: preserve the current landing-page DOM and interaction design while generating localized static HTML and the new content hub at build time.

## Current-State Findings

The existing site already has a strong foundation:

- A crawlable English home page with substantial static HTML.
- Client-side English and Chinese copy through `data-en` and `data-zh` attributes.
- A canonical URL, language alternates, Open Graph metadata, Twitter metadata, `SoftwareApplication` JSON-LD, `robots.txt`, `sitemap.xml`, and `llms.txt`.
- A no-JavaScript stylesheet and readable static fallbacks for the engine story.

The main gaps are:

- Chinese is selected through `?lang=zh` and JavaScript rather than being served as an independent static document.
- The canonical always points to the English root, so the query-string Chinese state is not a durable Chinese canonical page.
- The sitemap lists only the home page and has a stale modification date.
- Page titles, descriptions, social metadata, and structured data do not exist for topic-specific search intent.
- The Open Graph image is SVG; a 1200 x 630 PNG is more broadly interoperable with social preview consumers.
- The content is concentrated on one landing page, leaving architecture, API compatibility, model choice, comparisons, releases, and common questions without stable citation targets.
- The current `SoftwareApplication` entity has useful semantic data but no verified rating or review, so it must not be presented as eligible for Google's software-app rich result.
- The long explicit crawler list in `robots.txt` creates unnecessary maintenance even though the default policy already allows crawling.

## Design Approach

Use a hybrid static content system.

- The home page retains its existing visual structure, engine scrollytelling, CSS, images, and interaction behavior.
- Home-page copy and metadata move to build-time locale values without changing the rendered layout.
- New technical pages use a shared editorial template.
- A bilingual fact registry supplies release status, version, verification date, official sources, and localized wording.
- A dependency-free Node build script generates complete English and Chinese HTML, Markdown alternates, sitemap files, and LLM-oriented indexes into `public/`.
- Browser JavaScript remains optional enhancement. It does not generate indexable body content or translate the document after load.

This approach preserves the proven visual work while avoiding the drift of manually duplicating every fact across two languages and multiple pages.

## Information Architecture

English routes:

```text
/
├── architecture/
├── api-compatibility/
├── faq/
├── models/
│   ├── choosing-a-model/
│   └── vision-language-models/
├── compare/
│   ├── ollama/
│   ├── lm-studio/
│   └── omlx/
└── releases/
    └── v0-5-3/
```

Chinese mirrors every route under `/zh/` with the same route segments. Keeping route segments stable makes counterpart mapping deterministic and simplifies hreflang generation.

Additional discovery files:

```text
/llms.txt
/llms-full.txt
/zh/llms.txt
/zh/llms-full.txt
/content/en/*.md
/content/zh/*.md
/sitemap.xml
/robots.txt
```

Swama and SwiftLM appear in the comparison overview but do not receive individual pages in this scope. This avoids thin pages for lower-volume intents while still representing the competitive landscape.

## Navigation and Internal Links

- Add a visible `Learn / 了解` entry from the home page to the architecture page or content-hub landing surface.
- Keep the current home-page section anchors.
- Expand the footer into concise product, learn, compare, and project groups.
- Knowledge pages use breadcrumbs, related-page links, a language counterpart link, and a persistent download action.
- Every generated page must have at least two contextual incoming links from other HTML pages.
- Comparison, model, and release index pages link to every child page.
- The language control is a real anchor pointing to the exact counterpart URL. Theme switching remains a button.

## Language URL Migration

- `/` is the self-canonical English home page.
- `/zh/` is the self-canonical Simplified Chinese home page.
- Every English page links to its Chinese counterpart with `hreflang="zh-Hans"`.
- Every Chinese page links back with `hreflang="en"`.
- `x-default` points to the English counterpart.
- Existing `/?lang=zh` links are migration inputs only. The home-page script replaces them with `/zh/` while preserving the fragment when possible.
- The query-string state is not added to the sitemap and never becomes a canonical target.
- The site does not redirect based on IP, browser language, cookies, or `Accept-Language`.

This follows Google's recommendation to give each language version a distinct URL and to make language switching explicit.

## Content Model

### Fact Registry

Each reusable product fact has:

```text
id
status: released | development | planned
sinceVersion
lastVerified
sourceUrls[]
en: title, summary, detail
zh: title, summary, detail
pageAssignments[]
```

Requirements:

- `released` facts describe only capabilities available in the stated release.
- `development` facts identify main-branch or active engineering work that is not in the current release.
- `planned` facts are explicitly future work.
- A fact cannot build without both locales, at least one official source, and a verification date.
- The same fact ID drives visible HTML, relevant JSON-LD descriptions, release pages, comparison matrices, and LLM-oriented text.

### Competitor Registry

Each competitor snapshot has:

```text
id
name
verifiedVersion
lastVerified
officialSources[]
dimensions{}
limitations{}
neutralSummary{}
```

Source priority:

1. Official documentation and release notes.
2. The project's official repository and code.
3. Official product pages.

Do not use community comments, unsourced comparison blogs, search-result snippets, or inferred implementation details as facts. An inference may appear only when labeled as analysis and tied to the evidence used.

### Release Registry

Release pages contain:

- Version and release date.
- Concise highlights.
- Compatibility or upgrade notes.
- Known limitations.
- Status of work deferred to the next release.
- Official release and changelog links.

Adding a release should require one registry entry and no hand-editing of the page template, sitemap, or LLM indexes.

## Page Content Pattern

Every knowledge page follows an answer-first structure:

1. A 40-80 word direct answer to the page's primary question.
2. A facts-at-a-glance block.
3. Detailed technical explanation or capability matrix.
4. Copyable commands or API examples when relevant.
5. Released, development, and planned boundaries.
6. Official sources and last verified date.
7. Related pages and a relevant next action.

Page-specific goals:

- **Architecture:** explain the Swift in-process engine, unified memory, MLX execution, and the shared App/CLI/API core.
- **API compatibility:** document OpenAI, Anthropic, Ollama, MCP, embeddings, and rerank surfaces through endpoint matrices and examples.
- **Model guides:** help users choose an MLX model by memory, task, quantization, and vision requirements without presenting unstable benchmark claims as universal truth.
- **Comparison pages:** compare engine language, runtime shape, UI surfaces, model format, API surface, and target user. Each page states compared versions and verification dates.
- **Release pages:** provide a search-friendly, readable release summary that links to the authoritative changelog and release artifact.
- **FAQ:** answer real installation, Gatekeeper, model-format, privacy, memory, API, and Python-runtime questions in visible HTML.

## Build-Time Source Layout

```text
site/
├── content/
│   ├── facts.mjs
│   ├── competitors.mjs
│   ├── releases.mjs
│   ├── pages.mjs
│   └── locales/
│       ├── en.mjs
│       └── zh-Hans.mjs
├── templates/
│   ├── home.html
│   └── article.html
└── assets/

scripts/
├── build-public-site.mjs
└── test-public-site.mjs
```

The generator uses only Node built-ins.

- The home template preserves the current element structure and class names.
- Locale placeholders replace copy, accessibility labels, and metadata at build time.
- Page content is represented as typed structured blocks such as paragraph, heading, list, table, code, citation, and callout. Raw unescaped HTML is not accepted from fact values.
- Output order is deterministic.
- The generator writes only manifest-owned HTML, Markdown, XML, text, and social-image outputs. It does not delete unrelated static assets.
- Missing translations, missing sources, invalid dates, duplicate routes, duplicate canonicals, unsupported block types, or unresolved placeholders fail the build.

## Visual System

The content hub extends the current site rather than introducing a documentation product with a different brand.

- Reuse the warm-paper and near-black themes, system type, blue technical accents, lime or green status accents, rounded geometry, and restrained grid lines.
- Preserve the home page's large editorial typography and engine scrollytelling.
- Knowledge pages prioritize reading: shorter hero regions, stable text widths, clear tables, code blocks, citations, and related links.
- Avoid sticky sidebars that reduce reading width on smaller laptops.
- Comparison tables become dimension-by-dimension cards below the mobile breakpoint; no required horizontal scrolling.
- Reuse existing engine illustrations where they support the topic. Do not generate new images solely to decorate every article.
- Every page works in light and dark themes, without JavaScript, and with reduced motion.

## Metadata and Canonical Rules

Every generated HTML page includes:

- A unique locale-specific title in the same language and script as the main content.
- A unique locale-specific meta description.
- A self-referencing canonical.
- Reciprocal English, `zh-Hans`, and `x-default` alternates.
- Locale-correct Open Graph title, description, URL, locale, and image alt text.
- A `summary_large_image` Twitter card.
- A crawlable 1200 x 630 PNG social image.
- A Markdown alternate link for the page's generated Markdown representation.
- `dateModified` only when it is truthful and visible as a verification date.

The sitemap includes every canonical HTML route, truthful `lastmod` values, and reciprocal language alternates. It omits `changefreq` and `priority`, which do not provide durable value for this site.

## Structured Data

Structured data mirrors visible content and never adds claims that are absent from the page.

- Home pages: `WebSite` and `SoftwareApplication` entities.
- Architecture, API, model, comparison, and release pages: `TechArticle` plus `BreadcrumbList`.
- Page entities use `inLanguage`, `headline`, `description`, `dateModified`, `mainEntityOfPage`, and `about` where applicable.
- Comparison pages remain editorial technical articles. They do not use review ratings, product scores, or winner badges.
- FAQ questions and answers remain visible and machine-readable. `FAQPage` semantics may be emitted only when valid, but the design makes no Google FAQ rich-result claim because Google removed that feature in 2026.
- The software entity includes truthful name, operating system, application category, price, repository, download URL, version, license, runtime platform, and screenshot properties.
- Do not invent an aggregate rating or review to satisfy Google's software-app rich-result requirements.

Validate every JSON-LD block as JSON and verify that key claims are present in visible text.

## Crawler Policy and GEO Files

`robots.txt` becomes a small, maintainable allow policy:

- Default crawling is allowed.
- OAI-SearchBot is explicitly allowed for ChatGPT search discovery.
- GPTBot remains independently allowed under the project's current open policy, but its training purpose is not conflated with search discovery.
- The sitemap location is declared once.
- Obsolete or unverified crawler-specific entries are removed unless a documented product requirement exists.

GEO files:

- `/llms.txt` is a concise English project overview and link index.
- `/llms-full.txt` contains the complete English fact set and article summaries.
- `/zh/llms.txt` and `/zh/llms-full.txt` provide equivalent Chinese content.
- Each knowledge page has a Markdown representation under `/content/{locale}/`.
- Markdown and LLM-oriented files include canonical source URLs, verification dates, and the same release-status labels as HTML.

These files are supplemental discovery surfaces. They do not replace crawlable HTML, sitemaps, internal links, or standard metadata, and the project will not claim that `llms.txt` guarantees AI ranking or citation.

## Performance and Accessibility

- The primary answer and page heading are present in the initial HTML response.
- Content pages use the existing system-font stack and shared CSS; they do not add a framework bundle.
- JavaScript is deferred and limited to theme, copy controls, and optional interaction enhancement.
- Images declare width and height and use appropriate eager or lazy loading.
- Breadcrumbs, headings, tables, code, citations, and navigation use semantic HTML.
- Tables include captions and header associations before they become mobile cards.
- Language links expose meaningful accessible names.
- Contrast is verified in both themes, including status badges and secondary copy.
- Focus order and skip navigation remain correct.

## Failure Behavior

- Build validation fails before writing an invalid route set.
- A missing optional illustration does not remove the article text or metadata.
- JavaScript failure leaves every page readable, navigable, and in its canonical language.
- A missing competitor source prevents that comparison fact from publishing.
- A stale verification date is surfaced by the test suite rather than silently accepted.
- Unknown content blocks and unescaped values fail generation.
- The build performs no network fetches. External facts are reviewed and committed as source data.

## Release Maintenance Workflow

For each macMLX release:

1. Update the current version, release date, release facts, and official release links.
2. Reclassify facts whose status changed.
3. Review affected comparison and model guidance.
4. Update verification dates only for facts that were actually checked.
5. Build the site and inspect the generated diff.
6. Run automated and browser verification.
7. Publish the static output.
8. Submit or refresh the sitemap in search-engine consoles as a separate authorized deployment step.

Search Console, Bing Webmaster Tools, and IndexNow submissions are external actions. They are not performed by the local implementation without separate authorization and deployment credentials.

## Verification

### Generator and Content Tests

- A clean build produces every declared English and Chinese route.
- Repeating the build without source changes produces byte-identical text outputs.
- No unresolved template token remains.
- Every page has a unique title, description, canonical, and route.
- Every route has an exact language counterpart and reciprocal hreflang links.
- Every page contains one primary heading and a language-correct `<html lang>` value.
- Every fact has two locales, a valid status, a source, and a verification date.
- Competitor facts include a compared version or dated official snapshot.
- No generated page contains stale prohibited claims such as `only native GUI`.
- Released, development, and planned claims remain visibly distinct.

### Search-Surface Tests

- Parse every JSON-LD block.
- Confirm structured-data claims appear in visible content.
- Confirm the sitemap contains every canonical route once and only once.
- Confirm `robots.txt` allows canonical pages and points to the sitemap.
- Confirm HTML, Markdown, `llms.txt`, and `llms-full.txt` agree on current version and status labels.
- Crawl all internal HTML links locally and fail on missing assets or pages.
- Confirm the deprecated `?lang=zh` URL is absent from canonical, hreflang, sitemap, and generated navigation.
- Confirm all social image targets exist and are PNG files with declared dimensions.

### Browser and Visual Tests

- Preserve the current home-page desktop and mobile visual baselines.
- Test at least one representative knowledge page and one comparison page at desktop and mobile widths.
- Test English and Chinese, light and dark themes, no JavaScript, and reduced motion.
- Confirm comparison matrices do not cause horizontal overflow.
- Confirm language switching reaches the exact counterpart route.
- Confirm browser console errors and failed formal resources are empty.
- Run `visual-verdict` for every visual iteration and require a score of at least 90.

### External Validation After Deployment

- Google Rich Results Test for supported structured-data types.
- Google Search Console URL Inspection for representative English and Chinese pages.
- Sitemap submission to Google and Bing.
- Optional IndexNow setup only after a host key and deployment workflow are authorized.

## Rollout

The implementation may be divided into reviewable tasks, but the approved deliverable includes the complete route set.

1. Build system, content registries, and validation.
2. Static English and Chinese home pages with URL migration.
3. Core architecture, API, FAQ, model, comparison, and release pages.
4. Markdown alternates, LLM indexes, sitemap, robots, and PNG social images.
5. Automated crawl verification and browser visual QA.

Do not publish a partial sitemap or language graph that references routes not present in the same output.

## Non-Goals

- Do not convert the website into a client-rendered SPA.
- Do not add a CMS, framework, analytics platform, cookie banner, or runtime localization service.
- Do not add third-party build dependencies.
- Do not rewrite the native app or its in-app documentation in this scope.
- Do not scrape competitor sites during the build.
- Do not create aggressive winner/loser scoring or unverifiable performance comparisons.
- Do not fabricate reviews, ratings, adoption numbers, benchmarks, or compatibility.
- Do not claim that structured data, `llms.txt`, crawler access, or content formatting guarantees rankings or AI citations.
- Do not submit URLs, verification keys, or search-console changes without separate authorization.

## Primary Guidance Sources

- [Google: Managing multi-regional and multilingual sites](https://developers.google.com/search/docs/advanced/crawling/managing-multi-regional-sites)
- [Google: JavaScript SEO basics](https://developers.google.com/search/docs/crawling-indexing/javascript/javascript-seo-basics)
- [Google: Software application structured data](https://developers.google.com/search/docs/appearance/structured-data/software-app)
- [Google: General structured data guidelines](https://developers.google.com/search/docs/appearance/structured-data/sd-policies)
- [Google: Search documentation updates](https://developers.google.com/search/updates#removing-faq-rich-result)
- [OpenAI: Overview of OpenAI crawlers](https://developers.openai.com/api/docs/bots)
- [Schema.org: SoftwareApplication](https://schema.org/SoftwareApplication)
- [IndexNow protocol documentation](https://www.indexnow.org/documentation)
