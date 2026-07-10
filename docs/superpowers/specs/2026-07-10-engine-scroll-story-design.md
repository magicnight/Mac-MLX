# Engine Scroll Story Design

**Date:** 2026-07-10

**Status:** Approved for implementation planning

**Surface:** `public/` marketing website

## Purpose

Add a high-impact, technically honest explanation of how macMLX runs local LLMs on Apple Silicon. The section should help newcomers understand the system at a glance while letting developers inspect MoE routing, KV-memory evolution, runtime admission, generation controls, and Mac-specific advantages.

The new section must preserve the existing site's restrained warm-ivory, charcoal, electric-blue, and lime visual language. It must not turn the home page into long-form documentation or imply that planned engine work has already shipped.

## Placement and Navigation

Insert a new section after `#features` (What ships today) and before `#progress` (Development progress).

- Section id: `engine`
- English navigation label: `Engine`
- Chinese navigation label: `引擎`
- English section title: `Inside the engine.`
- Chinese section title: `走进引擎内部。`

This order establishes the intended reading sequence:

1. What macMLX is.
2. What ships today.
3. How the engine works and where it is heading.
4. What is released versus actively being built.
5. How to get started.

## Information Architecture

The section is a five-chapter inference journey. Each chapter contains one conclusion-led headline, up to two short explanatory paragraphs, three to five technical tags, one explicit capability-status badge, and one primary image.

### Chapter 1 — Mac Foundation

Explain unified memory, CPU orchestration, integrated-GPU parallelism, and reduced memory copying.

- Status: `Platform advantage / 平台优势`
- Tags: `Unified Memory`, `CPU + GPU`, `MLX`, `In-process`
- Claim boundary: describe Apple Silicon and MLX execution characteristics, not Neural Engine inference.

### Chapter 2 — Sparse Intelligence

Explain how an MoE router activates a small subset of experts per token while shared experts remain available.

- Status: `Available / 已支持`
- Concrete proof point: the pure-Swift DeepSeek V3.2 MoE implementation and parity tests.
- Tags: `Router`, `Active Experts`, `Shared Expert`, `Top-k Routing`
- Claim boundary: do not imply that every MoE architecture is supported. Gemma 4 MoE remains explicitly unsupported by the current upstream path.

### Chapter 3 — Shareable Memory

Explain current tiered prompt-cache behavior and the next cache-virtualization steps.

- Current: hot RAM cache, exact-key SSD cold tier, promotion and demotion.
- Next: paged KV allocation, block sharing, copy-on-write branching, and wider sharing across concurrent requests.
- Status: `Current + next / 当前 + 下一步`
- Tags: `Paged KV`, `Block Sharing`, `CoW`, `SSD Cold Tier`
- Claim boundary: only RAM plus SSD cold-tier behavior may be described as implemented. Paged KV, block sharing, and CoW must be labeled as planned evolution.

### Chapter 4 — Adaptive Runtime

Explain how the runtime controls admission and memory pressure rather than accepting unlimited work.

- Current: bounded prefill admission via `prefillBatchSize`, memory-aware model-pool limits, eviction, and memory probes.
- Next: an adaptive memory guard that feeds live pressure back into cache, model-pool, and concurrency decisions.
- Status: `On main + evolving / 主线已有 + 持续演进`
- Tags: `Prefill Throttle`, `Memory Probe`, `Admission`, `Eviction`
- Claim boundary: do not name a standalone adaptive-memory-guard component as shipped until one exists in code.

### Chapter 5 — Controlled Generation

Explain the relationship between sampling controls, determinism, output character, speed, and memory.

- Current: temperature and top-p.
- Planned: top-k, min-p, presence/frequency/repetition penalties, per-request seed, and user-facing KV-cache quantization controls.
- Status: `Core controls + planned / 核心参数 + 规划中`
- Tags: `top-k`, `min-p`, `Penalties`, `Seed`, `KV Quant`
- Claim boundary: generation `seed` is not currently a per-request field; top-k, min-p, and repetition penalty are explicitly deferred in the parameter store.

## Desktop Interaction

Use cinematic sticky scrollytelling without scroll hijacking or mandatory scroll snap.

- The section has a two-column layout.
- The left column contains five chapter steps, each approximately 70–85 viewport-height units tall.
- The right visual stage is sticky and remains centered in the viewport while the five steps pass it.
- `IntersectionObserver` determines the active chapter from viewport-center proximity.
- The active chapter updates a section-level `data-engine-step` attribute or equivalent single source of truth.
- The visual stage switches images with opacity crossfades, restrained scale changes, clip reveals, and HTML overlay highlights.
- A five-segment progress rail identifies the active chapter.
- The interaction must never cancel wheel events, rewrite scroll position, or trap keyboard navigation.

Use only CSS and the existing JavaScript file. Do not add GSAP, a canvas renderer, or another runtime dependency.

## Mobile and Reduced-Motion Behavior

At the mobile breakpoint, disable the sticky stage and render the chapters as a normal single-column document.

- Each chapter contains its own image, copy, tags, and status.
- Images receive only a small opacity-and-scale reveal.
- No horizontal carousel or required swipe gesture.
- All technical content remains available without interaction.
- Under `prefers-reduced-motion: reduce`, render all chapters and images immediately with no crossfade, scale, or clip animation.
- Without JavaScript, the document order remains correct and every image remains visible.

## Image Production

Generate five new website assets with the explicit `gpt-image-2` CLI/API path at high quality.

1. `mac-silicon-foundation.webp`
2. `moe-routing.webp`
3. `paged-kv-memory.webp`
4. `adaptive-runtime.webp`
5. `generation-controls.webp`

Production requirements:

- 2048 × 1152 master WebP files.
- Shared palette: warm ivory `#f3f1ea`, near-black `#111311`, electric blue `#3f7fe8`, and small lime `#b9e769` accents.
- Cinematic 3D technical illustration with editorial clarity.
- No baked-in labels, readable text, logos, watermarks, cloud infrastructure, Python symbolism, discrete gaming GPUs, or inaccurate Neural Engine claims.
- Generate smaller web derivatives after visual approval.
- Target no more than approximately 600 KB per delivered web image and approximately 3 MB for all five web derivatives combined.

All labels and callouts must be HTML so they remain bilingual, sharp, indexable, and accessible.

## Asset Loading and Failure Behavior

- The first chapter image may load eagerly because it becomes visible near section entry.
- Chapters two through five use `loading="lazy"` and `decoding="async"`.
- Every image declares intrinsic width and height to prevent layout shift.
- Use a neutral styled figure background while image decoding is pending.
- Treat the generated images as supplementary illustrations: use empty image alt text and let the adjacent bilingual chapter heading, copy, tags, and status carry the complete accessible explanation.
- An image load failure must not hide the copy, status, tags, progress state, or subsequent chapters.
- JavaScript enhancement errors must leave the static document readable.

## Bilingual Content

Continue the current `data-en` / `data-zh` translation pattern.

- Every visible heading, paragraph, status badge, progress label, and accessibility label introduced by the section must have matching English and Chinese values.
- Technical tokens such as `MoE`, `Paged KV`, `CoW`, `top-k`, and `min-p` remain unchanged in both languages.
- Chinese copy should explain the term in natural language rather than transliterating it.
- The English and Chinese node counts must remain equal in the static site regression test.

## Accessibility

- Use semantic `section`, ordered-list or article chapter structure, headings, figures, and progress labels.
- Sticky visuals are supplementary; the text sequence contains the complete explanation.
- Active-state changes must not move keyboard focus or announce noisy live-region updates.
- Maintain text and status contrast in light and dark themes.
- Decorative visual overlays use `aria-hidden="true"`.
- Keyboard and screen-reader order follows the five chapter order.

## Verification

### Automated Checks

Extend `scripts/test-public-site.mjs` before implementation to cover:

- `#engine` exists between `#features` and `#progress`.
- Exactly five engine chapters exist.
- Each chapter contains a status, image, tags, English copy, and Chinese copy.
- New English and Chinese node counts remain equal.
- All five asset paths exist.
- Chapter one is eager and chapters two through five are lazy.
- Intrinsic image dimensions are present.
- Reduced-motion and mobile static-layout rules exist.
- No Homebrew command is reintroduced.

Run JavaScript syntax validation and SVG parsing alongside the site test.

### Browser Checks

Verify at minimum:

- 1440 × 900 desktop, light and dark themes.
- 390 × 844 mobile, light and dark themes.
- English and Chinese.
- Slow scrolling, fast scrolling, jumping directly to `#engine`, scrolling backward, and leaving/re-entering the section.
- No horizontal overflow, sticky overlap, layout shift, console error, or blank chapter.
- Reduced-motion renders a complete static section.
- Image failure leaves readable content.

Run a visual-verdict comparison on every visual iteration and require a score of at least 90 before proceeding.

## Non-Goals

- Do not turn the home page into API documentation.
- Do not add interactive parameter controls or a live inference simulator.
- Do not claim planned KV virtualization or sampling parameters are currently shipped.
- Do not use scroll-jacking, WebGL, canvas animation, video backgrounds, or new JavaScript dependencies.
- Do not redesign unrelated existing sections during this work.
