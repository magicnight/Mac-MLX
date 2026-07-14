# v0.6.2 Website Content Refresh Design

**Date:** 2026-07-15

**Status:** Approved for implementation planning

**Surface:** `site/` generated marketing and documentation website, deployed at `macmlx.app`

## Purpose

Refresh the public website from the audited v0.5.3 baseline to the current v0.6.2 release without redesigning the established visual system. The update must make the v0.6 capability wave discoverable while preserving the site's existing evidence-backed distinction between released, limited, theoretical, and planned work.

The immutable `v0.6.2` tag, its changelog, source files, model-support table, and release artifact are the authority for shipped claims. The website may explain cumulative v0.6 capabilities, but it must not attribute future cache virtualization or sampling controls to v0.6.2.

## Selected Approach

Use a full registry-driven content refresh rather than a version-number-only edit or a visual redesign.

- Update the shared project, fact, release, FAQ, and page registries.
- Let the existing renderer propagate those facts into visible pages, metadata, structured data, feeds, `llms.txt`, and sitemap output.
- Add only the content structures needed to explain current capabilities and models.
- Preserve the existing layout, Signal M identity, bilingual behavior, color themes, and engine scroll story.

A minimal version bump was rejected because it would leave released v0.6 functionality labeled as development or planned. A new visual campaign was rejected because the current site already has a strong engine narrative and the request is to reflect development progress, not reopen the visual direction.

## Release Baseline and Evidence Rules

Set the public current version to `0.6.2`, the release date to `2026-07-11`, and the content verification date to `2026-07-15`.

Evidence must follow these rules:

1. Shipped claims link to immutable `v0.6.2` tag paths or the v0.6.2 release.
2. Performance numbers appear only when the tagged changelog or model-support table records a real-checkpoint result.
3. A parity-verified architecture without a runnable public checkpoint remains theoretical.
4. `main` links are reserved for clearly labeled development or planned work.
5. Existing historical v0.5.3 pages remain historical records and must not be silently rewritten as if the capabilities shipped in that version.

## Capability Status Changes

Promote the following facts from development to released, with `sinceVersion: 0.6.0`:

- Agent tool loops for OpenAI Chat Completions and Anthropic Messages.
- Continuous batching for eligible dense-text concurrent requests, including the documented 2.5–3.2× four-client aggregate-throughput result and automatic serial fallback boundaries.
- Longest-common-prefix prompt-cache reuse backed by the RAM hot tier and SSD cold tier.
- Fixed prefill admission as part of the released continuous-batching scheduler, without presenting it as an adaptive memory controller.

Add released facts for:

- Structured output using `json_object` and the supported JSON Schema subset, including explicit rejection of unsupported schema keywords.
- Classic draft-model speculative decoding, GUI controls, acceptance telemetry, and graceful fallback for non-trimmable cache architectures.
- The v0.6 API compatibility pack: `logit_bias`, `logprobs`, `top_logprobs`, XTC, per-request LoRA adapters, KV-cache quantization, and tool pass-through.
- Hugging Face cache discovery and the associated GUI model workflow improvement where relevant to model-selection guidance.
- Per-model chat-template overrides introduced by v0.6.2, with the precedence order user file, built-in `model_type` override, then checkpoint template.

Preserve these as planned:

- Paged KV allocation.
- Block sharing.
- Copy-on-write cache branching.
- A unified adaptive memory guard spanning cache, model pool, and concurrency.
- User-facing generation top-k, min-p, presence/frequency/repetition penalties, and per-request seed.

Split the old combined sampling roadmap fact. XTC and KV-cache quantization are released in v0.6.0; top-k, min-p, the listed penalties, and per-request seed remain planned. Internal MoE expert-routing top-k must not be confused with user sampling top-k.

## Model Support Content

Add a concise v0.6 model-wave section to the model pages and current release record. The status and wording must match `docs/model-support.md`.

### Tested

- Seed-OSS-36B (`seed_oss`): tested with full parity coverage and an 18.2 tok/s real-checkpoint smoke on the 4-bit checkpoint; uses a built-in template override.
- Hunyuan V1 Dense (`hunyuan_v1_dense`): tested across the 0.5B–7B family; the tagged result is 80.3 tok/s on the 1.8B 4-bit checkpoint.
- Cohere Command R7B (`cohere2`): tested; the tagged result is 21.7 tok/s on the 7B 4-bit checkpoint; uses a built-in template override for the unsupported stock template branch.
- MiniCPM3-4B (`minicpm3`): tested; the tagged result is 18.7 tok/s on the 4-bit checkpoint.

### Theoretical

- InternLM3-8B (`internlm3`): parity-verified but theoretical because published checkpoints currently provide `tokenizer.model` rather than the `tokenizer.json` required by the Swift tokenizer stack.

The page may also summarize the earlier v0.6.0 architecture wave—Qwen3.6 and Mellum2 tested; Solar-Open and GLM-5.1 theoretical—when it helps users understand cumulative current support. It must retain checkpoint-level caveats and avoid implying universal compatibility for every quantization or processor variant.

## Page-Level Information Architecture

### Home

- Update the current-version label and release call to action.
- Refresh the shipped-capability summary to mention batching, LCP reuse, structured output, speculative decoding, tool loops, and the Track G model wave.
- Keep the hero concise and retain the current visual hierarchy.
- Update the engine-story status language so released v0.6 features no longer read as post-tag development.

### Architecture

- Explain released continuous batching and its eligibility/fallback boundary.
- Explain LCP reuse as the released prompt-cache behavior while keeping paged KV, block sharing, and CoW planned.
- Keep the fixed prefill throttle separate from the future adaptive memory guard.
- Add speculative decoding and structured-output execution boundaries where they fit the existing flow.

### API Compatibility

- Update the matrix for OpenAI and Anthropic tool loops.
- Add structured output, logit bias, logprobs/top-logprobs, XTC, per-request adapters, and KV-cache quantization.
- Preserve explicit endpoint and parameter-combination limitations; do not claim full provider-wide compatibility.
- Reflect v0.6.1 hardening where it materially defines behavior, including explicit 400 responses for unsupported VLM/structured-output and tools/structured-output combinations.

### Models and Model Guidance

- Add the v0.6.2 Track G tested/theoretical entries.
- Add chat-template override guidance for affected checkpoints.
- Keep memory, quantization, tokenizer, and checkpoint-variant cautions visible.

### FAQ

- Update the API, model, MoE, and roadmap answers to the v0.6.2 baseline.
- Make released versus planned status readable in the answer itself, not only through source links.

### Releases

- Add a current v0.6.2 release page and card.
- Describe v0.6.2 specifically while allowing the release hub to summarize the cumulative v0.6 capability baseline.
- Retain the v0.5.3 page as a historical record.
- Link to the immutable release, tagged changelog, model-support table, and relevant tagged source files.

### Comparisons

- Update macMLX's side of comparison metadata from v0.5.3 to v0.6.2.
- Change only claims affected by the new release; do not refresh competitor versions or claims without new official evidence.
- Preserve neutral language and dated snapshots.

## SEO and GEO Output

The registry refresh must propagate consistently to:

- Page titles and descriptions where a stale v0.5.3 reference changes search intent.
- Open Graph and social descriptions.
- Software application and release structured data.
- Breadcrumbs and release hierarchy.
- FAQ structured data generated from visible FAQ copy.
- Canonical and alternate-language pages.
- Sitemap `lastmod` values.
- `llms.txt` and any long-form machine-readable site summary.

Machine-readable descriptions should answer the core entity questions directly: what macMLX is, which Macs it supports, how its Swift in-process engine differs, which interfaces it exposes, what v0.6.2 ships, and which features remain planned. Structured data must never contain a stronger claim than the visible page.

## Bilingual and Accessibility Requirements

- Every new visible English string receives a natural Simplified Chinese counterpart.
- Technical identifiers remain exact while explanatory copy is localized.
- Existing English/Chinese structural parity checks remain green.
- Status terms use the existing released, limited, theoretical, development, and planned vocabulary consistently.
- Added tables or cards must retain semantic headings, readable source links, keyboard accessibility, and light/dark contrast.

## Implementation Boundaries

- Do not add dependencies.
- Do not redesign unrelated sections or regenerate existing imagery.
- Do not change the Signal M logo or theme system.
- Do not convert historical pages into undated evergreen claims.
- Do not advertise the SwiftLM or Python compatibility engines as the default path.
- Do not claim Neural Engine inference, universal MoE support, universal OpenAI/Anthropic compatibility, or a shipped paged-KV system.

## Verification

### Content and Build

- Extend registry validation tests first for the v0.6.2 release and changed fact statuses.
- Test that every source URL is immutable where the corresponding fact is released.
- Test that planned cache and sampling facts remain planned.
- Test that current release links resolve to `/releases/v0-6-2/` while the v0.5.3 page remains generated.
- Run the complete site test suite and production build.
- Search generated output for stale current-baseline statements such as “current release v0.5.3” and for accidental claims that planned features shipped.

### Browser and Visual QA

Verify at minimum:

- Home, architecture, API, models, FAQ, releases, and v0.6.2 release pages.
- English and Simplified Chinese.
- Light and dark themes.
- Desktop and mobile widths.
- Direct navigation to `#engine` and the updated engine-story statuses.
- No overflow, missing sources, broken internal links, console errors, or structured-data mismatch.

Use the visual-verdict workflow for any iteration that changes rendered layout. Pure copy updates that do not alter layout still require representative browser screenshots and manual readability checks.

### Deployment

After local verification, publish the generated static output using the repository's Bun-managed Wrangler release workflow. Verify:

- The deployment reports success.
- `https://macmlx.app/` serves the v0.6.2 content.
- `https://www.macmlx.app/` follows the configured canonical redirect.
- Representative English and Chinese pages return successful responses.
- Security and cache headers remain present.
- The sitemap, robots file, `llms.txt`, and structured data reflect the new release.

Deployment must not rewrite DNS or storage configuration unless verification identifies a concrete configuration defect.

## Success Criteria

The work is complete when a visitor and a search or answer engine can correctly determine that v0.6.2 is current, understand the cumulative v0.6 shipped capabilities and Track G model support, distinguish theoretical and planned work, follow immutable evidence links, and receive the same factual story in English and Chinese without a visual regression.
