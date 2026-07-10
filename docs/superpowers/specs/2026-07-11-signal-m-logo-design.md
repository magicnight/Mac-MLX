# Signal M Logo Design

## Goal

Replace the current macMLX website mark with a distinctive, deterministic logo that matches the site's editorial grid, high-contrast typography, and restrained Apple Silicon aesthetic. The mark must remain recognizable from the navigation bar down to a 16-pixel favicon.

## Selected direction

The approved concept is **Signal M**, selected after two visual comparison rounds:

1. Neural M was chosen over Unified Core and Metal Ribbon.
2. Signal M was chosen over Continuous M and Core Fold.

Signal M is a bold, monoline `M` inside a rounded square. Its center activation node represents one token entering the shared inference core. A small green signal node represents available, local execution.

## Geometry

The canonical artwork uses a 128 by 128 coordinate system:

- Background squircle: inset 4 units, 34-unit corner radius.
- `M`: two vertical stems joined through a centered valley; 14-unit rounded stroke with rounded joins.
- Activation node: centered on the `M` valley, 8.5-unit radius.
- Signal node: aligned to the upper-right `M` shoulder, 6-unit radius.

The 16-pixel favicon variant removes the green signal node and slightly increases the `M` stroke. This is an optical-size adjustment, not a different logo.

## Color system

- Squircle: warm paper `#F3F1EA`.
- `M`: near-black `#111311`.
- Activation node: cobalt blue `#7196FF`.
- Signal node: live green `#89E67A`.

The mark does not invert between light and dark themes. Its paper squircle creates a stable silhouette in both modes. No gradients, transparency effects, shadows, glass, or photographic texture are part of the canonical asset.

## Assets

Create one canonical SVG source at `site/assets/brand/macmlx-mark.svg`. Derive and track:

- `site/assets/brand/favicon.svg`, using the optical-size simplification.
- `site/assets/brand/apple-touch-icon.png`, 180 by 180 pixels.
- `site/assets/brand/icon-192.png` and `icon-512.png` for manifest-quality web icons.

The SVG files are the design source. PNG files are deterministic rasterizations made from those SVGs, not generated interpretations.

## Website integration

- Replace the current inline decorative mark in the home and article navigation with the canonical SVG asset.
- Preserve the existing text wordmark `macMLX` and its typography.
- Add the SVG favicon and Apple touch icon to every generated HTML document through centralized metadata rendering.
- Add a minimal web app manifest referencing the 192- and 512-pixel icons.
- Replace the old logo in both localized social cards, then regenerate their tracked PNG output.
- Use a localized accessible label on linked logo instances; decorative duplicates use an empty alternative.
- Do not change hero layout, navigation spacing, body typography, or article content beyond adjustments required by the new mark dimensions.

## Build and validation

The build treats every logo asset as an explicit tracked input and generated output. Tests verify:

- exact SVG view boxes, palette, and absence of scripts or external references;
- existence and dimensions of every raster icon;
- favicon, touch icon, and manifest links on all 26 localized HTML pages;
- navigation logo usage on home and article templates;
- English and Chinese social card regeneration;
- deterministic builds and complete asset manifests;
- visible focus and contrast in light and dark themes;
- recognizable rendering at 16, 32, 64, and navigation sizes.

Browser QA covers desktop and mobile navigation in both themes, both locales, and the installed production favicon.

## Release coordination

Implement the logo and the approved `www.macmlx.app` permanent redirect in separate, testable commits. Run their focused tests independently, then the complete site suite. Deploy the updated apex Static Assets Worker first and verify it. Next delete only the recorded DNS-only `www` CNAME, deploy the isolated redirect Worker, and verify both hosts. This produces one coordinated release window while preserving independent rollback paths.

## Non-goals

- No app icon redesign outside website-distributed assets.
- No generated bitmap logo as the source of truth.
- No change to the `macMLX` spelling or wordmark font.
- No animated logo, alternate mascot, tagline, or expanded brand identity system.
