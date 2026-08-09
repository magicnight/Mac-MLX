# Release Rules
<!-- last-analyzed: 2026-07-08T12:30:00Z -->

## Version Sources
- **git tag `vX.Y.Z`** — authoritative. Injected at build time by scripts via
  `GITHUB_REF_NAME`: `scripts/build.sh` sets app `MARKETING_VERSION="${TAG#v}"` +
  `CURRENT_PROJECT_VERSION=$GITHUB_RUN_NUMBER`; `package-dmg.sh` / `package-cli.sh` /
  `render-formula.sh` name artifacts from the tag. (pbxproj's checked-in
  `MARKETING_VERSION = 1.0` is a placeholder — do NOT bump it.)
- `MacMLXCore/Sources/MacMLXCore/MacMLXCore.swift` → `public static let version` —
  compiled into `macmlx --version` (CLI) via `MacmlxCommand.coreVersion`. Must be
  bumped manually each release (was stale at "0.1.0" until v0.5.3).
- `CHANGELOG.md` — Keep-a-Changelog; rename `[Unreleased]` → `[X.Y.Z] - date`,
  add fresh empty `[Unreleased]` above.

## Release Trigger
Push annotated tag `v*.*.*` → `.github/workflows/release.yml` (job "Build, Sign &
Release", macos-26, Xcode 26.4.1 pinned).

## Test Gate
release.yml has NO test step — tests run only in ci.yml on push/PR. RULE: verify
main CI green (`gh run list --branch main -L1`) before tagging.

## Registry / Distribution
GitHub Release with assets: signed DMG (create-dmg), `macmlx-vX.Y.Z-arm64.tar.gz`
CLI tarball, rendered `dist/macmlx.rb` Homebrew formula (for tap
magicnight/homebrew-mac-mlx, manual/automated pull). Sparkle EdDSA appcast:
signs DMG with secret `SPARKLE_PRIVATE_KEY` (skips gracefully if unset) and
**commits appcast.xml back to main** — pull main after the workflow finishes.

## Release Notes Strategy
Keep a Changelog (`CHANGELOG.md`), semver. GitHub Release body from the
CHANGELOG section (workflow creates the release; verify body afterwards).

## CI Workflow Files
- `.github/workflows/release.yml` (tag-triggered release)
- `.github/workflows/ci.yml` (push/PR tests; docs paths-ignored)
- Helper scripts: `scripts/build.sh`, `scripts/package-dmg.sh`,
  `scripts/package-cli.sh`, `scripts/render-formula.sh`

## First-Time Setup Gaps
none (workflow + tags + .gitignore all in place). Historical note: v0.4.x was
never tagged — tags jump v0.3.7 → v0.5.0.
