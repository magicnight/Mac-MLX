# Distribution

## Build Requirements

- Xcode 15+
- macOS 14.0+ build machine (Apple Silicon)
- No Apple Developer account required for development
- Sparkle framework for auto-update

## Sparkle Auto-Update Setup

Sparkle is the standard macOS auto-update framework used by hundreds of apps.

### SPM Dependency

```swift
// Package.swift or Xcode SPM
.package(url: "https://github.com/sparkle-project/Sparkle", from: "2.0.0")
```

### AppDelegate Integration

```swift
import Sparkle

final class AppDelegate: NSObject, NSApplicationDelegate {
    // Sparkle updater — must be stored as property
    let updaterController = SPUStandardUpdaterController(
        startingUpdater: true,
        updaterDelegate: nil,
        userDriverDelegate: nil
    )

    // Expose to SwiftUI for "Check for Updates" menu item
    var updater: SPUUpdater { updaterController.updater }
}
```

### SwiftUI Menu Item

```swift
// In App commands
CommandGroup(after: .appInfo) {
    Button("Check for Updates...") {
        appDelegate.updater.checkForUpdates()
    }
    .disabled(!appDelegate.updater.canCheckForUpdates)
}
```

### Appcast XML

Maintained at `appcast.xml` in repo root.
GitHub Pages serves it at:
`https://raw.githubusercontent.com/magicnight/mac-mlx/main/appcast.xml`

```xml
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>macMLX</title>
    <link>https://github.com/magicnight/mac-mlx</link>
    <description>macMLX releases</description>
    <language>en</language>
    <item>
      <title>Version 0.1.0</title>
      <sparkle:version>1</sparkle:version>
      <sparkle:shortVersionString>0.1.0</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>14.0</sparkle:minimumSystemVersion>
      <pubDate>Mon, 16 Apr 2026 00:00:00 +0000</pubDate>
      <enclosure
        url="https://github.com/magicnight/mac-mlx/releases/download/v0.1.0/macMLX-v0.1.0.dmg"
        sparkle:edSignature="SIGNATURE_HERE"
        length="FILE_SIZE_BYTES"
        type="application/octet-stream"
      />
      <sparkle:releaseNotesLink>
        https://github.com/magicnight/mac-mlx/releases/tag/v0.1.0
      </sparkle:releaseNotesLink>
    </item>
  </channel>
</rss>
```

### EdDSA Signing Key

Sparkle 2.x uses EdDSA signatures (not DSA).
Generate once and store private key securely:

```bash
# Generate key pair (run once, keep private key secret)
./bin/generate_keys

# Sign DMG after building
./bin/sign_update macMLX-v0.1.0.dmg privatekey.pem
# Outputs: sparkle:edSignature value to paste into appcast.xml
```

**NEVER commit the private key to the repo.**
Store in GitHub Secrets as `SPARKLE_PRIVATE_KEY`.

### CI: Auto-update appcast.xml on release

```yaml
# In release.yml, after DMG is built:
- name: Sign DMG for Sparkle
  run: |
    echo "${{ secrets.SPARKLE_PRIVATE_KEY }}" > sparkle_private_key.pem
    SIGNATURE=$(./Sparkle/bin/sign_update dist/*.dmg sparkle_private_key.pem)
    FILE_SIZE=$(stat -f%z dist/*.dmg)
    echo "SPARKLE_SIGNATURE=$SIGNATURE" >> $GITHUB_ENV
    echo "DMG_SIZE=$FILE_SIZE" >> $GITHUB_ENV

- name: Update appcast.xml
  run: |
    VERSION="${GITHUB_REF_NAME}"
    python3 scripts/update_appcast.py \
      --version "$VERSION" \
      --signature "$SPARKLE_SIGNATURE" \
      --size "$DMG_SIZE"
    git config user.name "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"
    git add appcast.xml
    git commit -m "chore: update appcast for $VERSION"
    git push
```

## DMG Packaging

`scripts/package-dmg.sh`:

```bash
#!/bin/bash
set -e

APP_NAME="macMLX"
VERSION="${GITHUB_REF_NAME:-$(git describe --tags --abbrev=0 2>/dev/null || echo 'v0.0.0-dev')}"
DMG_NAME="${APP_NAME}-${VERSION}.dmg"

xcodebuild -scheme $APP_NAME \
           -configuration Release \
           -archivePath build/${APP_NAME}.xcarchive \
           archive

xcodebuild -exportArchive \
           -archivePath build/${APP_NAME}.xcarchive \
           -exportPath build/export \
           -exportOptionsPlist scripts/ExportOptions.plist

create-dmg \
  --volname "$APP_NAME $VERSION" \
  --window-size 600 400 \
  --icon-size 128 \
  --icon "${APP_NAME}.app" 150 185 \
  --app-drop-link 450 185 \
  --background scripts/dmg-background.png \
  "dist/${DMG_NAME}" \
  "build/export/${APP_NAME}.app"

echo "Built: dist/${DMG_NAME}"
```

## GitHub Actions CI

### release.yml (tag push)

Triggers on: `v*.*.*` tags

1. Build + archive
2. Package DMG
3. Sign for Sparkle
4. Compute SHA256
5. Update appcast.xml
6. Create GitHub Release with DMG

### ci.yml (push to main / PR)

1. Swift build + test
2. SwiftLint

## Without Developer Account

DMG is unsigned. First-launch workaround for users:

```
Right-click macMLX.app → Open → Open
```

Add to README Installation section.

## Versioning

Semantic versioning: `MAJOR.MINOR.PATCH`
Tag format: `v0.1.0`

CFBundleVersion = `$GITHUB_RUN_NUMBER` (integer, auto-increments)
CFBundleShortVersionString = tag without `v` prefix (e.g. `0.1.0`)

## Homebrew Tap (CLI)

The `macmlx` CLI ships through a Homebrew tap so developers can install
it with one command:

```bash
brew tap magicnight/mac-mlx
brew install macmlx
```

The GUI app stays on the GitHub Releases DMG path — Homebrew is CLI-only
to avoid dragging cask packaging into the same pipeline.

### Pieces

| Where | What |
|-------|------|
| `Formula/macmlx.rb` (this repo) | Source-of-truth template with `@@VERSION@@`, `@@URL@@`, `@@SHA256@@` placeholders. |
| `scripts/package-cli.sh` | Builds Release `macmlx`, strips it, packages `dist/macmlx-${TAG}-arm64.tar.gz` + `.sha256`. |
| `scripts/render-formula.sh` | Fills the template with the rendered tarball URL + sha and writes `dist/macmlx.rb`. |
| `.github/workflows/release.yml` | On `v*.*.*` tag: runs both scripts, attaches tarball + rendered formula to the Release, and (if `HOMEBREW_TAP_TOKEN` is set) pushes the formula to the tap repo. |
| `magicnight/homebrew-mac-mlx` (separate repo) | Tap repo Homebrew clones. Holds `Formula/macmlx.rb`. Other than that, it's empty. |

### Tarball layout

The tarball contains a single top-level executable so the formula's
`bin.install "macmlx"` works without unpacking nested directories:

```
macmlx-v0.3.8-arm64.tar.gz
└── macmlx        (Mach-O arm64, stripped, dynamic Swift stdlib)
```

### Bootstrapping the tap repo (one-time)

1. Create an empty public repo `magicnight/homebrew-mac-mlx` (the
   `homebrew-` prefix is mandatory; Homebrew uses it to resolve
   `brew tap magicnight/mac-mlx`).
2. Add a minimal `README.md` explaining the install command.
3. Cut a release in this repo (`git tag v0.X.Y && git push --tags`).
   The release workflow will produce `dist/macmlx.rb` and attach it to
   the GitHub Release.
4. Either copy `macmlx.rb` into `Formula/macmlx.rb` of the tap repo
   manually, or:
5. Generate a fine-grained GitHub PAT with `Contents: Read+Write` on
   `magicnight/homebrew-mac-mlx`, store it as `HOMEBREW_TAP_TOKEN` in
   this repo's secrets, and re-run the release. The "Publish formula to
   Homebrew tap" step will commit the formula automatically on every
   subsequent release.

### Why a separate tap repo?

Homebrew's tap discovery is hard-coded: `brew tap <user>/<name>`
expects `github.com/<user>/homebrew-<name>`. Nesting the formula inside
the main repo wouldn't be discoverable. Keeping the tap repo otherwise
empty means the formula stays a single sourced-from-here artifact —
no drift risk, no separate test setup.

### Sanity check before release

```bash
# Smoke-test the renderer against the current tag.
GITHUB_REF_NAME=v0.0.0-dev \
MACMLX_CLI_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
    ./scripts/render-formula.sh
cat dist/macmlx.rb

# Lint the rendered formula. Requires `brew` locally.
brew audit --strict --new dist/macmlx.rb || true
```

`brew audit --new` only warns; the formula doesn't ship into
homebrew-core so we accept its tap-formula leniencies.

## Homebrew Cask (GUI, deferred)

A `cask` for `macMLX.app` is **not** in scope. Casks add notarization
requirements (`brew install --cask` validates `xattr` / Gatekeeper
state on macOS 14+), which we don't have until issue #19 lands. Users
who want GUI distribution via Homebrew can revisit this once the DMG
is signed + notarized.

## Before every release: scan upstream mlx for fixes we don't carry

`MacMLXCore/Package.swift` pins a **fork** of mlx-swift by revision. That buys
reproducibility and costs us what a version range gives for free: upstream's
correctness fixes stop arriving. Nothing warns us — build green, tests green,
and the shipped binary quietly runs unpatched kernels.

This is not hypothetical. `ml-explore/mlx#3810` (NAX split-K GEMM reading bf16
inputs as float — silent garbage, no crash, no NaN guard) sat in mlx v0.32.0
for a month while our DMG shipped without it. It surfaced only because a
stranger mentioned it in passing on an unrelated issue thread.

So, as a release step:

```bash
./scripts/scan-upstream-mlx-fixes.sh
```

It resolves the fork point of our vendored mlx, then lists upstream commits
since then that touch `mlx/backend/metal` and read like fixes, minus the ones
the fork already carries. It decides nothing — it hands a human a candidate
list.

For each candidate, the questions that matter:

- **Can it reach us at all?** Some fixes are CUDA-only.
- **Is it JIT-only?** mlx-swift builds with `MLX_METAL_JIT=ON`, so a JIT-only
  fix reaches us even when the AOT-compiled Python wheel is unaffected — this
  is exactly how #3810 hid.
- **What dtype/shape/hardware window triggers it?** Narrow windows still
  matter when the failure is silent; wrong numbers are worse than a crash,
  because they read as a bad model rather than a bug.

Anything that can produce wrong numbers gets a cherry-pick onto the fork
before release, plus a regression test that fails if a later bump drops it
(see `NAXSplitKGemmParityTests` for the shape of one).

When mlx-swift finally vendors mlx-core ≥ 0.32 and the fork dies, delete this
section and the script with it.

### Carrying a fix: bumping the submodule is not always enough

mlx is a git submodule of mlx-swift, but mlx-swift does **not** compile the
submodule's Metal kernel headers. It embeds them, stringified, into checked-in
files under `Source/Cmlx/mlx-generated/`, and the JIT compiles those. Moving
the submodule pointer leaves those copies holding the old kernel text.

So the rule depends on what the upstream fix touched:

| fix touches | submodule bump enough? |
|---|---|
| host C++ (`backend/metal/*.cpp`, `*.h` outside `kernels/`) | yes — compiled from source |
| `backend/metal/kernels/**` | **no** — regenerate `mlx-generated/` too |

`ml-explore/mlx#3810` was the first kind and made the second look unnecessary.
`#3497` and `#3631` were the second kind: after the bump their parity tests
failed with **byte-identical** numbers, because the JIT was still compiling the
pre-fix text. Identical readings rather than improved ones is the tell — if a
fix appears to have no effect at all, check what is actually being compiled
before concluding the fix doesn't apply to us.

When regenerating, replace only the section belonging to the header you
patched. Several generated files embed more than one source — `fp_quantized.cpp`
and `fp_quantized_nax.cpp` each carry `fp4.h` and `fp8.h` alongside the main
one — and replacing wholesale silently drops them.

Verify by diffing the embedded region against its source: they match verbatim
apart from a short auto-generated header, so a content difference means the
regeneration is wrong.

### Carrying a fix is not shipping it: check what each root resolves

`magicnight/mlx-swift` and the `ml-explore/mlx-swift` that `mlx-swift-lm`
depends on share the SwiftPM identity `mlx-swift`. When identities collide the
**root** package's declaration picks the winner, so every root that builds a
shippable artifact has to declare the fork itself. There are three:

| artifact | root | declares the fork in |
|---|---|---|
| `MacMLXCore` tests | `MacMLXCore/Package.swift` | itself |
| `macmlx` CLI / Homebrew tarball | `macmlx-cli/Package.swift` | itself |
| `macMLX.app` / the DMG | `macMLX/macMLX.xcodeproj` | its `packageReferences` |

Each was found the same way and separately: the CLI in 2026-08, the app in
2026-08 while reading unrelated `xcodebuild -list` output. In both cases the
fork was declared in `MacMLXCore` alone, both builds resolved upstream, and
nothing failed — the artifacts built, ran, and produced plausible output using
unpatched kernels. `MacMLXCore`'s parity tests could not catch it because they
are their own root and resolve the fork correctly, which is exactly what made
the gap invisible: the fixes were verified in a graph the shipped product did
not use.

The declaration tests (`CLIForkPinTests`, `AppForkPinTests`) guard against the
declaration being dropped. They do not prove the resolution honored it, so CI
also asserts the app's resolved `Package.resolved` names the fork. When adding
a fourth root, add both.

### A probe that passes has proved nothing until it reached the kernel

These fixes live in kernels chosen by a dispatch decision, so a probe answers
"does this bug reach us?" only if its inputs actually select the kernel that
carries the bug. The first probe for `ml-explore/mlx#3361` used a single query
row to keep the tensors small, and passed — MLX routes to the vector attention
kernel at `query_sequence_length <= 8` and to the steel/NAX one above it, so it
had exercised a kernel that never had the defect. Re-run at 128 query rows it
failed immediately: cosine 0.0 and non-finite output.

Before reading a pass as "not applicable", find the dispatch condition in the
backend and check the probe satisfies it. Read the vendored code for the defect
itself too — `#3422` really is inapplicable to us, but that is because our
vintage predates the refactor that introduced it, which only the source shows.
Absent both checks a green probe means "did not reach it", not "does not have
it".
