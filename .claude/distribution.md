# Distribution

## Build Requirements

- Xcode 26.4.1, the version `ci.yml` and `release.yml` pin (Xcode 27.0 builds and tests it too)
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

Four jobs on `macos-26`: `website` (static site build and tests, the
runner's Node), and with Xcode 26.4.1 pinned `spm` (`swift test` for
MacMLXCore and macmlx-cli, the lockfiles unchanged by resolve, the CLI's
lockfile in parity with MacMLXCore's, `macmlx --version`), `metal` (the full
MacMLXCore suite under `xcodebuild test`; strict numeric parity is skipped
on the runners' paravirtualized Metal), `app` (unsigned Debug build from the
committed workspace lockfile only, that lockfile tracked, unchanged by the
build and in parity with MacMLXCore's, `macMLXTests`).

`scripts/ci-local.sh [website] [spm] [metal] [app]` runs the same jobs with
the same commands on a developer machine, in a fresh worktree of the commit
under test (`CI_LOCAL_REF`, default HEAD): shared caches under
`~/Library/Caches/macmlx-ci-local` held by one run at a time, a `summary.md`
per run with per-job evidence from the result bundles, the Xcode `ci.yml`
pins when it is installed (else the selected one, and the verdict says so),
`CI_LOCAL_UNTRUSTED_METAL=1` to skip parity as CI does. When Actions minutes
are out it is the merge gate: the maintainer runs all four jobs from `main`'s
copy of the script against the exact PR head and posts the summary in the
PR. `ci.yml` and the script change together.

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

A patched kernel has **three** copies in the fork, and an audit that finds only
one of them will reach the wrong conclusion:

| copy | compiled? |
|---|---|
| `mlx/backend/metal/kernels/**` (the submodule) | no — the authoritative source only |
| `mlx-generated/<name>.cpp` (stringified) | **yes**, by the JIT at runtime |
| `mlx-generated/metal/**/<name>.h` | no, unless a `.metal` in `KERNEL_LIST` includes it |

Patch all three anyway. `tools/fix-metal-includes.sh` copies every header it
finds under the kernels directory, so the third one is one regeneration away
from becoming live, and until then it is a second answer to "did we patch this
kernel?" that contradicts the first.

Whether a NAX kernel is JIT-only is checkable: `KERNEL_LIST` in that script is
the set compiled ahead of time. `steel_attention_nax.metal` is not in it, and
nothing under `mlx-generated/metal/` includes the NAX header — that kernel is
built at runtime from `metal::steel_attention_nax()`, i.e. from the `.cpp`.

### Moving off v0.31.1 is not a version bump

The fork's exit condition — "drop the override once mlx-swift vendors core >=
0.32" — reads like a one-line change. It is not. This section was rewritten on
2026-10-02, when mlx-swift 0.32.3 (vendoring core v0.32.2) made the move
possible and we did it. What follows is what the move actually required, which
differs from what the previous version of this section predicted. Keep both
lessons: the facts, and the fact that a landmine table rots.

**What the fork was for, and why it shrank to two patches.** Twelve of the
thirteen cherry-picks the fork carried on top of v0.31.1 are in core v0.32.2 —
every one verified by `git merge-base --is-ancestor` against `1f8e74e3`, and
the three that live in kernel headers (`mlx#3631`, `mlx#3497`, `mlx#3361`) also
checked line by line in upstream's shipped `mlx-generated/` copies. Only
`mlx#3922` (sorted gather_qmm row overflow past 32K) landed later, in v0.32.3,
and the 2026-10-02 scan added `mlx#4009` (sorted gather_qmm on ragged K —
92–97% of output elements wrong at group size 32 when the quantized dimension is
not a multiple of 64; also only in v0.32.3). The fork now carries exactly those
two, both applied as cherry-picks onto a `v0.32.2-macmlx` branch of our mlx
mirror, with the generated copies regenerated by upstream's own
`tools/update-mlx.sh` rather than patched by hand.

**The cross-thread abort is gone, by construction.** `ml-explore/mlx#3348` made
`default_stream()` thread-local in core v0.31.2, and mlx-swift 0.31.x's
`Stream.gpu` was a process-global `static let`, so evaluating from a second
thread aborted (`There is no Stream(gpu, 0) in current thread`). That was
`ml-explore/mlx-swift#457`, closed by the maintainer on 2026-09-28. In 0.32.3,
`Source/MLX/Stream.swift` creates every stream through
`mlx_stream_new_thread_unsafe` (mlx-c `ebc88f1`, which contains
`ml-explore/mlx-c#122`), the default is a `@TaskLocal` override with a global
fallback, and there is no fatal path for a thread that never set one.
`CrossThreadEvalTripwireTests` is the runtime proof; reading the code only says
it should not abort. Its failure mode is still a process abort that takes the
test runner with it, so look at whether the runner survived, not at the summary.

**What the previous table got wrong.** Four of its five rows were fixed
upstream between being written and being read, and the fifth was a
misdiagnosis:

| it said | what was actually true at 0.32.3 |
|---|---|
| `qmv()` 6-vs-5 template args abort JIT builds | fixed by `mlx#4372`, in v0.32.2 — but see below, that PR had a second half we had missed |
| generated `*_nax.cpp` need `MLX_METAL_NO_NAX` below deployment target 26.2 | wrong. NAX is gated at **runtime** by `__builtin_available(macOS 26.2, …)`; `MLX_METAL_NO_NAX` is a CMake-only define SwiftPM never sets; all eight `*_nax.cpp` ship in 0.32.3 and `platforms:` still says `.macOS("14.0")` — upstream's own `update-mlx.sh` runs cmake with `-DMACOS_VERSION=14.0` |
| `update-mlx.sh` make-list missing `gemv`/`searchsorted` | fixed — the list is now derived from `make_jit_source()` in CMake (`mlx-swift#445`) |
| mlx-c compile-cache handle | adopted in `Transforms+Compile.swift` |
| FFT norm parameter | adopted in `FFT.swift` |

**What it missed.** `mlx#4372` fixed two unrelated things. The template-arg
half never applied to us (our `affine_qmv` had no `results_per_simdgroup`
parameter), which is why the whole PR was excluded in August. The other half
removed a trailing underscore from two kernel-name strings in
`quantized.cpp` — `"gather_qmm_t_nax_"` where the instantiated kernel is
`affine_gather_qmm_t_nax` — a JIT-only failure, and we are JIT-only. It had
been live in every build since the fork was created. Excluding a PR by its
title is how that happens; read the diff.

**The landmines that are real.**

| what | where it bites |
|---|---|
| Core v0.32.2 requests **Metal 4.1** on macOS 27 (`device.cpp` `get_metal_version()`); v0.31.1 capped at 4.0 | the first time our JIT kernels compile under the 4.1 language version is on a macOS 27 developer machine, never in CI, which pins Xcode 26.4.1 on `macos-26`. `mlx#3963` (explicit `thread` address space, 36 files) is in v0.32.2 and makes that compile; without the move it would have been the most expensive cherry-pick on the list |
| mlx-swift-lm 3.32.3 declares a SwiftPM trait `FoundationModelsIntegration`, default on, gating `MLXFoundationModels` — a target that compiles on any Xcode, since its contents sit behind `@available` and `canImport(FoundationModels)` | irrelevant unless a root depends on the `MLXFoundationModels` product. We do not; `MLXLLM`/`MLXVLM`/`MLXEmbedders`/`MLXLMCommon` do not pull it in. Verify by checking the build log for `MLXCXGrammar` if that ever changes |
| mlx-swift-lm 3.32.3 reworked KV-cache handling, deleted the "infer chains" (`#502`), and changed `LogitProcessor` copy semantics (`#533`) | our `Batching/*` and the ten hand-written model families under `Models/` are the test surface, not the eleven op-level behaviour changes in the 0.32.2 notes (none of which our own code uses) |
| 0.32.2 crashed on launch below macOS 26.4 (`Logger.isEnabled(type:)`, `mlx-swift#491`) | 0.32.3 replaces it with a C shim over `os_log_type_enabled`, a 10.12-era API, so macOS 14 and 15 are restored. Never pin 0.32.2 |
| The fork's base must move together with mlx-swift-lm | the root's revision pin overrides mlx-swift-lm's `.upToNextMinor(from: "0.32.3")`, so SwiftPM never forces the move — source compatibility does. Staying on a 0.31.x base means staying on mlx-swift-lm 3.31.4 and everything after it |
| mlx-swift-lm 3.32.3 changed the default prefill chunking from stride-of-step-size to `.balanced` (`PrefillParameters.Chunking`), and the deprecated `TokenIterator.init(…prefillStepSize:)` forwards into that new default | a silent change to chunk boundaries — and therefore to exact outputs — on every generation path, including `generateTokens`. The move pins `chunking: .remainder` where `GenerateParameters` is built, which restores the legacy stride and reserved tail on the generic `LLMModel` path and every VLM path (checked by reading against 3.31.4). It does not cover Gemma3Text, whose prefill 3.32.3 reworked on its own, and no output-level comparison was possible here. The batched path's prefill is hand-rolled in `ModelBatchInferenceCore` and never reads this setting. Adopting `.balanced` is a measured change of its own, not a dependency bump side effect |
| mlx-swift-lm 3.32.3 added a complete `compile()` overload matrix to `MLXLMCommon` whose bodies are `@Sendable` (`#589`, `CompileOverloads.swift`) | any module that imports both `MLX` and `MLXLMCommon` and calls bare `compile { … }` with a non-Sendable capture stops compiling, because the call now resolves to the `@Sendable` overload. mlx-audio-swift's `ParakeetModel.swift` had two such calls; the fix is to write `MLX.compile`, which is what they always were. Expect this in any other MLX-side dependency that imports `MLXLMCommon` |
| The prompt cache keys a finished response on the stop token the loop delivers with `includeStopToken` (#142), recognised with a copy of the loop's private `buildStopTokenIds` plus `unknownTokenId` | a bump that changes what the loop delivers, when it breaks, or which tokens it stops on is caught only by two gated smokes, so run both on every mlx-swift-lm bump: `MACMLX_RUN_MELLUM_SMOKE=1 TEST_RUNNER_MACMLX_RUN_MELLUM_SMOKE=1 MACMLX_RUN_GEMMA4_SMOKE=1 TEST_RUNNER_MACMLX_RUN_GEMMA4_SMOKE=1 xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' -skipPackagePluginValidation -only-testing:MacMLXCoreTests/Mellum2SmokeTests/testMellum2EOSStopRecordsTheStopTokenInItsKey -only-testing:MacMLXCoreTests/Gemma4PromptCacheSmokeTests` (both variables, with and without the `TEST_RUNNER_` prefix: xcodebuild hands the test runner only the prefixed ones, and a missing one is an `XCTSkip` that reports green; look for the `[eos-smoke]` lines). Their record checks fail when the cache and the key stop agreeing, and `MLXSwiftEngine.promptCacheKey` says what the engine assumes |

**Still true from v0.31.1 days.** `BatchPositionedCacheWrapper` existed because
0.31.6's `mx.fast.rope` corrupted batched single-token decode; its test file
declared a deletion tripwire for the day the vendored core reached 0.32.0. That
day is this move, and the wrapper went with it.

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
declaration being dropped. They do not prove the resolution honored it, and a
declaration pins one package while a root resolves a whole graph: the app's
workspace lockfile was gitignored until 2026-10, so a release archive on a
fresh runner took the newest release each manifest allowed on the day, and the
v0.9.0 DMG shipped swift-jinja 2.4.2 while every chat-template test had run on
the 2.3.6 both committed lockfiles pinned (#149). So every root's lockfile is committed (the
app's at `macMLX/macMLX.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`),
`scripts/check-lockfile-parity.py` holds the CLI's and the app's to
`MacMLXCore`'s in CI and in `scripts/ci-local.sh`, `AppLockfileParityTests`
does the same inside Xcode, and the app's CI build and `scripts/build.sh`'s
archive pass `-onlyUsePackageVersionsFromResolvedFile`, so a lockfile the
manifests have outgrown fails the build instead of being rewritten. Parity
with `MacMLXCore`'s lockfile subsumes the old check that the app resolved the
fork: Core's lockfile names the fork at the revision Core pins, and the spm
job keeps it fresh. When adding a fourth root, add all three: the declaration,
its test, and the lockfile under the parity check.

#### Moving a dependency

Moving a dependency moves three lockfiles. Change the manifest, then
`swift package --package-path MacMLXCore resolve`, the same for `macmlx-cli`,
and for the app `xcodebuild -resolvePackageDependencies -project
macMLX/macMLX.xcodeproj -scheme macMLX`, without the flag: with it the stale
lockfile is an error, which is what CI and the archive are meant to see. A
plain resolve takes the newest release the new requirement allows, so the
app can land on a different release from Core's (it did, 1.3.5 against
Core's 1.3.4, in the probe that proved the flag). When it does, copy
`MacMLXCore/Package.resolved` over the app's lockfile, put the app-only pins
back from the old file (today Sparkle's; the CLI's swift-argument-parser is
the same case on that side), and resolve again: Xcode keeps every pin that
satisfies the manifests and rewrites only the origin hash, which is how the
committed file was made, and a pin it re-adds lands on the newest release
that day. Keep SwiftPM's own JSON style when editing a lockfile by hand or
by script (sorted keys, pins sorted by identity, two-space indent,
`"key" : value` with the spaces, `/` unescaped, a trailing newline), or the
diff is the whole file and the next resolve rewrites it. Run
`scripts/check-lockfile-parity.py` on the CLI's and the app's lockfiles,
read the diff for a package that moved without being asked, and commit all
three. A change to Core's manifest alone never moves the app's origin
hash (probed with swift-jinja's floor raised from 2.4.2 to 2.4.3 under the
2.5.1 pin, and seen again when swift-huggingface's floor rose to 0.11.0
past the 0.9.0 pin): the hash covers the workspace's own package
references. When the pins still satisfy the new floor nothing else changes
either; when they do not, only the pins that must move change, as
swift-huggingface's did in that second case.

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
