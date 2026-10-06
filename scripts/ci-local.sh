#!/usr/bin/env bash
# scripts/ci-local.sh — Run the CI pipeline (.github/workflows/ci.yml) on this machine.
#
# Mirrors the four jobs of ci.yml step for step — website, spm, metal, app —
# with the same commands and the same environment, for when GitHub Actions is
# not available (minutes exhausted) or to check a branch before pushing it.
#
# Usage: scripts/ci-local.sh [website] [spm] [metal] [app]    (default: all four)
#
# Environment:
#   CI_LOCAL_CACHE            build caches shared across checkouts (SwiftPM
#                             scratch paths, one DerivedData per xcodebuild
#                             job); stands in for actions/cache.
#                             Default ~/Library/Caches/macmlx-ci-local
#   CI_LOCAL_LOGS             where the per-job logs and summary.md go.
#                             Default $CI_LOCAL_CACHE/logs/<utc time>-<head>-<pid>
#   CI_LOCAL_UNTRUSTED_METAL  set to 1 to skip the strict numeric parity suites
#                             the way ci.yml does on GitHub's paravirtualized
#                             Metal. Leave unset on real Apple Silicon: parity is
#                             enforced here, which is stricter than CI.
#   DEVELOPER_DIR             the Xcode to use. ci.yml pins Xcode 26.4.1; this
#                             script uses whatever xcode-select points at unless
#                             told otherwise, and prints what it used.
#
# Differences from ci.yml, all deliberate:
#   - no `sudo xcode-select`: DEVELOPER_DIR does the same without privileges;
#   - the Metal toolchain is checked, not downloaded (see CONTRIBUTING.md);
#   - paths-ignore and the concurrency group do not apply: you pick the jobs.
#
# Exit status: 0 when every selected job passed. The summary names the log of
# each failed job; the last lines of that log are printed as well.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MACMLX_SKIP_LOG_MANAGER_TESTS=1   # as ci.yml's top-level env

CACHE="${CI_LOCAL_CACHE:-$HOME/Library/Caches/macmlx-ci-local}"
HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
DIRTY=""
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then DIRTY=" (dirty tree)"; fi
LOGS="${CI_LOCAL_LOGS:-$CACHE/logs/$(date -u +%Y%m%dT%H%M%SZ)-$HEAD-$$}"
# One DerivedData per xcodebuild job, as CI has one runner per job: the
# package and the app project resolve the same dependencies into
# `SourcePackages` at their own versions, and sharing one directory left the
# package's resolution reading the app's checkouts (a hummingbird trait error).
DERIVED_METAL="$CACHE/DerivedData/metal"
DERIVED_APP="$CACHE/DerivedData/app"
mkdir -p "$LOGS" "$DERIVED_METAL" "$DERIVED_APP" "$CACHE/spm"

JOBS=("$@")
if [ ${#JOBS[@]} -eq 0 ]; then JOBS=(website spm metal app); fi
for job in "${JOBS[@]}"; do
    case "$job" in website|spm|metal|app) ;; *) echo "unknown job '$job' (website, spm, metal, app)" >&2; exit 2;; esac
done

echo "==> ci-local on $HEAD$DIRTY — jobs: ${JOBS[*]}"
echo "    logs:  $LOGS"
echo "    cache: $CACHE"
{
    sw_vers
    echo "DEVELOPER_DIR=${DEVELOPER_DIR:-$(xcode-select -p)}"
    xcodebuild -version || echo "xcodebuild: not found"
    swift --version || echo "swift: not found"
    node --version || echo "node: not found"
} > "$LOGS/toolchain.log" 2>&1 || true
sed 's/^/    /' "$LOGS/toolchain.log"

# ---------------------------------------------------------------- jobs ----

job_website() {
    node scripts/validate-social-cards.mjs
    node scripts/build-public-site.mjs
    node --test site/tests/*.test.mjs
    node scripts/crawl-public-site.mjs
    node scripts/test-public-site.mjs
    node --check scripts/build-public-site.mjs
    node --check scripts/crawl-public-site.mjs
    node --check scripts/render-brand-icons.mjs
    node --check scripts/render-social-cards.mjs
    node --check scripts/validate-social-cards.mjs
    node --check scripts/verify-cloudflare-deploy.mjs
    node --check site/lib/install-manifest.mjs
    node --check site/lib/png-source-digest.mjs
    node --check site/cloudflare/www-redirect.mjs
    node --check scripts/verify-www-redirect.mjs
    node --check public/assets/js/main.js
    if command -v xmllint >/dev/null 2>&1; then
        xmllint --noout site/assets/brand/macmlx-mark.svg site/assets/brand/favicon.svg public/assets/og-image.svg public/sitemap.xml
    fi
}

job_spm() {
    local core="$CACHE/spm/MacMLXCore" cli="$CACHE/spm/macmlx-cli"
    swift package --package-path MacMLXCore --scratch-path "$core" resolve
    swift test --package-path MacMLXCore --scratch-path "$core"
    swift package --package-path macmlx-cli --scratch-path "$cli" resolve
    swift test --package-path macmlx-cli --scratch-path "$cli"
    # The resolves above rewrite a stale Package.resolved in place; the rewrite
    # is the evidence (see the comment in ci.yml).
    if ! git diff --exit-code -- MacMLXCore/Package.resolved macmlx-cli/Package.resolved; then
        echo "error: a Package.resolved changed during resolve — the checked-in lockfile is stale."
        echo "Run 'swift package --package-path <pkg> resolve' and commit the result."
        return 1
    fi
    swift run --package-path macmlx-cli --scratch-path "$cli" macmlx --version
}

job_metal() {
    # ci.yml downloads the Metal toolchain component; here it must already be
    # installed (CONTRIBUTING.md: `sudo xcodebuild -downloadComponent MetalToolchain`).
    if ! xcrun --find metal >/dev/null 2>&1; then
        echo "error: the Metal toolchain is not installed; run: sudo xcodebuild -downloadComponent MetalToolchain"
        return 1
    fi
    if [ "${CI_LOCAL_UNTRUSTED_METAL:-}" = "1" ]; then
        export TEST_RUNNER_MACMLX_UNTRUSTED_METAL=1
    fi
    ( cd MacMLXCore && xcodebuild test \
        -scheme MacMLXCore \
        -destination 'platform=macOS' \
        -skipPackagePluginValidation \
        -derivedDataPath "$DERIVED_METAL" )
}

job_app() {
    xcodebuild \
        -project macMLX/macMLX.xcodeproj \
        -scheme macMLX \
        -configuration Debug \
        -destination 'platform=macOS' \
        -skipPackagePluginValidation \
        -derivedDataPath "$DERIVED_APP" \
        CODE_SIGN_IDENTITY="" \
        CODE_SIGNING_REQUIRED=NO \
        CODE_SIGNING_ALLOWED=NO \
        build

    # The app must resolve the controlled mlx-swift fork at the revision
    # MacMLXCore pins (the same check as ci.yml, verbatim).
    local resolved=macMLX/macMLX.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved
    if [ ! -f "$resolved" ]; then
        echo "error: $resolved is missing — the app never resolved its packages."
        return 1
    fi
    python3 - "$resolved" MacMLXCore/Package.swift <<'PY'
import json, re, sys

manifest = open(sys.argv[2]).read()
match = re.search(
    r'magicnight/mlx-swift\.git".*?revision:\s*"([0-9a-f]+)"', manifest, re.S)
if not match:
    print("error: MacMLXCore/Package.swift no longer pins the fork by revision.")
    sys.exit(1)
declared = match.group(1)

pins = json.load(open(sys.argv[1]))
pins = pins.get("pins") or pins.get("object", {}).get("pins", [])
for pin in pins:
    location = pin.get("location") or pin.get("repositoryURL", "")
    if "mlx-swift" not in location or "mlx-swift-lm" in location:
        continue
    if "magicnight/mlx-swift" not in location:
        print(
            f"error: the app resolved {location} instead of "
            "magicnight/mlx-swift. It would ship without the fork's "
            "cherry-picks."
        )
        sys.exit(1)
    resolved = pin.get("state", {}).get("revision", "")
    if resolved != declared:
        print(
            f"error: the app resolved the fork at {resolved}, but "
            f"MacMLXCore pins {declared}. A stale resolution ships "
            "cherry-picks that were never tested together."
        )
        sys.exit(1)
    print(f"app resolved the fork at the pinned revision {resolved}")
    sys.exit(0)
print("error: the app's resolved graph names no mlx-swift at all.")
sys.exit(1)
PY

    # App-target unit tests: the app is the TEST_HOST, so it is ad-hoc signed
    # here (an unsigned bundle will not launch), as in ci.yml.
    xcodebuild test \
        -project macMLX/macMLX.xcodeproj \
        -scheme macMLXTests \
        -configuration Debug \
        -destination 'platform=macOS' \
        -skipPackagePluginValidation \
        -derivedDataPath "$DERIVED_APP"
}

# ------------------------------------------------------------- runner ----

SUMMARY="$LOGS/summary.md"
{
    echo "ci-local on \`$HEAD\`$DIRTY — $(date -u +'%Y-%m-%d %H:%M UTC'), $(sw_vers -productVersion), $(xcodebuild -version | head -1), $(swift --version 2>/dev/null | head -1 | sed 's/ (.*//')"
    echo
    echo "| job | result | duration | evidence | log |"
    echo "|---|---|---|---|---|"
} > "$SUMMARY"

# The lines of a job's log that say what ran: swift-testing and XCTest totals,
# the fork-pin verdict, the CLI's version. Counted, since a suite's total line
# can be cut by an interleaved xcodebuild line while its per-test lines stay.
evidence() {
    {
        grep -o 'Test run with [0-9]* tests in [0-9]* suites passed' "$1" | sort | uniq -c | sed 's/^ *\([0-9]*\) /\1× /'
        grep -o 'Executed [0-9]* tests, with [0-9]* tests skipped and [0-9]* failures' "$1" | tail -1
        grep -o 'Executed [0-9]* tests, with [0-9]* failures' "$1" | tail -1
        n=$(grep -c '✔ Test "' "$1" || true); f=$(grep -c '✘ Test "' "$1" || true)
        [ "$n" -gt 0 ] && echo "$n ✔ / $f ✘ test lines"
        grep -o 'app resolved the fork at the pinned revision [0-9a-f]\{7\}' "$1" | head -1
        grep -o '^macmlx [0-9][^ ]*' "$1" | head -1
    } 2>/dev/null | paste -sd ';' - | sed 's/;/; /g'
}

FAILED=0
for job in "${JOBS[@]}"; do
    log="$LOGS/$job.log"
    start=$(date +%s)
    printf '==> %-8s started %s\n' "$job" "$(date +%H:%M:%S)"
    # The job runs in its own shell with errexit on, as a plain command: inside
    # an `if` or a `||` list bash switches errexit off for everything the
    # condition runs, functions included, and a job would then "pass" on its
    # last command alone.
    set +e
    ( set -e; "job_$job" ) > "$log" 2>&1
    status=$?
    set -e
    if [ "$status" -eq 0 ]; then
        result="passed"
    else
        result="FAILED"
        FAILED=$((FAILED + 1))
    fi
    duration=$(( $(date +%s) - start ))
    printf '==> %-8s %s in %dm%02ds (%s)\n' "$job" "$result" $((duration / 60)) $((duration % 60)) "$log"
    echo "| $job | $result | $((duration / 60))m$((duration % 60))s | $(evidence "$log") | \`$log\` |" >> "$SUMMARY"
    if [ "$result" = "FAILED" ]; then
        echo "    --- last 40 lines of $log ---"
        tail -40 "$log" | sed 's/^/    /'
    fi
done

echo
cat "$SUMMARY"
echo
if [ "$FAILED" -ne 0 ]; then
    echo "==> $FAILED job(s) failed"
    exit 1
fi
echo "==> all ${#JOBS[@]} job(s) passed"
