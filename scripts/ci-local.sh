#!/usr/bin/env bash
# scripts/ci-local.sh — Run the CI pipeline (.github/workflows/ci.yml) on this machine.
#
# Mirrors the four jobs of ci.yml job for job and step for step — website,
# spm, metal, app — with the same commands and the same environment, for
# when GitHub Actions is not available (minutes exhausted) or to check a
# commit before pushing it. The jobs run in a throwaway worktree of the
# commit under test, as CI runs in a fresh checkout: uncommitted and
# untracked files in this checkout are not part of the run, and nothing
# gitignored (the app's Package.resolved, say) carries over between runs.
#
# Usage: scripts/ci-local.sh [website] [spm] [metal] [app]    (default: all four)
#
# Environment:
#   CI_LOCAL_REF              the commit to test (default HEAD). A PR head is
#                             graded by main's copy of this script:
#                             CI_LOCAL_REF=<sha> scripts/ci-local.sh
#   CI_LOCAL_CACHE            build caches shared across checkouts (SwiftPM
#                             scratch paths, one DerivedData per xcodebuild
#                             job); stands in for actions/cache. One run at a
#                             time holds them. Default ~/Library/Caches/macmlx-ci-local
#   CI_LOCAL_LOGS             where the per-job logs, result bundles and
#                             summary.md go.
#                             Default $CI_LOCAL_CACHE/logs/<utc time>-<head>-<pid>
#   CI_LOCAL_UNTRUSTED_METAL  set to 1 to skip the strict numeric parity suites
#                             the way ci.yml does on GitHub's paravirtualized
#                             Metal. Leave unset on real Apple Silicon: parity is
#                             enforced here, which is stricter than CI.
#   DEVELOPER_DIR             the Xcode to use. Unset, the Xcode ci.yml pins is
#                             used when it is installed, else the selected one;
#                             the summary and the verdict say when that differs
#                             from the pin.
#
# Differences from ci.yml, all deliberate:
#   - no `sudo xcode-select`: DEVELOPER_DIR does the same without privileges;
#   - the Metal toolchain is checked, not downloaded (see CONTRIBUTING.md);
#   - the machine's own macOS, Node and (unless the pin is installed) Xcode,
#     not the runner's;
#   - the inherited environment is cleaned of every TEST_RUNNER_*, MACMLX_*,
#     MLX_* and TOOLCHAINS variable first, so a shell's leftovers cannot turn
#     gated suites on or off;
#   - no per-job timeout (ci.yml has 5/30/30/30 minutes): a hang is killed by hand;
#   - paths-ignore and the concurrency group do not apply: you pick the jobs.
#
# Exit status: 0 when every selected job passed, 1 when one failed, 2 for a
# bad argument, 3 when another run holds the caches. The summary names the
# log of each failed job; the last lines of that log are printed as well.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ------------------------------------------------------ the environment ----

while IFS= read -r name; do
    case "$name" in
        TEST_RUNNER_*|MACMLX_*|MLX_*|TOOLCHAINS) unset "$name" ;;
    esac
done <<EOF
$(env | sed 's/=.*//')
EOF
export MACMLX_SKIP_LOG_MANAGER_TESTS=1   # as ci.yml's top-level env

# ------------------------------------------------------------- the jobs ----

JOBS=()
for job in "$@"; do
    case "$job" in
        website|spm|metal|app) ;;
        *) echo "unknown job '$job' (website, spm, metal, app)" >&2; exit 2 ;;
    esac
    case " ${JOBS[*]:-} " in
        *" $job "*) ;;
        *) JOBS+=("$job") ;;
    esac
done
if [ ${#JOBS[@]} -eq 0 ]; then JOBS=(website spm metal app); fi

# -------------------------------------------------------- the toolchain ----

PIN="$(grep -o 'Xcode_[0-9.]*\.app' "$ROOT/.github/workflows/ci.yml" | head -1 || true)"
PIN_VERSION="${PIN#Xcode_}"
PIN_VERSION="${PIN_VERSION%.app}"
if [ -z "${DEVELOPER_DIR:-}" ] && [ -n "$PIN" ] && [ -d "/Applications/$PIN" ]; then
    export DEVELOPER_DIR="/Applications/$PIN/Contents/Developer"
fi
XCODE="$(xcodebuild -version 2>/dev/null | head -1 || true)"
XCODE="${XCODE:-Xcode: not found}"
TOOLCHAIN_NOTE=""
if [ -z "$PIN" ]; then
    TOOLCHAIN_NOTE="ci.yml pins no Xcode; this run used $XCODE"
else
    case "$XCODE" in
        "Xcode $PIN_VERSION"|"Xcode $PIN_VERSION "*) ;;
        *) TOOLCHAIN_NOTE="toolchain differs from ci.yml: $XCODE here, Xcode $PIN_VERSION on CI" ;;
    esac
fi
if [ "${CI_LOCAL_UNTRUSTED_METAL:-}" = "1" ]; then
    PARITY="parity suites skipped (CI_LOCAL_UNTRUSTED_METAL=1, as CI)"
else
    PARITY="strict parity suites run"
fi

# ---------------------------------------------- caches, lock, logs, tree ----

CACHE="${CI_LOCAL_CACHE:-$HOME/Library/Caches/macmlx-ci-local}"
mkdir -p "$CACHE"
CACHE="$(cd "$CACHE" && pwd)"
if ! mkdir "$CACHE/.lock" 2>/dev/null; then
    echo "another ci-local run holds $CACHE/.lock (remove it if no run is alive)" >&2
    exit 3
fi
TREE=""
cleanup() {
    if [ -n "$TREE" ]; then
        git -C "$ROOT" worktree remove --force "$TREE" >/dev/null 2>&1 || true
    fi
    rmdir "$CACHE/.lock" 2>/dev/null || true
}
trap cleanup EXIT

REF="${CI_LOCAL_REF:-HEAD}"
SHA="$(git -C "$ROOT" rev-parse --verify "$REF^{commit}")"
SHORT="${SHA:0:7}"
LOGS="${CI_LOCAL_LOGS:-$CACHE/logs/$(date -u +%Y%m%dT%H%M%SZ)-$SHORT-$$}"
mkdir -p "$LOGS"
LOGS="$(cd "$LOGS" && pwd)"
DERIVED_METAL="$CACHE/DerivedData/metal"
DERIVED_APP="$CACHE/DerivedData/app"
mkdir -p "$DERIVED_METAL" "$DERIVED_APP" "$CACHE/spm"

# The commit under test, checked out on its own, as CI checks it out. The
# path is fixed, since the caches key on paths and stay warm only then.
TREE="$CACHE/tree"
if [ -e "$TREE" ]; then
    git -C "$ROOT" worktree remove --force "$TREE" >/dev/null 2>&1 || rm -rf "${TREE:?}"
fi
git -C "$ROOT" worktree prune
git -C "$ROOT" worktree add --detach --quiet "$TREE" "$SHA"
LEFT_BEHIND="$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
cd "$TREE"

echo "==> ci-local on $SHORT ($REF) — jobs: ${JOBS[*]}"
echo "    tree:  $TREE (fresh worktree of $SHORT)"
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
[ -n "$TOOLCHAIN_NOTE" ] && echo "    note:  $TOOLCHAIN_NOTE"
[ "$LEFT_BEHIND" != "0" ] && echo "    note:  $LEFT_BEHIND uncommitted or untracked path(s) in $ROOT are not part of this run"

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
    echo "### swift test MacMLXCore"
    swift package --package-path MacMLXCore --scratch-path "$core" resolve
    swift test --package-path MacMLXCore --scratch-path "$core"
    echo "### swift test macmlx-cli"
    swift package --package-path macmlx-cli --scratch-path "$cli" resolve
    swift test --package-path macmlx-cli --scratch-path "$cli"
    # The resolves above rewrite a stale Package.resolved in place; the rewrite
    # is the evidence (see the comment in ci.yml).
    echo "### lockfiles"
    if ! git diff --exit-code -- MacMLXCore/Package.resolved macmlx-cli/Package.resolved; then
        echo "error: a Package.resolved changed during resolve — the checked-in lockfile is stale."
        echo "Run 'swift package --package-path <pkg> resolve' and commit the result."
        return 1
    fi
    echo "### macmlx --version"
    swift run --package-path macmlx-cli --scratch-path "$cli" macmlx --version
}

job_metal() {
    # ci.yml downloads the Metal toolchain component; here it must already be
    # installed (CONTRIBUTING.md: `sudo xcodebuild -downloadComponent MetalToolchain`).
    # `xcrun --find metal` is no check: Xcode ships a stub at that path.
    if ! xcodebuild -showComponent MetalToolchain 2>/dev/null | grep -q 'Status: installed'; then
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
        -derivedDataPath "$DERIVED_METAL" \
        -resultBundlePath "$LOGS/metal.xcresult" )
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
    # MacMLXCore pins: the same check as ci.yml, its messages without the
    # `::error::` prefix. The lockfile it reads is gitignored and the tree is
    # fresh, so the build above resolved it, as on CI.
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
        -derivedDataPath "$DERIVED_APP" \
        -resultBundlePath "$LOGS/app.xcresult"
}

# ------------------------------------------------------------ evidence ----

# What a job ran. An xcodebuild job's counts come from its result bundle: the
# console is xcodebuild's own log interleaved with the test runner's, and a
# summary line can be cut in two. `swift test` writes its summaries itself,
# in order, so the spm job's come from the console, per package.
evidence() {
    local job="$1" log="$2" bundle="$LOGS/$1.xcresult"
    {
        if [ -d "$bundle" ]; then
            local summary="$LOGS/$job.summary.json"
            if xcrun xcresulttool get test-results summary --path "$bundle" > "$summary" 2>/dev/null; then
                python3 - "$summary" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
def g(key): return s.get(key, "?")
print(f"{g('passedTests')} passed, {g('failedTests')} failed, {g('skippedTests')} skipped of {g('totalTestCount')} ({g('result')})")
PY
            else
                echo "result bundle unreadable"
            fi
        elif [ "$job" = "spm" ]; then
            awk '
                /^### swift test / { pkg = $4; order[++n] = pkg; next }
                /^### / { pkg = "" ; next }
                pkg != "" && match($0, /Test run with [0-9]+ tests in [0-9]+ suites passed/) { st[pkg] = substr($0, RSTART, RLENGTH) }
                pkg != "" && match($0, /Executed [0-9]+ tests, with [^(]*failures/) { xc[pkg] = "XCTest " substr($0, RSTART, RLENGTH) }
                END { for (i = 1; i <= n; i++) { p = order[i]; print p ": " st[p] ";" xc[p] } }
            ' "$log"
            # `macmlx --version` prints the bare version on its own line.
            grep -o '^[0-9][0-9]*\.[0-9][0-9]*\.[0-9][^ ]*$' "$log" | tail -1 | sed 's/^/macmlx --version: /'
        fi
        grep -o 'app resolved the fork at the pinned revision [0-9a-f]\{7\}' "$log" | head -1
    } 2>/dev/null | paste -sd ';' - | sed 's/;/; /g'
}

# ------------------------------------------------------------- runner ----

SUMMARY="$LOGS/summary.md"
{
    echo "ci-local on \`$SHA\` ($REF) — $(date -u +'%Y-%m-%d %H:%M UTC'), macOS $(sw_vers -productVersion), $XCODE, $(swift --version 2>/dev/null | head -1 | sed 's/ (.*//' || true); $PARITY; jobs: ${JOBS[*]}"
    [ -n "$TOOLCHAIN_NOTE" ] && echo "" && echo "**$TOOLCHAIN_NOTE**"
    [ "$LEFT_BEHIND" != "0" ] && echo "" && echo "$LEFT_BEHIND uncommitted or untracked path(s) in the checkout were not part of this run."
    echo ""
    echo "logs: \`${LOGS/#$HOME/~}\`"
    echo ""
    echo "| job | result | duration | evidence | log |"
    echo "|---|---|---|---|---|"
} > "$SUMMARY"

FAILED=0
FAILED_JOBS=""
for job in "${JOBS[@]}"; do
    log="$LOGS/$job.log"
    rm -rf "${LOGS:?}/${job:?}.xcresult"
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
        FAILED_JOBS="$FAILED_JOBS $job"
    fi
    duration=$(( $(date +%s) - start ))
    printf '==> %-8s %s in %dm%02ds (%s)\n' "$job" "$result" $((duration / 60)) $((duration % 60)) "$log"
    echo "| $job | $result | $((duration / 60))m$((duration % 60))s | $(evidence "$job" "$log") | \`$job.log\` |" >> "$SUMMARY"
    if [ "$result" = "FAILED" ]; then
        echo "    --- last 40 lines of $log ---"
        tail -40 "$log" | sed 's/^/    /'
    fi
done

{
    echo ""
    if [ "$FAILED" -ne 0 ]; then
        echo "**$FAILED of ${#JOBS[@]} job(s) failed:$FAILED_JOBS**${TOOLCHAIN_NOTE:+ — $TOOLCHAIN_NOTE}"
    else
        echo "**all ${#JOBS[@]} job(s) passed**${TOOLCHAIN_NOTE:+ — $TOOLCHAIN_NOTE}"
    fi
} >> "$SUMMARY"

echo
cat "$SUMMARY"
echo
if [ "$FAILED" -ne 0 ]; then
    echo "==> $FAILED job(s) failed"
    exit 1
fi
echo "==> all ${#JOBS[@]} job(s) passed"
