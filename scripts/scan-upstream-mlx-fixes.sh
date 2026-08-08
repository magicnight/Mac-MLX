#!/usr/bin/env bash
#
# List upstream mlx correctness fixes that our controlled fork does NOT carry.
#
# WHY THIS EXISTS
#
# MacMLXCore pins a fork of mlx-swift (see the comment on the `magicnight/
# mlx-swift` dependency in MacMLXCore/Package.swift). Pinning by revision buys
# reproducibility, and costs us the thing a version range gives for free:
# upstream's correctness fixes stop flowing in. Nothing warns us — the build is
# green, the tests are green, and the shipped binary quietly runs unpatched
# kernels.
#
# That is not hypothetical. ml-explore/mlx#3810 (the NAX split-K GEMM reading
# bf16 inputs as float — silent garbage, no crash, no NaN) sat in mlx v0.32.0
# for a month while our DMG shipped without it. We only found out because a
# stranger mentioned it in passing on an unrelated issue thread.
#
# So: run this before cutting a release, and read the output. It does not
# decide anything — it surfaces candidates for a human to judge.
#
# WHAT IT DOES
#
#   1. Reads the pinned mlx-swift revision out of MacMLXCore/Package.swift.
#   2. Resolves which mlx commit that revision's submodule points at.
#   3. Walks the fork's mlx history until it finds a commit that is an ancestor
#      of upstream main — that is the fork point. Everything above it is ours.
#   4. Lists upstream commits since the fork point that touch the Metal backend
#      and read like fixes, minus the ones we already cherry-picked.
#
# Ancestry is decided with the compare API's `ahead_by`, NOT by asking whether
# upstream can resolve the SHA: GitHub forks share an object store, so
# `repos/ml-explore/mlx/commits/<our-fork-only-sha>` answers 200 and would make
# every fork commit look upstream.
#
# Requires: gh (authenticated), python3.

set -euo pipefail

UPSTREAM_REPO="ml-explore/mlx"
# Where fixes that can silently corrupt inference live. Both fixes we know of
# landed here: #3498 (rope.cpp) and #3810 (matmul.cpp, jit_kernels.cpp,
# kernels.h). Widen deliberately, not by reflex — every extra path is noise a
# human has to read past.
WATCH_PATH="mlx/backend/metal"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_MANIFEST="$REPO_ROOT/MacMLXCore/Package.swift"

command -v gh >/dev/null 2>&1 || { echo "error: gh not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "error: python3 not found" >&2; exit 1; }
[[ -f "$CORE_MANIFEST" ]] || { echo "error: $CORE_MANIFEST not found" >&2; exit 1; }

# --- 1. the pinned fork revision -------------------------------------------

FORK_URL="$(grep -oE 'https://github\.com/[^"]+/mlx-swift\.git' "$CORE_MANIFEST" | head -1 || true)"
if [[ -z "$FORK_URL" || "$FORK_URL" == *"ml-explore"* ]]; then
    echo "No mlx-swift fork is pinned in MacMLXCore/Package.swift."
    echo "If the dependency is back on upstream, this scan has nothing to do — delete it."
    exit 0
fi
FORK_SLUG="$(sed -E 's#https://github\.com/([^/]+/[^/]+)\.git#\1#' <<<"$FORK_URL")"

FORK_REVISION="$(grep -A3 "$FORK_URL" "$CORE_MANIFEST" \
    | grep -oE 'revision: "[0-9a-f]{40}"' | head -1 | grep -oE '[0-9a-f]{40}' || true)"
[[ -n "$FORK_REVISION" ]] || { echo "error: could not read the pinned revision" >&2; exit 1; }

echo "mlx-swift fork : $FORK_SLUG @ ${FORK_REVISION:0:12}"

# --- 2. the mlx commit its submodule points at ------------------------------

MLX_SHA="$(gh api "repos/$FORK_SLUG/contents/Source/Cmlx?ref=$FORK_REVISION" \
    --jq '.[] | select(.name=="mlx") | .sha' 2>/dev/null || true)"
[[ -n "$MLX_SHA" ]] || { echo "error: could not resolve the mlx submodule pointer" >&2; exit 1; }

MLX_FORK_SLUG="$(sed -E 's#(.*)/mlx-swift#\1/mlx#' <<<"$FORK_SLUG")"
echo "mlx submodule  : $MLX_FORK_SLUG @ ${MLX_SHA:0:12}"

# --- 3. walk down to the fork point ----------------------------------------

CARRIED_PRS=""
BASE_SHA=""
BASE_DATE=""

while read -r sha date subject; do
    ahead="$(gh api "repos/$UPSTREAM_REPO/compare/main...$sha" --jq '.ahead_by' 2>/dev/null || echo "")"
    if [[ "$ahead" == "0" ]]; then
        BASE_SHA="$sha"
        BASE_DATE="$date"
        break
    fi
    # Ours. Record which upstream PR it back-ported, if the message says.
    pr="$(grep -oE '#[0-9]+' <<<"$subject" | head -1 || true)"
    [[ -n "$pr" ]] && CARRIED_PRS+="${pr} "
    echo "  carries      : ${sha:0:12}  $subject"
done < <(gh api "repos/$MLX_FORK_SLUG/commits?sha=$MLX_SHA&per_page=20" \
    --jq '.[] | .sha + " " + .commit.committer.date + " " + (.commit.message|split("\n")[0])')

[[ -n "$BASE_SHA" ]] || { echo "error: no fork point found in the last 20 commits" >&2; exit 1; }
echo "fork point     : ${BASE_SHA:0:12} ($BASE_DATE)"
echo

# --- 4. upstream fixes since then, minus what we carry ----------------------

gh api --paginate \
    "repos/$UPSTREAM_REPO/commits?path=$WATCH_PATH&since=$BASE_DATE" \
    --jq '.[] | (.sha[0:12]) + "\t" + (.commit.committer.date[0:10]) + "\t" + (.commit.message|split("\n")[0])' \
    | CARRIED="$CARRIED_PRS" WATCH="$WATCH_PATH" python3 -c '
import os, re, sys

carried = set(os.environ.get("CARRIED", "").split())
# Titles that read like a correctness fix. Deliberately generous: a false
# positive costs a glance, a false negative costs a silently wrong release.
pattern = re.compile(
    r"\b(fix|fixes|fixed|bug|correct|incorrect|wrong|invalid|garbage|nan|"
    r"overflow|out.of.bounds|race|crash|regression)\b", re.I)

rows, skipped, unparsed = [], [], []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    # maxsplit=2: a commit subject is copied verbatim from GitHub and may
    # contain tabs. Splitting on every tab would yield >3 fields and drop the
    # row — silently losing exactly the kind of commit this script exists to
    # surface. The subject is the last field, so it keeps whatever it contains.
    parts = line.split("\t", 2)
    if len(parts) != 3:
        # Never swallow a row. A scanner that quietly discards input is worse
        # than no scanner, because it reads as "nothing to see here".
        unparsed.append(line)
        continue
    sha, date, subject = parts
    if not pattern.search(subject):
        continue
    pr = re.search(r"#\d+", subject)
    if pr and pr.group(0) in carried:
        skipped.append((sha, date, subject))
        continue
    rows.append((sha, date, subject))

if unparsed:
    print(f"WARNING: {len(unparsed)} row(s) could not be parsed and were NOT")
    print("scanned. Read them by hand — this tool must never lose a commit:")
    for line in unparsed:
        print(f"  {line!r}")
    print()

if skipped:
    print("Already carried by the fork:")
    for sha, date, subject in skipped:
        print(f"  {date}  {sha}  {subject}")
    print()

watch = os.environ.get("WATCH", "")
if not rows:
    print(f"No un-carried fix-shaped commits under {watch}. Nothing to weigh.")
    sys.exit(0)

print(f"{len(rows)} un-carried fix-shaped commit(s) under {watch}:")
print()
for sha, date, subject in rows:
    print(f"  {date}  {sha}  {subject}")
print()
print("These are CANDIDATES, not verdicts. For each, ask: can it reach us?")
print("  - Does the path run on Apple silicon at all (some are CUDA-only)?")
print("  - Is it JIT-only? mlx-swift builds with MLX_METAL_JIT=ON, so JIT-only")
print("    fixes reach us even when the Python wheel is unaffected.")
print("  - What dtype/shape window triggers it — can our models land in it?")
print("Anything that can produce wrong numbers rather than a crash deserves a")
print("cherry-pick onto the fork before release, plus a regression test.")
'
