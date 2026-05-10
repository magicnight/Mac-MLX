#!/usr/bin/env bash
# scripts/render-formula.sh — Render Formula/macmlx.rb for the current
# release into dist/macmlx.rb (issue #20).
#
# Reads:
#   GITHUB_REF_NAME      e.g. "v0.3.8"; falls back to the latest git tag.
#   MACMLX_CLI_SHA256    sha256 of the CLI tarball; falls back to
#                        reading dist/<tarball>.sha256 if present.
# Writes:
#   dist/macmlx.rb       rendered formula, ready to commit to the tap.
#
# Idempotent — safe to re-run. Exits non-zero if the sha256 cannot be
# resolved (better than shipping a formula that points at nothing).

set -euo pipefail

REPO_OWNER="${MACMLX_REPO_OWNER:-magicnight}"
REPO_NAME="${MACMLX_REPO_NAME:-mac-mlx}"
ARCH="arm64"
CLI_NAME="macmlx"
TEMPLATE="Formula/macmlx.rb"

TAG="${GITHUB_REF_NAME:-$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0-dev")}"
VERSION="${TAG#v}"
TARBALL_NAME="${CLI_NAME}-${TAG}-${ARCH}.tar.gz"
URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${TAG}/${TARBALL_NAME}"

SHA256="${MACMLX_CLI_SHA256:-}"
if [[ -z "$SHA256" && -f "dist/${TARBALL_NAME}.sha256" ]]; then
    SHA256="$(awk '{print $1}' "dist/${TARBALL_NAME}.sha256")"
fi

if [[ -z "$SHA256" ]]; then
    echo "error: sha256 unknown — set MACMLX_CLI_SHA256 or run scripts/package-cli.sh first." >&2
    exit 1
fi

if [[ ! -f "$TEMPLATE" ]]; then
    echo "error: ${TEMPLATE} missing." >&2
    exit 1
fi

mkdir -p dist

# sed -i differs on macOS vs GNU; use a portable form.
sed \
    -e "s|@@VERSION@@|${VERSION}|g" \
    -e "s|@@URL@@|${URL}|g" \
    -e "s|@@SHA256@@|${SHA256}|g" \
    "$TEMPLATE" > "dist/macmlx.rb"

echo "==> Rendered dist/macmlx.rb (version ${VERSION}, sha256 ${SHA256})"
