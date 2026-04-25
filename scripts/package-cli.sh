#!/usr/bin/env bash
# scripts/package-cli.sh — Build the `macmlx` CLI in Release configuration
# and package it as a tarball suitable for a Homebrew tap formula
# (issue #20).
#
# Inputs:
#   GITHUB_REF_NAME   e.g. "v0.3.8"; falls back to current git tag.
# Outputs:
#   dist/macmlx-${TAG}-arm64.tar.gz       binary tarball
#   dist/macmlx-${TAG}-arm64.tar.gz.sha256
#
# The tarball contains a single top-level `macmlx` executable so the
# Homebrew formula can `bin.install "macmlx"` without unpacking
# directory layers.

set -euo pipefail

CLI_NAME="macmlx"
PACKAGE_PATH="macmlx-cli"
TAG="${GITHUB_REF_NAME:-$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0-dev")}"
ARCH="arm64"
TARBALL_BASENAME="${CLI_NAME}-${TAG}-${ARCH}"
TARBALL_NAME="${TARBALL_BASENAME}.tar.gz"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "error: package-cli.sh must run on an Apple Silicon host (got $(uname -m))." >&2
    exit 1
fi

echo "==> Building ${CLI_NAME} ${TAG} (Release, arm64)"

swift build \
    --package-path "$PACKAGE_PATH" \
    --configuration release \
    --arch arm64

BIN_SRC="$(swift build --package-path "$PACKAGE_PATH" --configuration release --arch arm64 --show-bin-path)/${CLI_NAME}"

if [[ ! -x "$BIN_SRC" ]]; then
    echo "error: built binary missing at ${BIN_SRC}" >&2
    exit 1
fi

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
cp "$BIN_SRC" "$STAGE_DIR/${CLI_NAME}"

# Strip debug symbols. Swift stdlib is dynamically linked from the
# system toolchain on macOS 14+, so we only need the executable itself.
strip -S "$STAGE_DIR/${CLI_NAME}" || true

mkdir -p dist
rm -f "dist/${TARBALL_NAME}" "dist/${TARBALL_NAME}.sha256"

tar -czf "dist/${TARBALL_NAME}" -C "$STAGE_DIR" "${CLI_NAME}"

shasum -a 256 "dist/${TARBALL_NAME}" | tee "dist/${TARBALL_NAME}.sha256"

# Emit the bare sha for downstream steps (formula rendering).
SHA256="$(awk '{print $1}' "dist/${TARBALL_NAME}.sha256")"
if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
        echo "MACMLX_CLI_TARBALL=dist/${TARBALL_NAME}"
        echo "MACMLX_CLI_SHA256=${SHA256}"
    } >> "$GITHUB_ENV"
fi

echo "==> Packaged dist/${TARBALL_NAME} (sha256 ${SHA256})"
