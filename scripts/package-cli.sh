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
# The tarball contains the `macmlx` executable AND the mlx-swift
# `mlx-swift_Cmlx.bundle` resource bundle (which carries
# default.metallib) side by side. MLX resolves its Metal library via
# the bundle next to the executable — a bare binary aborts with
# "Failed to load the default metallib" on the first inference, so the
# bundle MUST travel with the binary. The Homebrew formula installs
# both into libexec and exposes a bin shim (see Formula/macmlx.rb).
#
# Build goes through xcodebuild (NOT bare `swift build`): only the
# Xcode build pipeline runs mlx-swift's Metal shader compilation and
# emits the resource bundle. A bare SPM command-line build produces no
# metallib at all.

set -euo pipefail

CLI_NAME="macmlx"
PACKAGE_PATH="macmlx-cli"
CMLX_BUNDLE="mlx-swift_Cmlx.bundle"
TAG="${GITHUB_REF_NAME:-$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0-dev")}"
ARCH="arm64"
TARBALL_BASENAME="${CLI_NAME}-${TAG}-${ARCH}"
TARBALL_NAME="${TARBALL_BASENAME}.tar.gz"
DERIVED_DATA="${PACKAGE_PATH}/.xcodebuild"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "error: package-cli.sh must run on an Apple Silicon host (got $(uname -m))." >&2
    exit 1
fi

echo "==> Building ${CLI_NAME} ${TAG} (xcodebuild Release, arm64)"

# Run from inside the package directory so xcodebuild discovers the
# SPM package (there is no .xcworkspace/.xcodeproj to point at).
(
    cd "$PACKAGE_PATH"
    # Capture xcodebuild's REAL exit via PIPESTATUS[0] — the `grep` is only for log
    # brevity and must never mask a build failure (the old `| grep … || true` turned
    # every failed build into success, packaging stale Release products). `set +e`
    # around the pipeline keeps a grep "no match" — or the build failure itself —
    # from tripping errexit before we can read PIPESTATUS.
    set +e
    xcodebuild \
        -scheme "$CLI_NAME" \
        -configuration Release \
        -destination 'platform=macOS,arch=arm64' \
        -skipPackagePluginValidation \
        -derivedDataPath ".xcodebuild" \
        build \
        | grep -E "BUILD (SUCCEEDED|FAILED)|error:"
    xcodebuild_status=${PIPESTATUS[0]}
    set -e
    if [[ "$xcodebuild_status" -ne 0 ]]; then
        echo "error: xcodebuild failed (exit ${xcodebuild_status}); refusing to package stale Release products." >&2
        exit "$xcodebuild_status"
    fi
)

PRODUCTS_DIR="${DERIVED_DATA}/Build/Products/Release"
BIN_SRC="${PRODUCTS_DIR}/${CLI_NAME}"
BUNDLE_SRC="${PRODUCTS_DIR}/${CMLX_BUNDLE}"

if [[ ! -x "$BIN_SRC" ]]; then
    echo "error: built binary missing at ${BIN_SRC}" >&2
    exit 1
fi
if [[ ! -f "${BUNDLE_SRC}/Contents/Resources/default.metallib" ]]; then
    echo "error: ${CMLX_BUNDLE} missing default.metallib at ${BUNDLE_SRC} —" >&2
    echo "       inference would abort at runtime; refusing to package." >&2
    exit 1
fi

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
cp "$BIN_SRC" "$STAGE_DIR/${CLI_NAME}"
cp -R "$BUNDLE_SRC" "$STAGE_DIR/${CMLX_BUNDLE}"

# Strip debug symbols. Swift stdlib is dynamically linked from the
# system toolchain on macOS 14+, so we only need the executable itself.
strip -S "$STAGE_DIR/${CLI_NAME}" || true

# Package self-check: the staged layout must at least run --version.
# (Real inference needs a downloaded model, so it can't run in CI; the
# metallib presence check above is the packaging-side guard for that.)
if ! "$STAGE_DIR/${CLI_NAME}" --version >/dev/null 2>&1; then
    echo "error: staged ${CLI_NAME} --version failed; refusing to package." >&2
    exit 1
fi

mkdir -p dist
rm -f "dist/${TARBALL_NAME}" "dist/${TARBALL_NAME}.sha256"

tar -czf "dist/${TARBALL_NAME}" -C "$STAGE_DIR" "${CLI_NAME}" "${CMLX_BUNDLE}"

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
