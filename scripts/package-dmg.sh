#!/usr/bin/env bash
# scripts/package-dmg.sh — Wrap build/export/macMLX.app into a DMG.
#
# Inputs: GITHUB_REF_NAME (e.g. "v0.1.0"); falls back to current git tag.
# Outputs: dist/macMLX-vX.Y.Z.dmg

set -euo pipefail

APP_NAME="macMLX"
TAG="${GITHUB_REF_NAME:-$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0-dev")}"
DMG_NAME="${APP_NAME}-${TAG}.dmg"

if [[ ! -d "build/export/${APP_NAME}.app" ]]; then
    echo "error: build/export/${APP_NAME}.app missing — run scripts/build.sh first." >&2
    exit 1
fi

mkdir -p dist
rm -f "dist/${DMG_NAME}"

if ! command -v create-dmg >/dev/null 2>&1; then
    echo "error: create-dmg not installed. Run: brew install create-dmg" >&2
    exit 1
fi

create-dmg \
    --volname "${APP_NAME} ${TAG}" \
    --window-size 600 400 \
    --icon-size 128 \
    --icon "${APP_NAME}.app" 150 185 \
    --app-drop-link 450 185 \
    "dist/${DMG_NAME}" \
    "build/export/${APP_NAME}.app"

# Compute SHA256 for the release notes / CI artifact.
shasum -a 256 "dist/${DMG_NAME}" | tee "dist/${DMG_NAME}.sha256"

echo "==> Built dist/${DMG_NAME}"
