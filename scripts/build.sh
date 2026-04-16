#!/usr/bin/env bash
# scripts/build.sh — Archive macMLX.app for distribution.
#
# Inputs: GITHUB_REF_NAME (e.g. "v0.1.0"); falls back to current git tag.
# Outputs: build/macMLX.xcarchive (then exported to build/export/macMLX.app).

set -euo pipefail

APP_NAME="macMLX"
TAG="${GITHUB_REF_NAME:-$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0-dev")}"
VERSION="${TAG#v}"

echo "==> Archiving $APP_NAME $VERSION"

xcodebuild \
    -project "macMLX/${APP_NAME}.xcodeproj" \
    -scheme "$APP_NAME" \
    -configuration Release \
    -archivePath "build/${APP_NAME}.xcarchive" \
    MARKETING_VERSION="$VERSION" \
    CURRENT_PROJECT_VERSION="${GITHUB_RUN_NUMBER:-1}" \
    CODE_SIGN_IDENTITY="" \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGNING_ALLOWED=NO \
    archive

echo "==> Exporting"

xcodebuild \
    -exportArchive \
    -archivePath "build/${APP_NAME}.xcarchive" \
    -exportPath "build/export" \
    -exportOptionsPlist scripts/ExportOptions.plist

echo "==> Done. App: build/export/${APP_NAME}.app"
