#!/usr/bin/env bash
# scripts/generate_sparkle_keys.sh — One-time EdDSA keypair for Sparkle updates.
#
# Run this AFTER the Sparkle SPM dep has been added to the macMLX Xcode target.
# Outputs:
#   - The public key (base64, 44 chars) to paste into macMLX/macMLX/Info.plist
#     as the SUPublicEDKey string.
#   - The private key file written to ~/.mac-mlx/sparkle_private_key.pem.
#     NEVER commit this file. Copy its contents into the GitHub repo Secret
#     named SPARKLE_PRIVATE_KEY at:
#     https://github.com/magicnight/Mac-MLX/settings/secrets/actions
#
# Safe to re-run: refuses to overwrite an existing private key.

set -euo pipefail

PRIVATE_KEY_PATH="${HOME}/.mac-mlx/sparkle_private_key.pem"

# --- Locate generate_keys ---

find_generate_keys() {
    # Sparkle ships the tool inside the SPM checkout's bin/ after resolution.
    local hit
    hit=$(find ~/Library/Developer/Xcode/DerivedData -type f -name generate_keys -perm +111 2>/dev/null | head -1)
    if [[ -n "$hit" ]]; then
        echo "$hit"
        return
    fi
    # Homebrew cask fallback.
    hit=$(find /opt/homebrew/Caskroom/sparkle -type f -name generate_keys 2>/dev/null | head -1)
    if [[ -n "$hit" ]]; then
        echo "$hit"
        return
    fi
    echo ""
}

GENERATE_KEYS=$(find_generate_keys)

if [[ -z "$GENERATE_KEYS" ]]; then
    cat <<'HINT' >&2
error: could not find Sparkle's generate_keys tool.

Either:
  1. Add Sparkle to the macMLX Xcode target as an SPM dependency
     (File → Add Package Dependencies → https://github.com/sparkle-project/Sparkle
     → from 2.0.0 → add to macMLX target). Then xcodebuild -resolvePackageDependencies
     at least once and re-run this script.
  OR
  2. brew install --cask sparkle    # bundles the tool

HINT
    exit 1
fi

echo "Using generate_keys at: $GENERATE_KEYS"

# --- Refuse to clobber existing key ---

if [[ -f "$PRIVATE_KEY_PATH" ]]; then
    cat <<HINT >&2

warning: A Sparkle private key already exists at:
  $PRIVATE_KEY_PATH

Keep it unless you're rotating. To rotate:
  - Move the old file aside (mv "$PRIVATE_KEY_PATH" "$PRIVATE_KEY_PATH.old")
  - Re-run this script
  - Update the SPARKLE_PRIVATE_KEY GitHub Secret
  - Update the SUPublicEDKey string in Info.plist on the NEXT release tag
    (old installations won't validate signatures made with the new key —
    users will need to reinstall manually one last time)

Aborting. No files changed.
HINT
    exit 2
fi

# --- Generate ---

mkdir -p "$(dirname "$PRIVATE_KEY_PATH")"
chmod 700 "$(dirname "$PRIVATE_KEY_PATH")"

# generate_keys on Sparkle 2.x writes to the macOS Keychain AND prints both
# keys. We extract them from the output.
OUTPUT=$("$GENERATE_KEYS")
echo "$OUTPUT"

PUBLIC_KEY=$(echo "$OUTPUT" | grep -oE '[A-Za-z0-9+/]{43}=' | tail -1 || true)
PRIVATE_KEY=$(echo "$OUTPUT" | grep -oE 'SUSignUpdate -s [A-Za-z0-9+/=]+' | sed 's|SUSignUpdate -s ||' || true)

if [[ -z "$PUBLIC_KEY" ]]; then
    echo "error: could not extract public key from generate_keys output." >&2
    echo "Re-read the output above and paste the values manually." >&2
    exit 3
fi

# Persist private key to disk for CI usage.
if [[ -n "$PRIVATE_KEY" ]]; then
    printf '%s' "$PRIVATE_KEY" > "$PRIVATE_KEY_PATH"
    chmod 600 "$PRIVATE_KEY_PATH"
fi

echo
echo "=========================================================="
echo " Public key (paste into macMLX/macMLX/Info.plist as"
echo " SUPublicEDKey, replacing REPLACE_WITH_OUTPUT_OF_GENERATE_KEYS):"
echo "=========================================================="
echo
echo "  $PUBLIC_KEY"
echo

if [[ -f "$PRIVATE_KEY_PATH" ]]; then
    cat <<HINT
==========================================================
 Private key saved to:
==========================================================

  $PRIVATE_KEY_PATH

 Next step — set the GitHub Secret:

   gh secret set SPARKLE_PRIVATE_KEY \\
       --repo magicnight/Mac-MLX \\
       --body "\$(cat '$PRIVATE_KEY_PATH')"

 Or via the web UI at:
   https://github.com/magicnight/Mac-MLX/settings/secrets/actions

 NEVER commit this file or paste its contents into chat.
HINT
fi
