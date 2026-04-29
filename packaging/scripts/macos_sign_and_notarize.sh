#!/usr/bin/env bash
# Template: codesign + notarize BetaSafe.app (and optionally the DMG) for distribution.
# Run on a Mac with Xcode command-line tools and an Apple Developer account.
#
# Required environment (examples):
#   export APPLE_TEAM_ID="XXXXXXXXXX"
#   export APPLE_ID="you@example.com"
#   export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"
#   export SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
#
# Optional entitlements file (e.g. screen recording / accessibility):
#   export ENTITLEMENTS="$PWD/packaging/macos/BetaSafe.entitlements.plist"
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$REPO_ROOT/dist/BetaSafe.app"
DMG="${DMG:-$REPO_ROOT/dist/BetaSafe-macOS.dmg}"

if [[ -z "${SIGNING_IDENTITY:-}" ]]; then
  echo "Set SIGNING_IDENTITY to your 'Developer ID Application: …' certificate name." >&2
  exit 1
fi
if [[ -z "${APPLE_TEAM_ID:-}" ]]; then
  echo "Set APPLE_TEAM_ID." >&2
  exit 1
fi

if [[ ! -d "$APP" ]]; then
  echo "Missing $APP" >&2
  exit 1
fi

ENT_ARGS=()
if [[ -n "${ENTITLEMENTS:-}" && -f "$ENTITLEMENTS" ]]; then
  ENT_ARGS=(--entitlements "$ENTITLEMENTS")
fi

echo "Signing $APP ..."
codesign --deep --force --options runtime \
  --sign "$SIGNING_IDENTITY" \
  "${ENT_ARGS[@]}" \
  "$APP"

if [[ -f "$DMG" ]]; then
  echo "Signing $DMG ..."
  codesign --force --sign "$SIGNING_IDENTITY" "$DMG"
fi

if [[ -n "${APPLE_ID:-}" && -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ]]; then
  echo "Notarizing (app) ..."
  APP_ZIP="$REPO_ROOT/dist/BetaSafe.app.notarize.zip"
  rm -f "$APP_ZIP"
  ditto -c -k --keepParent "$APP" "$APP_ZIP"
  xcrun notarytool submit "$APP_ZIP" --apple-id "$APPLE_ID" --password "$APPLE_APP_SPECIFIC_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" --wait
  rm -f "$APP_ZIP"
  xcrun stapler staple "$APP"
  if [[ -f "$DMG" ]]; then
    echo "Notarizing (dmg) ..."
    xcrun notarytool submit "$DMG" --apple-id "$APPLE_ID" --password "$APPLE_APP_SPECIFIC_PASSWORD" \
      --team-id "$APPLE_TEAM_ID" --wait
    xcrun stapler staple "$DMG"
  fi
else
  echo "Skipping notarization (set APPLE_ID and APPLE_APP_SPECIFIC_PASSWORD to enable)."
fi

echo "Done."
