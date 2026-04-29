#!/usr/bin/env bash
# Build a drag-and-drop DMG from dist/BetaSafe.app (after PyInstaller on macOS).
# Optional: install create-dmg (brew install create-dmg) for a branded layout + Applications link.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$REPO_ROOT/dist/BetaSafe.app"
DMG_OUT="${1:-$REPO_ROOT/dist/BetaSafe-macOS.dmg}"

if [[ ! -d "$APP" ]]; then
  echo "Missing $APP — run packaging/scripts/build_pyinstaller.sh on macOS first." >&2
  exit 1
fi

BG="$REPO_ROOT/packaging/assets/dmg/background.png"
if [[ ! -f "$BG" ]]; then
  BG="$REPO_ROOT/packaging/assets/dmg/backgound.png"
fi

rm -f "$DMG_OUT"

if command -v create-dmg >/dev/null 2>&1; then
  echo "Using create-dmg..."
  STAGE="$(mktemp -d "${TMPDIR:-/tmp}/betasafe-dmg-XXXXXX")"
  cp -R "$APP" "$STAGE/"
  cleanup() { rm -rf "$STAGE"; }
  trap cleanup EXIT

  EXTRA=()
  if [[ -f "$BG" ]]; then
    EXTRA+=(--background "$BG")
    echo "Background: $BG"
  else
    echo "No background.png (or backgound.png) — DMG will use a plain layout."
  fi
  # Usage: create-dmg [options] <output.dmg> <source_folder/>
  create-dmg \
    --volname "BetaSafe" \
    --window-pos 200 120 \
    --window-size 660 420 \
    --icon-size 110 \
    --icon "BetaSafe.app" 160 200 \
    --hide-extension "BetaSafe.app" \
    --app-drop-link 480 200 \
    "${EXTRA[@]}" \
    "$DMG_OUT" \
    "$STAGE"
else
  echo "create-dmg not installed; using plain compressed DMG (hdiutil)."
  echo "Tip: brew install create-dmg for branded layout + Applications shortcut."
  TMP_DMG="$REPO_ROOT/dist/.betasafe-tmp.dmg"
  rm -f "$TMP_DMG"
  hdiutil create -volname "BetaSafe" -srcfolder "$APP" -ov -format UDRW "$TMP_DMG"
  hdiutil convert "$TMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$DMG_OUT"
  rm -f "$TMP_DMG"
fi

echo "Wrote $DMG_OUT"
