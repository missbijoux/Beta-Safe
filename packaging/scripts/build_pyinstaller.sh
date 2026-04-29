#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "Repo: $REPO_ROOT"
echo "Running PyInstaller (onedir)..."

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "PyInstaller not found. Run: pip install -r requirements-build.txt" >&2
  exit 1
fi

pyinstaller -y --clean "$REPO_ROOT/packaging/pyinstaller/betasafe.spec"

echo "Done. Output under dist/"
if [[ -d "$REPO_ROOT/dist/BetaSafe.app" ]]; then
  echo "macOS app: dist/BetaSafe.app"
elif [[ -f "$REPO_ROOT/dist/BetaSafe/BetaSafe.exe" ]]; then
  echo "Windows exe: dist/BetaSafe/BetaSafe.exe"
fi
