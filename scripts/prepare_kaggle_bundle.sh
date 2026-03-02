#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/dist}"
BUNDLE_NAME="AIMO3_source_bundle"

mkdir -p "$OUT_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

rsync -a \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'dist' \
  --exclude '*.pyc' \
  "$ROOT_DIR/" "$TMP_DIR/$BUNDLE_NAME/"

cd "$TMP_DIR"
zip -qr "$OUT_DIR/${BUNDLE_NAME}.zip" "$BUNDLE_NAME"

echo "Created: $OUT_DIR/${BUNDLE_NAME}.zip"
