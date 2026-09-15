#!/usr/bin/env bash
set -euo pipefail

VOICE_ROOT=${VOICE_ENGINE_ROOT:-/opt/swade-voice}
B2_REMOTE=${B2_REMOTE:-b2}
B2_BUCKET=${B2_BUCKET:?Set B2_BUCKET in the host environment}
B2_PREFIX=${B2_PREFIX:-voice-engine}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TMP_DIR=$(mktemp -d)
ARCHIVE="$TMP_DIR/swade-voice-$STAMP.tar.gz"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if ! command -v rclone >/dev/null 2>&1; then
  echo "ERROR: rclone is required for Backblaze B2 backup."
  exit 1
fi

if [ ! -d "$VOICE_ROOT/data" ]; then
  echo "ERROR: Voice data directory not found: $VOICE_ROOT/data"
  exit 1
fi

# Snapshot all persistent Voicebox state. Model cache is intentionally excluded;
# HuggingFace models can be re-downloaded and would make backups unnecessarily large.
tar -C "$VOICE_ROOT" -czf "$ARCHIVE" data output

sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

rclone copy "$ARCHIVE" "$B2_REMOTE:$B2_BUCKET/$B2_PREFIX/backups/"
rclone copy "$ARCHIVE.sha256" "$B2_REMOTE:$B2_BUCKET/$B2_PREFIX/backups/"

# Keep a continuously mirrored copy for fast restore in addition to timestamped archives.
rclone sync "$VOICE_ROOT/data" "$B2_REMOTE:$B2_BUCKET/$B2_PREFIX/live/data" --checksum
rclone sync "$VOICE_ROOT/output" "$B2_REMOTE:$B2_BUCKET/$B2_PREFIX/live/output" --checksum

echo "Backup completed: $STAMP"
