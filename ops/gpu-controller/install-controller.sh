#!/usr/bin/env bash
set -euo pipefail

SRC_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BIN=/usr/local/bin/gpuctl
STATE_DIR=/opt/swade-gpu-controller

install -d -m 700 "$STATE_DIR"
install -m 755 "$SRC_DIR/gpuctl" "$BIN"

cat <<'EOF'
Swade GPU Controller installed.

Commands:
  gpuctl register <GPU_IP>
  gpuctl status
  gpuctl provision
  gpuctl health
  gpuctl drain
  gpuctl safe-to-destroy
  gpuctl clear

Recommended flow:
  1. Manually deploy Hostinger GPU.
  2. gpuctl register <GPU_IP>
  3. gpuctl provision
  4. generate voice jobs
  5. gpuctl drain
  6. gpuctl safe-to-destroy
  7. destroy the GPU in Hostinger only after SAFE_TO_DESTROY=YES
  8. gpuctl clear
EOF
