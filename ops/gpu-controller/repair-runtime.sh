#!/usr/bin/env bash
set -euo pipefail

STATE_FILE=${GPU_CONTROLLER_STATE_FILE:-/opt/swade-gpu-controller/worker.env}
SSH_KEY=${GPU_SSH_KEY:-/root/.ssh/swade_gpu_controller_ed25519}

if [ ! -f "$STATE_FILE" ]; then
  echo "ERROR: GPU controller state not found: $STATE_FILE" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$STATE_FILE"
GPU_PORT=${GPU_PORT:-22}
GPU_USER=${GPU_USER:-root}
: "${GPU_HOST:?GPU_HOST missing from state}"

if [ ! -f "$SSH_KEY" ]; then
  echo "ERROR: GPU controller SSH key not found: $SSH_KEY" >&2
  exit 3
fi

echo "Repairing NVIDIA container runtime on ${GPU_USER}@${GPU_HOST}:${GPU_PORT} ..."

ssh \
  -i "$SSH_KEY" \
  -p "$GPU_PORT" \
  -o BatchMode=yes \
  -o ConnectTimeout=15 \
  -o StrictHostKeyChecking=accept-new \
  "${GPU_USER}@${GPU_HOST}" \
  "sudo -n bash -s" <<'REMOTE'
set -u

export DEBIAN_FRONTEND=noninteractive

echo "=== HOST GPU ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available on the GPU worker." >&2
  exit 10
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== DOCKER ==="
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is missing; installing Docker and Compose."
  apt-get update
  apt-get install -y docker.io docker-compose-v2 curl ca-certificates gnupg || true
fi

systemctl start containerd docker 2>/dev/null || true
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not available." >&2
  systemctl status docker --no-pager -l 2>/dev/null || true
  exit 11
fi
docker --version
docker compose version

echo "=== NVIDIA CONTAINER TOOLKIT ==="
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "Installing NVIDIA Container Toolkit from NVIDIA's official repository."

  if ! command -v curl >/dev/null 2>&1 || ! command -v gpg >/dev/null 2>&1; then
    apt-get update
    apt-get install -y curl ca-certificates gnupg || true
  fi

  mkdir -p /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

  apt-get update
  # Some Hostinger GPU images can have an unrelated AppArmor package left in a
  # partially configured state. Let apt finish what it can, then verify the
  # toolkit itself explicitly instead of trusting apt's final exit code.
  apt-get install -y nvidia-container-toolkit || true
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "ERROR: nvidia-container-toolkit is still unavailable after install attempt." >&2
  dpkg -l | grep -E 'nvidia-container|apparmor' || true
  exit 12
fi

nvidia-ctk --version || true

echo "=== CONFIGURE NVIDIA RUNTIME FOR DOCKER ==="
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
sleep 2

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker failed after NVIDIA runtime configuration." >&2
  systemctl status docker --no-pager -l 2>/dev/null || true
  exit 13
fi

echo "=== VERIFY B200 INSIDE DOCKER ==="
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 \
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

REMOTE

echo "NVIDIA container runtime repair completed successfully."
