#!/usr/bin/env bash
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/voiceboxsoul}
VOICE_ROOT=${VOICE_ENGINE_ROOT:-/opt/swade-voice}
REPO_URL=${REPO_URL:-https://github.com/swadesoul/voiceboxsoul.git}
BRANCH=${BRANCH:-hostinger-nvidia-voice-engine}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA driver not detected. Confirm Hostinger GPU image/driver before continuing."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is required. Install Docker first or deploy from a Hostinger Docker/Coolify image."
  exit 1
fi

mkdir -p "$VOICE_ROOT/data" "$VOICE_ROOT/output" "$VOICE_ROOT/hf-cache"

if [ ! -d "$APP_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
else
  git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
fi

cd "$APP_DIR"

if [ ! -f .env.hostinger ]; then
  cp .env.hostinger.example .env.hostinger
  TOKEN=$(openssl rand -hex 32)
  sed -i "s/replace-with-a-long-random-secret/$TOKEN/" .env.hostinger
  chmod 600 .env.hostinger
  echo "Created .env.hostinger with a generated API token. Save this token in your secrets manager before deleting/replacing the file."
fi

echo "Checking NVIDIA runtime from Docker..."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo "Building and starting Swade Voice Engine..."
docker compose -f docker-compose.hostinger-nvidia.yml --env-file .env.hostinger up -d --build

echo "Waiting for health endpoint..."
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:17601/health >/dev/null; then
    echo "Voice Engine is healthy."
    docker compose -f docker-compose.hostinger-nvidia.yml ps
    exit 0
  fi
  sleep 5
done

echo "ERROR: Voice Engine did not become healthy."
docker compose -f docker-compose.hostinger-nvidia.yml logs --tail=200
exit 1
