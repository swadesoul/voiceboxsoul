# Swade Voice Engine on Hostinger GPU

This production profile turns `voiceboxsoul` into the shared voice service for SSSBrain, SWIDEO, BlackTales, and other approved Swade Soul apps.

## Architecture

- Hostinger NVIDIA GPU: inference only.
- Voicebox FastAPI: private container port `17493`.
- Nginx gateway: public host port `17601` with Bearer-token authentication.
- Persistent working state: `/opt/swade-voice/data` and `/opt/swade-voice/output`.
- Backblaze B2: external backup/restore target for voice profiles, SQLite state, samples, and generated audio.
- HuggingFace model cache: local only; models can be redownloaded.

## First deployment

```bash
sudo mkdir -p /opt
cd /opt
sudo git clone --branch hostinger-nvidia-voice-engine https://github.com/swadesoul/voiceboxsoul.git
cd voiceboxsoul
sudo bash ops/hostinger-gpu/bootstrap.sh
```

The bootstrap script:

1. Verifies `nvidia-smi`.
2. Verifies Docker.
3. Creates `/opt/swade-voice` directories.
4. Generates `.env.hostinger` with a random API token if it does not exist.
5. Verifies NVIDIA passthrough inside Docker.
6. Builds and starts Voicebox + gateway.
7. Waits for `/health` to succeed.

## Firewall

Do not expose container port `17493` publicly.

Expose `17601` only after the gateway is healthy. Prefer restricting the source IPs to the VPS/SSSBrain service when practical.

## API usage

Health check does not require a token:

```bash
curl http://GPU_HOST:17601/health
```

All other routes require:

```http
Authorization: Bearer <VOICEBOX_API_TOKEN>
```

Example:

```bash
curl http://GPU_HOST:17601/profiles \
  -H "Authorization: Bearer $VOICEBOX_API_TOKEN"
```

## Backblaze B2 backup

Install and configure `rclone` using a restricted Backblaze B2 application key. Do not use the B2 master key.

Set these variables outside the repo:

```bash
export B2_BUCKET=YOUR_BUCKET
export B2_REMOTE=b2
export B2_PREFIX=voice-engine
```

Run:

```bash
bash ops/hostinger-gpu/backup-to-b2.sh
```

Recommended: run backup before manually stopping/destroying a GPU instance and on a recurring schedule while production voice profiles are changing.

## 30% Hostinger credit rule

Hostinger GPU instances are ephemeral if the account runs out of GPU credit. Operational policy for Swade Voice Engine:

- At 30% remaining GPU credit: alert the operator and immediately verify the latest B2 backup.
- At 20%: block nonessential batch generation and preserve capacity for active projects.
- At 10%: stop new generation, flush queues, run a final B2 backup, and prepare to shut down the GPU cleanly.
- Never use the GPU filesystem as the only copy of a voice profile, sample, character-to-voice mapping, or final audio asset.

The actual Hostinger credit watcher should live in SSSBrain/VPS orchestration so it continues running even if the GPU is stopped.

## Integration rule

SSSBrain should be the control plane. SWIDEO and BlackTales should use separate app credentials at the SSSBrain gateway layer rather than sharing a single downstream credential. This prevents one app from impersonating another and preserves per-app authorization and audit logs.
