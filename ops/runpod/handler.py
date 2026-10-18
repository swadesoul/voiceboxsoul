"""SWADE Voice Engine worker for RunPod serverless.

Runs the open-source Voicebox server (swadesoul/voiceboxsoul, MIT) inside the worker and exposes it
as queue jobs so the GPU only bills while a job runs. Models and voice profiles live on the network
volume, so nothing is lost when the worker scales to zero.

Job input:
  {"op": "health"}
  {"op": "raw", "method": "GET", "path": "/models/status", "json": {...}}          # admin/debug
  {"op": "transcribe", "audio_url": "..."}
  {"op": "preload", "models": ["chatterbox-turbo", "qwen-tts-1.7B", "whisper-base"]}   # once per new volume
  {"op": "speak", "text": "...", "engine": "chatterbox_turbo", "language": "en", "instruct": "...",
   "profile": {"key": "unique-stable-key", "name": "Display name", "language": "en",
               "samples": [{"url": "https://signed-url", "text": "optional transcript"}]},
   "upload_url": "https://signed-PUT-url (optional)", "format": "mp3"}
No secrets are stored here; sample and upload URLs are short-lived signed URLs supplied per job.
"""
import base64, hashlib, json, os, shutil, subprocess, sys, tempfile, time, urllib.request

import requests
import runpod

VOL = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/root/swade-voice"
DATA = f"{VOL}/voicebox-data"
os.makedirs(DATA, exist_ok=True)
os.environ.setdefault("HF_HOME", f"{VOL}/hf-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
BASE = "http://127.0.0.1:17493"
_server = None


def start_server():
    global _server
    if _server and _server.poll() is None:
        return
    _server = subprocess.Popen(
        [sys.executable, "-m", "backend.main", "--host", "127.0.0.1", "--port", "17493", "--data-dir", DATA],
        cwd="/app", stdout=sys.stdout, stderr=sys.stderr)
    for _ in range(180):
        try:
            if requests.get(f"{BASE}/health", timeout=3).ok:
                return
        except Exception:
            pass
        if _server.poll() is not None:
            raise RuntimeError("voicebox server exited during start")
        time.sleep(1)
    raise RuntimeError("voicebox server did not become healthy")


def call_wait(make_request, what, max_wait=900):
    """Run a request; while Voicebox says 202 (model downloading) wait and retry. Raise with the body on errors."""
    t0 = time.time()
    while True:
        r = make_request()
        # Voicebox reports "model is being downloaded" as 202, but some routes wrap it in a 500.
        waiting = r.status_code == 202 or (r.status_code >= 500 and "being downloaded" in r.text)
        if waiting and time.time() - t0 < max_wait:
            time.sleep(6)
            continue
        if not r.ok or waiting:
            raise RuntimeError(f"{what} {r.status_code}: {r.text[:500]}")
        return r


def fetch(url, suffix):
    path = tempfile.mktemp(suffix=suffix)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return path


def ext_of(url):
    name = url.split("?")[0].rsplit("/", 1)[-1].lower()
    for e in (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"):
        if name.endswith(e):
            return e
    return ".wav"


def to_wav(path):
    """Normalise any sample to mono 24 kHz wav, trimmed to 28 s (Voicebox rejects longer samples), so every engine accepts it."""
    out = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "24000", "-t", "28", out], check=True)
    return out


def transcribe(path):
    def go():
        with open(path, "rb") as f:
            return requests.post(f"{BASE}/transcribe", files={"file": (os.path.basename(path), f, "audio/wav")}, timeout=900)
    return (call_wait(go, "transcribe").json().get("text") or "").strip()


def ensure_profile(p):
    key = str(p["key"])
    if len(key) > 40:  # keep the identity intact inside Voicebox's 100-character name limit
        key = hashlib.sha256(key.encode()).hexdigest()[:24]
    label = (p.get("name") or "voice")[: 100 - len(key) - 3]
    name = f"{label} [{key}]"
    for existing in requests.get(f"{BASE}/profiles", timeout=30).json():
        if existing.get("name") == name:
            return existing["id"], None
    body = {"name": name, "language": p.get("language", "en"), "voice_type": p.get("voice_type", "cloned")}
    for k in ("preset_engine", "preset_voice_id", "design_prompt", "default_engine", "description"):
        if p.get(k):
            body[k] = p[k]
    pid = call_wait(lambda: requests.post(f"{BASE}/profiles", json=body, timeout=60), "create profile").json()["id"]
    texts = []
    try:
        for s in (p.get("samples") or [])[:4]:
            wav = to_wav(fetch(s["url"], ext_of(s["url"])))
            text = (s.get("text") or "").strip() or transcribe(wav)
            if not text:
                continue
            def add():
                with open(wav, "rb") as f:
                    return requests.post(f"{BASE}/profiles/{pid}/samples", files={"file": ("sample.wav", f, "audio/wav")},
                                         data={"reference_text": text[:1000]}, timeout=900)
            call_wait(add, "add sample")
            texts.append({"url_path": s["url"].split("?")[0][-160:], "text": text[:1000]})
        if p.get("samples") and not texts:
            raise RuntimeError("no usable voice sample: every sample was silent or had no transcript")
    except Exception:
        requests.delete(f"{BASE}/profiles/{pid}", timeout=30)
        raise
    return pid, texts


def speak(inp):
    pid, sample_texts = ensure_profile(inp["profile"])
    body = {"profile_id": pid, "text": inp["text"], "language": inp.get("language", "en"),
            "engine": inp.get("engine", "chatterbox_turbo")}
    for k in ("instruct", "seed", "model_size", "max_chunk_chars"):
        if inp.get(k) is not None:
            body[k] = inp[k]
    gen = call_wait(lambda: requests.post(f"{BASE}/generate", json=body, timeout=1800), "generate").json()
    gid, t0 = gen["id"], time.time()
    while gen.get("status") not in ("completed", "failed") and time.time() - t0 < 1700:
        time.sleep(2)
        h = requests.get(f"{BASE}/history/{gid}", timeout=30)
        if h.ok:
            gen = h.json()
    if gen.get("status") != "completed":
        return {"error": f"generation {gen.get('status')}: {str(gen.get('error'))[:400]}"}
    a = requests.get(f"{BASE}/audio/{gid}", timeout=300)
    a.raise_for_status()
    src = tempfile.mktemp(suffix=".wav")
    open(src, "wb").write(a.content)
    fmt = inp.get("format", "mp3")
    out = tempfile.mktemp(suffix="." + fmt)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src]
    cmd += ["-codec:a", "libmp3lame", "-b:a", "192k", out] if fmt == "mp3" else [out]
    subprocess.run(cmd, check=True)
    data = open(out, "rb").read()
    result = {"ok": True, "engine": body["engine"], "duration": gen.get("duration"), "bytes": len(data),
              "seconds_taken": round(time.time() - t0, 1), "profile_id": pid}
    if sample_texts:
        result["sample_texts"] = sample_texts
    if inp.get("upload_url"):
        req = urllib.request.Request(inp["upload_url"], data=data, method="PUT",
                                     headers={"Content-Type": "audio/mpeg" if fmt == "mp3" else "audio/wav",
                                              "x-upsert": "true"})
        with urllib.request.urlopen(req, timeout=300) as up:
            result["uploaded"] = up.status
    elif len(data) < 6_000_000:
        result["audio_base64"] = base64.b64encode(data).decode()
    else:
        result["error"] = "audio too large to return inline; pass upload_url"
    return result


DEFAULT_PRELOAD = ["chatterbox-turbo", "qwen-tts-1.7B", "whisper-base"]


def preload(inp, budget=1700):
    """Download models into the volume's Hugging Face cache once, so later cold starts only load them."""
    wanted = [str(m) for m in (inp.get("models") or DEFAULT_PRELOAD)]
    for m in wanted:
        r = requests.post(f"{BASE}/models/download", json={"model_name": m}, timeout=60)
        if not r.ok:
            raise RuntimeError(f"download {m} {r.status_code}: {r.text[:300]}")
    t0 = time.time()
    while True:
        rows = requests.get(f"{BASE}/models/status", timeout=60).json().get("models", [])
        state = {row.get("model_name"): bool(row.get("downloaded")) for row in rows}
        missing = [m for m in wanted if not state.get(m)]
        if not missing:
            return {"ok": True, "downloaded": wanted, "seconds": round(time.time() - t0, 1), "volume": VOL}
        if time.time() - t0 > budget:
            return {"ok": False, "still_downloading": missing, "seconds": round(time.time() - t0, 1),
                    "hint": "run preload again; finished downloads are kept on the volume"}
        time.sleep(10)


def handler(job):
    inp = job.get("input") or {}
    op = inp.get("op", "speak")
    # Every temp file of this job lives in one directory that is removed afterwards, so a warm
    # worker never fills its disk. RunPod runs one job at a time per worker by default.
    job_dir = tempfile.mkdtemp(prefix="job-")
    tempfile.tempdir = job_dir
    try:
        start_server()
        if op == "preload":
            return preload(inp)
        if op == "health":
            return {"ok": True, "health": requests.get(f"{BASE}/health", timeout=10).json(), "volume": VOL}
        if op == "raw":
            r = requests.request(inp.get("method", "GET"), BASE + inp["path"], json=inp.get("json"), timeout=1700)
            try:
                return {"status": r.status_code, "json": r.json()}
            except Exception:
                return {"status": r.status_code, "text": r.text[:4000]}
        if op == "transcribe":
            return {"ok": True, "text": transcribe(to_wav(fetch(inp["audio_url"], ext_of(inp["audio_url"]))))}
        if op == "speak":
            return speak(inp)
        return {"error": f"unknown op {op}"}
    except Exception as e:  # report, never crash the worker
        return {"error": f"{type(e).__name__}: {str(e)[:600]}"}
    finally:
        tempfile.tempdir = None
        shutil.rmtree(job_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
