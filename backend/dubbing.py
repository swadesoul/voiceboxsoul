"""CLI for Spanish and French dubbing approval samples.

Talks to a running Voicebox. Translation uses whatever
``SWADE_AI_GATEWAY_URL`` the server was started with. This process does
not read OpenAI, Anthropic, or xAI keys.

From the repo root, with Voicebox on port 17493:

    backend/venv/bin/python -m backend.dubbing \\
        --text "The deploy finished cleanly. Please approve this read." \\
        --profile Morgan \\
        --out-dir ./dub-samples

That writes ``sample-es.wav``, ``sample-fr.wav``, and ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:17493"
DEFAULT_LANGUAGES = ("es", "fr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.dubbing",
        description=(
            "Generate es and fr dubbing samples (OmniRoute translate, then voicebox speak) "
            "for caller approval before production use."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  backend/venv/bin/python -m backend.dubbing \\\n"
            '    --text "The deploy finished cleanly. Please approve this read." \\\n'
            "    --profile Morgan \\\n"
            "    --out-dir ./dub-samples\n"
        ),
    )
    parser.add_argument("--text", help="Transcript to dub. Required unless --text-file is set.")
    parser.add_argument("--text-file", type=Path, help="Read the transcript from a UTF-8 file.")
    parser.add_argument(
        "--language",
        action="append",
        choices=["es", "fr"],
        dest="languages",
        help="Target language. Repeat for both. Default: es and fr.",
    )
    parser.add_argument("--profile", help="Voice profile name or id. Falls back to the server default.")
    parser.add_argument(
        "--engine",
        choices=["qwen", "qwen_custom_voice", "luxtts", "chatterbox", "chatterbox_turbo", "tada", "kokoro"],
        help="TTS engine override. Must support the target language.",
    )
    parser.add_argument(
        "--model-size",
        choices=["1.7B", "0.6B", "1B", "3B"],
        help="Engine model size. TADA needs 3B for es/fr. Qwen defaults to 1.7B.",
    )
    parser.add_argument(
        "--gateway-model",
        help="OmniRoute model id. Default on the server is 'auto'.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Voicebox origin. Default {DEFAULT_BASE_URL}.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("dub-samples"), help="Where to write wavs and manifest.json."
    )
    parser.add_argument("--timeout", type=float, default=180, help="Seconds to wait for each sample.")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return as soon as generation is queued. Does not download audio.",
    )
    return parser


def load_text(args: argparse.Namespace) -> str:
    if args.text and args.text_file:
        raise SystemExit("Pass --text or --text-file, not both.")
    if args.text_file:
        return args.text_file.read_text(encoding="utf-8").strip()
    if args.text:
        return args.text.strip()
    raise SystemExit("Pass --text or --text-file.")


def sample_payload(args: argparse.Namespace, text: str, language: str) -> dict:
    payload: dict = {
        "text": text,
        "language": language,
        "wait": not args.no_wait,
        "timeout_s": args.timeout,
    }
    if args.profile:
        payload["profile"] = args.profile
    if args.engine:
        payload["engine"] = args.engine
    if args.model_size:
        payload["model_size"] = args.model_size
    if args.gateway_model:
        payload["model"] = args.gateway_model
    return payload


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return response.text.strip() or f"HTTP {response.status_code}"


def run_samples(
    args: argparse.Namespace,
    text: str,
    *,
    client: httpx.Client,
) -> list[dict]:
    """POST each language, download the wav, and return manifest rows."""
    base = args.base_url.rstrip("/")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    languages = args.languages or list(DEFAULT_LANGUAGES)
    rows: list[dict] = []
    for language in languages:
        response = client.post(
            f"{base}/dub/sample",
            json=sample_payload(args, text, language),
            timeout=httpx.Timeout(args.timeout + 30, connect=10.0),
        )
        if response.status_code >= 400:
            raise SystemExit(f"{language}: {_detail(response)}")
        sample = response.json()
        audio_url = sample.get("audio_url") or ""
        row = {
            "language": language,
            "profile_name": sample.get("profile_name"),
            "engine": sample.get("engine"),
            "model_size": sample.get("model_size"),
            "generation_id": sample.get("generation_id"),
            "status": sample.get("status"),
            "translated_text": sample.get("translated_text"),
            "audio_url": audio_url,
            "file": None,
        }
        if not args.no_wait:
            if sample.get("status") != "completed" or not audio_url:
                raise SystemExit(f"{language}: sample did not complete ({sample.get('status')}).")
            audio = client.get(audio_url, timeout=60.0)
            if audio.status_code >= 400:
                raise SystemExit(f"{language}: audio download failed: {_detail(audio)}")
            dest = out_dir / f"sample-{language}.wav"
            dest.write_bytes(audio.content)
            row["file"] = dest.name
        rows.append(row)
        print(f"{language}  {row['file'] or '(queued)'}  {audio_url}")
        translated = row.get("translated_text") or ""
        if translated:
            print(f"    {translated}")
    manifest = {
        "profile": args.profile,
        "source_text": text,
        "purpose": "approval-sample",
        "samples": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest  {out_dir / 'manifest.json'}")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    text = load_text(args)
    if not text:
        raise SystemExit("Transcript is empty.")
    with httpx.Client() as client:
        run_samples(args, text, client=client)


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as exc:
        print(f"Could not reach Voicebox: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
