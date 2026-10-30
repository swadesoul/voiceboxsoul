"""Approval-sample dubbing: OmniRoute translate, then voicebox speak (es/fr)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from starlette.testclient import TestClient

from backend import models
from backend.database import get_db
from backend.dubbing import build_parser, run_samples
from backend.routes.dub import router as dub_router
from backend.services import dub as dub_service, omniroute
from backend.services.dub import DubError, build_generation_request, create_dub_sample, select_model_size
from backend.services.omniroute import (
    OmniRouteConfigError,
    OmniRouteError,
    chat_completions_url,
    translate_transcript,
)


def _clear_gateway(monkeypatch) -> None:
    monkeypatch.delenv(omniroute.GATEWAY_URL_ENV, raising=False)
    monkeypatch.delenv(omniroute.GATEWAY_STANDBY_URL_ENV, raising=False)
    for name in omniroute.IGNORED_PROVIDER_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


def _chat(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_chat_completions_url_appends_v1_once():
    assert chat_completions_url("http://127.0.0.1:20128") == "http://127.0.0.1:20128/v1/chat/completions"
    assert chat_completions_url("http://127.0.0.1:20128/") == "http://127.0.0.1:20128/v1/chat/completions"
    assert chat_completions_url("http://127.0.0.1:20128/v1") == "http://127.0.0.1:20128/v1/chat/completions"
    assert (
        chat_completions_url("http://127.0.0.1:20128/v1/chat/completions")
        == "http://127.0.0.1:20128/v1/chat/completions"
    )


def test_chat_completions_url_rejects_non_http():
    with pytest.raises(OmniRouteConfigError):
        chat_completions_url("ftp://files.example/models")
    with pytest.raises(OmniRouteConfigError):
        chat_completions_url("")


@pytest.mark.asyncio
async def test_translate_ignores_provider_keys(monkeypatch):
    _clear_gateway(monkeypatch)
    monkeypatch.setenv(omniroute.GATEWAY_URL_ENV, "http://omni.test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content.decode())
        seen["url"] = str(request.url)
        blob = request.content.decode()
        assert "sk-test-openai" not in blob
        assert "sk-ant-test" not in blob
        assert "xai-test" not in blob
        return httpx.Response(200, json=_chat("Hola, el despliegue terminó."))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        text = await translate_transcript("The deploy finished.", "es", client=client)

    assert text == "Hola, el despliegue terminó."
    assert seen["auth"] is None
    assert seen["url"] == "http://omni.test/v1/chat/completions"
    assert seen["body"]["model"] == "auto"
    assert seen["body"]["messages"][1]["content"] == "The deploy finished."
    assert "Spanish" in seen["body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_translate_fails_over_to_standby(monkeypatch):
    _clear_gateway(monkeypatch)
    monkeypatch.setenv(omniroute.GATEWAY_URL_ENV, "http://primary.test")
    monkeypatch.setenv(omniroute.GATEWAY_STANDBY_URL_ENV, "http://standby.test")
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "primary.test":
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=_chat("Bonjour."))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        text = await translate_transcript("Hello.", "fr", model="combo/translate", client=client)

    assert hosts == ["primary.test", "standby.test"]
    assert text == "Bonjour."


@pytest.mark.asyncio
async def test_translate_does_not_failover_on_401(monkeypatch):
    _clear_gateway(monkeypatch)
    monkeypatch.setenv(omniroute.GATEWAY_URL_ENV, "http://primary.test")
    monkeypatch.setenv(omniroute.GATEWAY_STANDBY_URL_ENV, "http://standby.test")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OmniRouteError, match="401"):
            await translate_transcript("Hello.", "es", client=client)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_missing_gateway_names_ignored_provider_keys(monkeypatch):
    _clear_gateway(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    with pytest.raises(OmniRouteConfigError, match="OPENAI_API_KEY") as exc:
        await translate_transcript("Hello.", "es")
    assert "SWADE_AI_GATEWAY_URL" in str(exc.value)
    assert "sk-test-openai" not in str(exc.value)


def test_spanish_and_french_model_sizes():
    assert select_model_size("qwen", "es", None) == "1.7B"
    assert select_model_size("qwen", "fr", "0.6B") == "0.6B"
    assert select_model_size("kokoro", "es", None) is None
    assert select_model_size("kokoro", "fr", None) is None
    assert select_model_size("chatterbox", "fr", None) is None
    assert select_model_size("tada", "es", None) == "3B"
    assert select_model_size("tada", "fr", "3B") == "3B"
    with pytest.raises(DubError, match="TADA 1B"):
        select_model_size("tada", "es", "1B")
    with pytest.raises(DubError, match="luxtts"):
        select_model_size("luxtts", "fr", None)
    with pytest.raises(DubError, match="chatterbox_turbo"):
        select_model_size("chatterbox_turbo", "es", None)


@pytest.mark.asyncio
async def test_create_sample_translates_then_speaks(monkeypatch):
    captured = {}

    async def fake_translate(text, language, *, model=None, client=None):
        captured["translate"] = (text, language, model)
        return "Bonjour, le déploiement est terminé."

    async def fake_speak(**kwargs):
        captured["speak"] = kwargs
        return SimpleNamespace(id="gen-fr", status="generating", audio_path="", error=None)

    async def fake_wait(generation_id, db, timeout_s):
        captured["wait"] = generation_id
        return SimpleNamespace(
            status="completed",
            audio_path="generations/gen-fr.wav",
            error=None,
        )

    monkeypatch.setattr(dub_service, "translate_transcript", fake_translate)
    monkeypatch.setattr(
        dub_service,
        "lookup_profile",
        lambda *a, **k: SimpleNamespace(
            id="profile-1",
            name="Siwis",
            default_engine="kokoro",
            preset_engine="kokoro",
        ),
    )
    monkeypatch.setattr(dub_service, "_speak_translation", fake_speak)
    monkeypatch.setattr(dub_service, "wait_for_generation", fake_wait)

    sample = await create_dub_sample(
        text="The deploy finished.",
        language="fr",
        profile="Siwis",
        engine=None,
        model_size=None,
        gateway_model=None,
        client_id=None,
        db=None,
    )

    assert captured["translate"] == ("The deploy finished.", "fr", None)
    spoken = captured["speak"]
    assert spoken["language"] == "fr"
    assert spoken["text"] == "Bonjour, le déploiement est terminé."
    assert spoken["profile_id"] == "profile-1"
    assert spoken["engine"] == "kokoro"
    assert spoken["model_size"] is None
    request = build_generation_request(
        profile_id=spoken["profile_id"],
        text=spoken["text"],
        language=spoken["language"],
        engine=spoken["engine"],
        model_size=spoken["model_size"],
    )
    assert request.personality is False
    assert request.language == "fr"
    assert sample.generation_id == "gen-fr"
    assert sample.audio_path == "generations/gen-fr.wav"
    assert sample.status == "completed"
    assert sample.profile_name == "Siwis"


@pytest.mark.asyncio
async def test_create_sample_rejects_other_languages():
    with pytest.raises(DubError, match="es"):
        await create_dub_sample(
            text="Hello",
            language="de",
            profile="Morgan",
            engine="qwen",
            model_size=None,
            gateway_model=None,
            client_id=None,
            db=None,
        )


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(dub_router)

    def _db():
        yield None

    app.dependency_overrides[get_db] = _db
    return app


def test_route_returns_audio_url(monkeypatch):
    async def fake_create(**kwargs):
        assert kwargs["language"] == "es"
        assert kwargs["text"] == "Please approve this read."
        return dub_service.DubSample(
            language="es",
            profile_id="p1",
            profile_name="Dora",
            engine="kokoro",
            model_size=None,
            source_text=kwargs["text"],
            translated_text="Por favor, aprueba esta lectura.",
            generation_id="gen-es",
            status="completed",
            audio_path="generations/gen-es.wav",
        )

    monkeypatch.setattr(dub_service, "create_dub_sample", fake_create)
    # The route imported the function directly.
    monkeypatch.setattr("backend.routes.dub.create_dub_sample", fake_create)
    client = TestClient(_app())
    response = client.post(
        "/dub/sample",
        json={"text": "Please approve this read.", "language": "es", "profile": "Dora"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["purpose"] == "approval-sample"
    assert body["audio_url"].endswith("/audio/gen-es")
    assert body["translated_text"].startswith("Por favor")
    assert body["language"] == "es"


def test_route_download_returns_wav(monkeypatch, tmp_path):
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFFfake-wav")

    async def fake_create(**kwargs):
        return dub_service.DubSample(
            language="fr",
            profile_id="p1",
            profile_name="Siwis",
            engine="kokoro",
            model_size=None,
            source_text="Hello",
            translated_text="Bonjour",
            generation_id="gen-fr",
            status="completed",
            audio_path=str(wav),
        )

    monkeypatch.setattr("backend.routes.dub.create_dub_sample", fake_create)
    client = TestClient(_app())
    response = client.post("/dub/sample?download=true", json={"text": "Hello", "language": "fr"})
    assert response.status_code == 200
    assert response.content == b"RIFFfake-wav"
    assert response.headers["x-voicebox-language"] == "fr"
    assert response.headers["content-type"].startswith("audio/wav")


def test_route_maps_missing_gateway(monkeypatch):
    async def fake_create(**kwargs):
        raise OmniRouteConfigError("SWADE_AI_GATEWAY_URL is not set.")

    monkeypatch.setattr("backend.routes.dub.create_dub_sample", fake_create)
    client = TestClient(_app())
    response = client.post("/dub/sample", json={"text": "Hello", "language": "es"})
    assert response.status_code == 503
    assert "SWADE_AI_GATEWAY_URL" in response.json()["detail"]


def test_route_rejects_german():
    client = TestClient(_app())
    response = client.post("/dub/sample", json={"text": "Hello", "language": "de"})
    assert response.status_code == 422


def test_cli_writes_es_and_fr_samples(tmp_path, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content.decode())
            language = body["language"]
            assert body["profile"] == "Morgan"
            assert body["wait"] is True
            assert request.headers.get("authorization") is None
            gen = f"g-{language}"
            translated = "Hola." if language == "es" else "Bonjour."
            return httpx.Response(
                200,
                json={
                    "language": language,
                    "profile_name": "Morgan",
                    "engine": "qwen",
                    "model_size": "1.7B",
                    "generation_id": gen,
                    "status": "completed",
                    "translated_text": translated,
                    "audio_url": f"http://voicebox.test/audio/{gen}",
                },
            )
        assert request.url.path in ("/audio/g-es", "/audio/g-fr")
        return httpx.Response(200, content=b"RIFFsample-" + request.url.path.encode())

    args = build_parser().parse_args(
        [
            "--text",
            "The deploy finished cleanly. Please approve this read.",
            "--profile",
            "Morgan",
            "--base-url",
            "http://voicebox.test",
            "--out-dir",
            str(tmp_path),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rows = run_samples(args, args.text, client=client)

    assert [row["language"] for row in rows] == ["es", "fr"]
    assert (tmp_path / "sample-es.wav").read_bytes().startswith(b"RIFFsample-")
    assert (tmp_path / "sample-fr.wav").read_bytes().startswith(b"RIFFsample-")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["purpose"] == "approval-sample"
    assert manifest["profile"] == "Morgan"
    assert {item["file"] for item in manifest["samples"]} == {"sample-es.wav", "sample-fr.wav"}
    printed = capsys.readouterr().out
    assert "sample-es.wav" in printed
    assert "sample-fr.wav" in printed


def test_speak_request_model_accepts_dub_languages():
    es = models.DubSampleRequest(text="Hello", language="es")
    fr = models.DubSampleRequest(text="Hello", language="fr")
    assert es.language == "es"
    assert fr.wait is True
    with pytest.raises(ValidationError):
        models.DubSampleRequest(text="Hello", language="en")
