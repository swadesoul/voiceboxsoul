"""Approval-sample dubbing.

Transcript text is translated by OmniRoute, then spoken with the same
generation path as ``voicebox.speak`` / ``POST /speak``, pinned to
Spanish (``es``) or French (``fr``) and a chosen voice profile.

Samples are for a caller to approve before any production use. Personality
rewrite is off so the audio is the translation, not a second rewrite.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from . import history
from .omniroute import DUB_LANGUAGE_NAMES, translate_transcript

# Preferred size when an engine ships more than one and several can speak
# the target language. TADA's multilingual weights are 3B; Qwen defaults
# to 1.7B. 1B (TADA English) is never chosen implicitly.
_SIZE_PREFERENCE = ("1.7B", "3B", "0.6B", "1B")


class DubError(Exception):
    """A dub sample cannot be produced. ``status_code`` maps to HTTP."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class DubSample:
    language: str
    profile_id: str
    profile_name: str
    engine: str
    model_size: str | None
    source_text: str
    translated_text: str
    generation_id: str
    status: str
    audio_path: str | None
    error: str | None = None


def engines_supporting(language: str) -> list[str]:
    """TTS engine ids that ship at least one model speaking ``language``."""
    from ..backends import get_tts_model_configs

    found: list[str] = []
    for cfg in get_tts_model_configs():
        if language in cfg.languages and cfg.engine not in found:
            found.append(cfg.engine)
    return found


def select_model_size(engine: str, language: str, requested: str | None) -> str | None:
    """Pick a model size that can speak ``language``, or reject the engine.

    Returns ``None`` for engines that have a single size (Kokoro, Chatterbox,
    LuxTTS). Those sizes are not forwarded into generation.
    """
    from ..backends import engine_has_model_sizes, get_tts_model_configs

    configs = [cfg for cfg in get_tts_model_configs() if cfg.engine == engine]
    if not configs:
        known = ", ".join(engines_supporting(language))
        raise DubError(
            f"Unknown engine '{engine}'. For {language}, use one of: {known}.",
            status_code=400,
        )

    if requested:
        matched = [cfg for cfg in configs if cfg.model_size == requested]
        if not matched:
            sizes = ", ".join(sorted({cfg.model_size for cfg in configs}))
            raise DubError(
                f"Engine '{engine}' has no model size '{requested}'. Available: {sizes}.",
                status_code=400,
            )
        if language not in matched[0].languages:
            raise DubError(_unsupported_engine_message(engine, language, matched[0].display_name), status_code=400)
        if not engine_has_model_sizes(engine):
            return None
        return requested

    supporting = [cfg for cfg in configs if language in cfg.languages]
    if not supporting:
        raise DubError(_unsupported_engine_message(engine, language), status_code=400)
    if not engine_has_model_sizes(engine):
        return None
    for size in _SIZE_PREFERENCE:
        for cfg in supporting:
            if cfg.model_size == size:
                return cfg.model_size
    return supporting[0].model_size


def _unsupported_engine_message(engine: str, language: str, display_name: str | None = None) -> str:
    name = DUB_LANGUAGE_NAMES.get(language, language)
    who = display_name or f"Engine '{engine}'"
    options = ", ".join(engines_supporting(language))
    return (
        f"{who} cannot dub into {name} ({language}). "
        f"Pick a voice profile whose engine is one of: {options}. "
        "English-only weights (LuxTTS, Chatterbox Turbo, TADA 1B) are rejected. "
        "Pass engine explicitly to override the profile default."
    )


def resolve_dub_engine(explicit: str | None, binding, profile) -> str:
    """Same engine precedence as POST /speak, then the profile default."""
    engine = explicit
    if engine is None and binding is not None:
        engine = getattr(binding, "default_engine", None)
    return engine or getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen"


def lookup_profile(explicit: str | None, client_id: str | None, db: Session):
    """Resolve a voice profile the same way voicebox.speak does.

    Imported lazily so translation tests do not load the TTS stack.
    """
    from ..mcp_server.resolve import resolve_profile

    return resolve_profile(explicit, client_id, db)


def publish_speak_start(payload: dict) -> None:
    """Tell the speaking pill a dub sample started. Lazy for the same reason."""
    from ..mcp_server.events import publish

    publish("speak-start", payload)


def build_generation_request(
    *,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str | None,
):
    """Generation request voicebox.speak would build for this translation.

    Personality stays off so the spoken text is the translation.
    """
    from .. import models

    return models.GenerationRequest(
        profile_id=profile_id,
        text=text,
        language=language,
        engine=engine,
        personality=False,
        model_size=model_size,
    )


def _binding_for_client(client_id: str | None, db: Session):
    if not client_id:
        return None
    from ..database.models import MCPClientBinding

    return db.query(MCPClientBinding).filter(MCPClientBinding.client_id == client_id).first()


def _present_path(path: str | None) -> str | None:
    if path is None or path == "":
        return None
    return path


async def wait_for_generation(generation_id: str, db: Session, timeout_s: float):
    """Poll until the speak job completes, fails, or the timeout elapses."""
    deadline = time.monotonic() + timeout_s
    while True:
        db.expire_all()
        gen = await history.get_generation(generation_id, db)
        if gen is None:
            raise DubError(
                f"Dub sample {generation_id} disappeared before it finished.",
                status_code=500,
            )
        status = gen.status or "completed"
        if status in ("completed", "failed"):
            return gen
        if time.monotonic() >= deadline:
            raise DubError(
                f"Dub sample {generation_id} was still '{status}' after {timeout_s:.0f}s. "
                f"Poll /generate/{generation_id}/status, or raise timeout_s.",
                status_code=504,
            )
        await asyncio.sleep(0.5)


async def create_dub_sample(
    *,
    text: str,
    language: str,
    profile: str | None,
    engine: str | None,
    model_size: str | None,
    gateway_model: str | None,
    client_id: str | None,
    db: Session,
    wait: bool = True,
    timeout_s: float = 180,
) -> DubSample:
    """Translate ``text`` and speak it in ``language`` with ``profile``."""
    if language not in DUB_LANGUAGE_NAMES:
        raise DubError(
            "Dubbing samples support language 'es' (Spanish) or 'fr' (French).",
            status_code=400,
        )

    voice = lookup_profile(profile, client_id, db)
    if voice is None:
        if profile:
            raise DubError(f"Voice profile '{profile}' not found.", status_code=404)
        raise DubError(
            "No voice profile resolved. Pass profile (name or id), "
            "or configure a default in Voicebox → Settings → MCP.",
            status_code=400,
        )

    binding = _binding_for_client(client_id, db)
    resolved_engine = resolve_dub_engine(engine, binding, voice)
    resolved_size = select_model_size(resolved_engine, language, model_size)

    translated = await translate_transcript(text, language, model=gateway_model)

    generation = await _speak_translation(
        profile_id=voice.id,
        profile_name=voice.name,
        text=translated,
        language=language,
        engine=resolved_engine,
        model_size=resolved_size,
        client_id=client_id,
        db=db,
    )

    status = getattr(generation, "status", None) or "generating"
    audio_path = _present_path(getattr(generation, "audio_path", None))
    error = getattr(generation, "error", None)
    if wait:
        finished = await wait_for_generation(generation.id, db, timeout_s)
        status = finished.status or "completed"
        audio_path = _present_path(finished.audio_path)
        error = finished.error
        if status == "failed":
            detail = error or "TTS generation failed."
            raise DubError(
                f"Dub sample failed after translation: {detail} "
                "A multilingual voice model may still need to be downloaded "
                f"for engine '{resolved_engine}'" + (f" ({resolved_size})" if resolved_size else "") + ".",
                status_code=502,
            )

    return DubSample(
        language=language,
        profile_id=voice.id,
        profile_name=voice.name,
        engine=resolved_engine,
        model_size=resolved_size,
        source_text=text,
        translated_text=translated,
        generation_id=generation.id,
        status=status,
        audio_path=audio_path,
        error=error,
    )


async def _speak_translation(
    *,
    profile_id: str,
    profile_name: str,
    text: str,
    language: str,
    engine: str,
    model_size: str | None,
    client_id: str | None,
    db: Session,
):
    """Speak translated text. Same generation request voicebox.speak builds.

    ``personality`` is forced off. A personality rewrite would replace the
    translation before the caller hears it.
    """
    from ..routes.generations import generate_speech

    generation = await generate_speech(
        build_generation_request(
            profile_id=profile_id,
            text=text,
            language=language,
            engine=engine,
            model_size=model_size,
        ),
        db,
    )
    await history.set_generation_source(generation.id, "dub", db)
    publish_speak_start(
        {
            "generation_id": generation.id,
            "profile_name": profile_name,
            "source": "dub",
            "client_id": client_id,
        }
    )
    return generation
