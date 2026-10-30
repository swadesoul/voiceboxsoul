"""POST /dub/sample — approval dub before any production use.

Transcript text goes to OmniRoute (``SWADE_AI_GATEWAY_URL`` /v1), then
through the same speak path as ``voicebox.speak`` in Spanish or French.
The JSON body includes an ``audio_url`` Big Mamma can forward. ``download=true``
returns the wav itself once generation finishes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..database import get_db
from ..services.dub import DubError, create_dub_sample
from ..services.omniroute import OmniRouteConfigError, OmniRouteError

router = APIRouter()


def _audio_url(request: Request, generation_id: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/audio/{generation_id}"


@router.post("/dub/sample", response_model=None)
async def dub_sample(
    data: models.DubSampleRequest,
    request: Request,
    download: bool = False,
    db: Session = Depends(get_db),
):
    """Dub one transcript into Spanish or French and return a sample.

    ``download=true`` waits (unless ``wait`` is false) and responds with the
    wav file. The default JSON response carries ``audio_url`` plus the
    translation, so the sample can be sent for approval before production.
    """
    client_id = request.headers.get("X-Voicebox-Client-Id")
    try:
        sample = await create_dub_sample(
            text=data.text,
            language=data.language,
            profile=data.profile,
            engine=data.engine,
            model_size=data.model_size,
            gateway_model=data.model,
            client_id=client_id,
            db=db,
            wait=data.wait,
            timeout_s=data.timeout_s,
        )
    except DubError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except OmniRouteConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OmniRouteError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    payload = models.DubSampleResponse(
        language=sample.language,
        profile_id=sample.profile_id,
        profile_name=sample.profile_name,
        engine=sample.engine,
        model_size=sample.model_size,
        source_text=sample.source_text,
        translated_text=sample.translated_text,
        generation_id=sample.generation_id,
        status=sample.status,
        audio_url=_audio_url(request, sample.generation_id),
        poll_url=f"/generate/{sample.generation_id}/status",
        audio_path=sample.audio_path,
        purpose="approval-sample",
    )

    if not download:
        return payload

    if sample.status != "completed" or not sample.audio_path:
        raise HTTPException(
            status_code=409,
            detail=(
                "Sample audio is not ready to download. "
                f"status={sample.status}. Poll {payload.poll_url} and fetch {payload.audio_url}."
            ),
        )
    audio_file = config.resolve_storage_path(sample.audio_path)
    if audio_file is None or not audio_file.is_file():
        raise HTTPException(status_code=404, detail="Dub sample finished but the audio file is missing.")
    return FileResponse(
        audio_file,
        media_type="audio/wav",
        filename=f"dub-{sample.language}-{sample.generation_id}.wav",
        headers={
            "X-Voicebox-Language": sample.language,
            "X-Voicebox-Generation-Id": sample.generation_id,
            "X-Voicebox-Audio-Url": payload.audio_url,
        },
    )
