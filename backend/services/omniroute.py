"""Translate dubbing transcripts through OmniRoute.

Voicebox talks only to the Swade AI gateway (``SWADE_AI_GATEWAY_URL``,
optional ``SWADE_AI_GATEWAY_STANDBY_URL``). Provider credentials for
OpenAI, Anthropic, and xAI stay inside OmniRoute — this client never
reads those environment variables and never sends an Authorization header.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx

GATEWAY_URL_ENV = "SWADE_AI_GATEWAY_URL"
GATEWAY_STANDBY_URL_ENV = "SWADE_AI_GATEWAY_STANDBY_URL"

# Present in the process environment or not, these are never forwarded.
# Provider keys belong on the OmniRoute host, not in the Voicebox env.
IGNORED_PROVIDER_KEY_ENVS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY")

DEFAULT_MODEL = "auto"
DUB_LANGUAGE_NAMES = {
    "es": "Spanish",
    "fr": "French",
}

# Gateway is unhealthy or overloaded — try the standby host.
_FAILOVER_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class OmniRouteError(Exception):
    """The gateway was reached (or both hosts failed) and translation did not."""


class OmniRouteConfigError(OmniRouteError):
    """``SWADE_AI_GATEWAY_URL`` is missing. Nothing was sent."""


def ignored_provider_keys() -> list[str]:
    """Names of provider-key env vars that are set and deliberately unused."""
    return [name for name in IGNORED_PROVIDER_KEY_ENVS if os.environ.get(name)]


def chat_completions_url(base_url: str) -> str:
    """Join a gateway base URL with the OpenAI-compatible chat path.

    ``SWADE_AI_GATEWAY_URL`` is the OmniRoute origin (or an origin that
    already ends in ``/v1``). The translate call always lands on
    ``/v1/chat/completions``.
    """
    base = base_url.strip().rstrip("/")
    if not base:
        raise OmniRouteConfigError("SWADE_AI_GATEWAY_URL is empty.")
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise OmniRouteConfigError(
            f"SWADE_AI_GATEWAY_URL must be an http(s) OmniRoute origin, not {parsed.scheme or 'an unparsed value'}."
        )
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        return base
    if path.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def gateway_urls() -> list[str]:
    """Primary gateway, then the optional standby. Blanks are dropped."""
    urls: list[str] = []
    for name in (GATEWAY_URL_ENV, GATEWAY_STANDBY_URL_ENV):
        value = (os.environ.get(name) or "").strip()
        if value:
            urls.append(value)
    return urls


def _host_label(url: str) -> str:
    parsed = urlparse(url.strip())
    return parsed.netloc or "gateway"


def _system_prompt(language: str) -> str:
    name = DUB_LANGUAGE_NAMES[language]
    return (
        f"You translate transcripts for voice dubbing into {name} ({language}). "
        "Return only the translated speech — the exact words to be spoken. "
        "Do not add notes, labels, quotation marks, or a language name. "
        "Preserve names, numbers, and the speaker's intent. "
        "Write it so it sounds natural aloud."
    )


def _clean_translation(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        else:
            cleaned = cleaned.strip("`")
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    for prefix in ("Spanish:", "Español:", "French:", "Français:", "Francais:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return _clean_translation(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return _clean_translation("".join(parts))
    return _clean_translation(str(content or ""))


def _extract_translation(payload: object) -> str:
    if not isinstance(payload, dict):
        raise OmniRouteError("OmniRoute returned a non-object chat response.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OmniRouteError("OmniRoute returned no translation choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    text = _message_text(content)
    if not text:
        raise OmniRouteError("OmniRoute returned an empty translation.")
    return text


def _config_error() -> OmniRouteConfigError:
    message = (
        "SWADE_AI_GATEWAY_URL is not set. Point it at OmniRoute; "
        "Voicebox calls {url}/v1/chat/completions. "
        "Optional failover: SWADE_AI_GATEWAY_STANDBY_URL. "
        "Do not put OpenAI, Anthropic, or xAI keys in the Voicebox environment — "
        "those credentials stay inside OmniRoute."
    )
    ignored = ignored_provider_keys()
    if ignored:
        message += " Set but ignored here: " + ", ".join(ignored) + "."
    return OmniRouteConfigError(message)


def _unauthorized_message(host: str) -> str:
    return (
        f"OmniRoute at {host} rejected the translation (401). "
        "Voicebox does not send OpenAI, Anthropic, or xAI keys. "
        "Allow this local client on the OmniRoute host (its own gateway key "
        "stays there), or the sample cannot be translated."
    )


async def translate_transcript(
    text: str,
    language: str,
    *,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Translate ``text`` into ``es`` or ``fr`` via OmniRoute chat completions."""
    if language not in DUB_LANGUAGE_NAMES:
        raise OmniRouteError(f"Dubbing translation supports es and fr, not '{language}'.")
    urls = gateway_urls()
    if not urls:
        raise _config_error()

    payload = {
        "model": (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "temperature": 0.2,
        "stream": False,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": _system_prompt(language)},
            {"role": "user", "content": text},
        ],
    }
    # Content-Type only. No Authorization, even if a provider key is exported.
    headers = {"Content-Type": "application/json"}

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    last_detail = "OmniRoute did not respond."
    try:
        for index, base in enumerate(urls):
            endpoint = chat_completions_url(base)
            host = _host_label(base)
            is_last = index == len(urls) - 1
            try:
                response = await client.post(endpoint, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_detail = f"{host} unreachable ({exc.__class__.__name__})."
                if is_last:
                    raise OmniRouteError(
                        f"Translation failed. {last_detail} "
                        "Check SWADE_AI_GATEWAY_URL" + (" and SWADE_AI_GATEWAY_STANDBY_URL." if len(urls) > 1 else ".")
                    ) from exc
                continue

            if response.status_code in _FAILOVER_STATUSES and not is_last:
                last_detail = f"{host} returned HTTP {response.status_code}."
                continue
            if response.status_code == 401:
                raise OmniRouteError(_unauthorized_message(host))
            if response.status_code >= 400:
                body = response.text.strip().replace("\n", " ")
                if len(body) > 300:
                    body = body[:300] + "…"
                detail = f"OmniRoute at {host} returned HTTP {response.status_code}."
                if body:
                    detail += f" {body}"
                raise OmniRouteError(detail)
            try:
                data = response.json()
            except ValueError as exc:
                raise OmniRouteError(f"OmniRoute at {host} returned a non-JSON translation response.") from exc
            return _extract_translation(data)
    finally:
        if owns_client:
            await client.aclose()

    raise OmniRouteError(f"Translation failed. {last_detail}")
