"""Report voice usage / throttles to huella-moderator for agent routing + spend."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import aiohttp

logger = logging.getLogger("agent.moderator_telemetry")


def _moderator_url() -> str:
    # Read at call time — module import often happens before load_dotenv().
    return (
        os.getenv("MODERATOR_URL") or os.getenv("HUELLA_MODERATOR_URL") or ""
    ).rstrip("/")


def _agent_name() -> str:
    return os.getenv("LIVEKIT_AGENT_NAME", "huella-guide").strip()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def job_dispatch_meta(ctx: Any) -> dict[str, Any]:
    """Parse RoomAgentDispatch metadata (kioskId, etc.)."""
    raw = getattr(getattr(ctx, "job", None), "metadata", None) or ""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def usage_deltas_from_event(
    ev: Any, prev: dict[str, int]
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (new_totals, deltas) from session_usage_updated."""
    usage = getattr(ev, "usage", None)
    model_usage = getattr(usage, "model_usage", None) or []
    totals = {"input": 0, "output": 0, "tokens": 0, "characters": 0}
    for item in model_usage:
        totals["input"] += int(getattr(item, "input_tokens", 0) or 0)
        totals["output"] += int(getattr(item, "output_tokens", 0) or 0)
        totals["tokens"] += int(getattr(item, "total_tokens", 0) or 0)
        totals["characters"] += int(
            getattr(item, "characters_count", None)
            or getattr(item, "charactersCount", None)
            or 0
        )
    if totals["input"] + totals["output"] == 0 and totals["tokens"] > 0:
        pass
    else:
        totals["tokens"] = totals["input"] + totals["output"]

    deltas = {
        "input": max(0, totals["input"] - prev.get("input", 0)),
        "output": max(0, totals["output"] - prev.get("output", 0)),
        "tokens": max(0, totals["tokens"] - prev.get("tokens", 0)),
        "characters": max(0, totals["characters"] - prev.get("characters", 0)),
    }
    return totals, deltas


async def report_voice_telemetry(
    event: str,
    *,
    tokens: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    characters: int | None = None,
    room: str | None = None,
    kiosk_id: str | None = None,
    user_id: str | None = None,
    nombre: str | None = None,
    detail: str | None = None,
    agent_id: str | None = None,
) -> None:
    """Best-effort POST to moderator /api/voice/telemetry."""
    base = _moderator_url()
    if not base:
        return
    payload: dict[str, Any] = {
        "agentId": (agent_id or _agent_name()).strip(),
        "event": event,
    }
    if tokens is not None:
        payload["tokens"] = int(tokens)
    if input_tokens is not None:
        payload["inputTokens"] = int(input_tokens)
    if output_tokens is not None:
        payload["outputTokens"] = int(output_tokens)
    if characters is not None:
        payload["characters"] = int(characters)
    if room:
        payload["room"] = room
    if kiosk_id:
        payload["kioskId"] = kiosk_id
    if user_id:
        payload["userId"] = user_id
    if nombre:
        payload["nombre"] = nombre[:200]
    if detail:
        payload["detail"] = detail[:500]

    url = f"{base}/api/voice/telemetry"
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(url, json=payload) as resp,
        ):
            if resp.status >= 400:
                body = await resp.text()
                logger.warning(
                    "moderator telemetry %s → %s %s at=%s",
                    event,
                    resp.status,
                    body[:200],
                    _now_iso(),
                )
            else:
                logger.info(
                    "moderator telemetry %s ok at=%s agent=%s tokens=%s in=%s out=%s chars=%s room=%s kiosk=%s",
                    event,
                    _now_iso(),
                    payload.get("agentId"),
                    payload.get("tokens"),
                    payload.get("inputTokens"),
                    payload.get("outputTokens"),
                    payload.get("characters"),
                    payload.get("room"),
                    payload.get("kioskId"),
                )
    except Exception as exc:
        logger.warning("moderator telemetry failed at=%s: %s", _now_iso(), exc)


def is_throttle_error(err: BaseException | str | None) -> bool:
    text = str(err or "").lower()
    return any(
        needle in text
        for needle in (
            "throttl",
            "rate limit",
            "too many requests",
            "429",
            "servicequotaexceeded",
            "too many connections",
        )
    )
