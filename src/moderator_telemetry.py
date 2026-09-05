"""Report voice usage / throttles to huella-moderator for agent routing."""

from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp

logger = logging.getLogger("agent.moderator_telemetry")

MODERATOR_URL = (
    os.getenv("MODERATOR_URL") or os.getenv("HUELLA_MODERATOR_URL") or ""
).rstrip("/")
AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "huella-guide")


async def report_voice_telemetry(
    event: str,
    *,
    tokens: int | None = None,
    room: str | None = None,
    detail: str | None = None,
    agent_id: str | None = None,
) -> None:
    """Best-effort POST to moderator /api/voice/telemetry."""
    if not MODERATOR_URL:
        return
    payload: dict[str, Any] = {
        "agentId": (agent_id or AGENT_NAME).strip(),
        "event": event,
    }
    if tokens is not None:
        payload["tokens"] = int(tokens)
    if room:
        payload["room"] = room
    if detail:
        payload["detail"] = detail[:500]

    url = f"{MODERATOR_URL}/api/voice/telemetry"
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(url, json=payload) as resp,
        ):
            if resp.status >= 400:
                body = await resp.text()
                logger.warning(
                    "moderator telemetry %s → %s %s",
                    event,
                    resp.status,
                    body[:200],
                )
    except Exception as exc:
        logger.debug("moderator telemetry failed: %s", exc)


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
