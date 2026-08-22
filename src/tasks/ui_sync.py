"""Present UI content over RPC before the voice narrates it.

Call present_and_speak from phase tasks so the kiosk UI updates first, then
generate a reply grounded in the returned spokenContent.

Important: do not pass per-reply tool_choice on Nova Sonic — Bedrock rejects
inference-config updates and cancels the AgentTask.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from livekit.agents import AgentSession

from rpc_client import rpc
from tasks.speech import generate_reply_safe

logger = logging.getLogger("agent.ui_sync")

# Must stay aligned with huella-digital AttractInteractionTour ATTRACT_TOUR_STEPS.
ATTRACT_CARD_SCRIPTS: list[dict[str, Any]] = [
    {
        "index": 0,
        "title": "Gestos",
        "speak": (
            "Con gestos: desliza la mano en el aire para mover el slider. "
            "Une pulgar e índice para elegir un botón, y cierra la mano "
            "para confirmar. Sin tocar la pantalla."
        ),
    },
    {
        "index": 1,
        "title": "Toque",
        "speak": (
            "Con toque: apunta con el dedo y toca la pantalla para elegir o confirmar."
        ),
    },
    {
        "index": 2,
        "title": "Voz",
        "speak": (
            "Con voz: hábame con naturalidad. Entiendo si quiere avanzar, "
            "volver, repetir o empezar."
        ),
    },
]


def _parse_rpc_json(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def rpc_present_content(
    *,
    target: str,
    index: int = 0,
    dimension_id: str = "",
    section: str = "",
) -> dict[str, Any]:
    """Focus one UI item and return the parsed RPC payload."""
    raw = await rpc(
        "present_content",
        {
            "target": target,
            "index": index,
            "dimensionId": dimension_id,
            "section": section,
        },
    )
    data = _parse_rpc_json(raw)
    if data.get("ok") is False:
        logger.warning(
            "present_content failed target=%s index=%s raw=%s", target, index, raw
        )
    return data


async def present_and_speak(
    session: AgentSession,
    *,
    target: str,
    index: int,
    fallback_speak: str,
    extra_instructions: str = "",
    dimension_id: str = "",
    section: str = "",
) -> dict[str, Any]:
    """Update the kiosk UI first, then narrate only that focused item."""
    data = await rpc_present_content(
        target=target,
        index=index,
        dimension_id=dimension_id,
        section=section,
    )
    spoken = (
        str(data.get("spokenContent") or data.get("narration") or "").strip()
        or fallback_speak
    )
    title = str(data.get("title") or "").strip()
    title_bit = f" Título en pantalla: {title}." if title else ""

    await generate_reply_safe(
        session,
        instructions=(
            "La tarjeta YA está visible en el espejo. No digas que la vas a mostrar. "
            "No llames herramientas en este turno. "
            f"Explica solo este contenido, breve y natural:{title_bit} {spoken} "
            f"{extra_instructions}"
        ).strip(),
        # Inference only — Nova omits this inside generate_reply_safe.
        tool_choice="none",
    )
    return data
