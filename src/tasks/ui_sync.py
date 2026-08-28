"""Present UI content over RPC before the voice narrates it.

Call present_and_speak from phase tasks so the kiosk UI updates first, then
generate a reply grounded in the returned spokenContent.

Important: do not pass per-reply tool_choice on Nova Sonic — Bedrock rejects
inference-config updates and cancels the AgentTask.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from livekit.agents import AgentSession

from rpc_client import rpc
from tasks.speech import generate_reply_safe

logger = logging.getLogger("agent.ui_sync")

IntroPace = Literal["card", "transition", "spotlight"]

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


def _facts_points(data: dict[str, Any]) -> list[str]:
    facts = data.get("facts")
    if not isinstance(facts, dict):
        return []
    points = facts.get("points")
    if not isinstance(points, list):
        return []
    return [str(p).strip() for p in points if str(p).strip()]


def _paint_delay_ms(target: str) -> int:
    if target == "intro_step":
        return 450
    if target in ("intro_card_dimension", "intro_deliverable"):
        return 320
    return 180


def _pace_instructions(
    pace: IntroPace,
    spoken: str,
    points: list[str],
    extra: str,
) -> str:
    if pace == "card":
        anchor = "; ".join(points) if points else spoken
        return (
            "El foco YA está en pantalla. No uses herramientas. "
            f"Cubre TODOS estos puntos en 4-6 frases naturales (~25 s), sin omitir ninguno: "
            f"{anchor}. Tono cálido, no lista mecánica. Para al terminar. "
            f"{extra}"
        ).strip()
    if pace == "transition":
        return (
            "El foco YA está en pantalla. No uses herramientas. "
            f"Di una o dos frases de puente, claras y breves: «{spoken}». "
            "Aún no entres en el detalle de los iconos. Para. "
            f"{extra}"
        ).strip()
    return (
        "El icono YA está resaltado. No uses herramientas. "
        f"Di una o dos frases claras sobre el concepto en pantalla: «{spoken}». "
        "No repitas el título del icono (ya es visible). Para. "
        f"{extra}"
    ).strip()


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
    brief: bool = False,
    pace: IntroPace | None = None,
) -> dict[str, Any]:
    """Update the kiosk UI first, then narrate only that focused item."""
    data = await rpc_present_content(
        target=target,
        index=index,
        dimension_id=dimension_id,
        section=section,
    )
    await asyncio.sleep(_paint_delay_ms(target) / 1000)
    spoken = (
        str(data.get("spokenContent") or data.get("narration") or "").strip()
        or fallback_speak
    )
    resolved_pace: IntroPace = pace or ("spotlight" if brief else "card")
    points = _facts_points(data)
    instructions = _pace_instructions(
        resolved_pace,
        spoken,
        points,
        extra_instructions,
    )
    await generate_reply_safe(
        session,
        instructions=instructions,
        tool_choice="none",
    )
    return data
