"""Present UI content over RPC before the voice narrates it.

Call present_and_speak (or run_present_steps) from phase tasks / orchestrators
so the kiosk UI updates first, then Nova speaks, then playout completes before
the next step.

Important: do not pass per-reply tool_choice on Nova Sonic — Bedrock rejects
inference-config updates and cancels the AgentTask.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from livekit.agents import AgentSession

from rpc_client import rpc
from tasks.speech import generate_reply_safe, wait_for_agent_idle

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


@dataclass(frozen=True)
class PresentStep:
    """One UI focus + narration beat in a director sequence."""

    target: str
    index: int = 0
    dimension_id: str = ""
    section: str = ""
    fallback_speak: str = ""
    anchors: tuple[str, ...] = ()
    extra_instructions: str = ""
    brief: bool = False
    pace: IntroPace | None = None


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
    anchors: tuple[str, ...],
    extra: str,
) -> str:
    clean_anchors = [item for item in anchors if item]
    anchor_text = json.dumps(clean_anchors, ensure_ascii=False)
    shape = {
        "card": "3 a 6 frases breves que cubran los puntos",
        "transition": "una sola frase breve de transición",
        "spotlight": "una frase breve sobre este elemento",
    }[pace]
    return (
        "El foco ya está visible. No uses herramientas. "
        f"Compón {shape} en español a partir de estas anclas: {anchor_text}. "
        "Parafrasea con naturalidad: no leas las anclas como lista ni copies una frase de la UI. "
        "No repitas contenido de segmentos anteriores y para al terminar. "
        f"{extra}"
    ).strip()


async def speak_director_line(
    session: AgentSession,
    *,
    segment_id: str,
    instructions: str,
    wait_for_playout: bool = True,
    interrupt_first: bool = False,
) -> bool:
    """Speak one director line and wait for server-observed audio playout.

    Do not interrupt between orchestrator steps — that cuts the previous
    segment while room audio is still playing. Only the orchestrator entry
    should pass interrupt_first=True to clear welcome/navigate tail speech.
    """
    if interrupt_first:
        session.interrupt()
    await wait_for_agent_idle(session)

    handle = await generate_reply_safe(
        session,
        instructions=instructions,
        wait_for_playout=False,
    )
    if wait_for_playout:
        await handle.wait_for_playout()
    await wait_for_agent_idle(session)
    # SpeechHandle playout is the sequencing authority. Active-speaker events
    # are useful for UI chrome, but are too coarse to acknowledge a segment.
    return True


def _present_label(
    *,
    target: str,
    index: int,
    dimension_id: str,
    data: dict[str, Any],
) -> str:
    surface = data.get("surface") or target
    parts = [f"{surface}@{index}" if index >= 0 else surface]
    if dimension_id:
        parts.append(f"dim={dimension_id}")
    if data.get("already_focused"):
        parts.append("already_focused")
    return " ".join(parts)


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
    anchors: tuple[str, ...] = (),
    extra_instructions: str = "",
    dimension_id: str = "",
    section: str = "",
    brief: bool = False,
    pace: IntroPace | None = None,
    wait_for_playout: bool = True,
) -> dict[str, Any]:
    """Update the kiosk UI first, narrate, then wait for room audio to finish."""
    data = await rpc_present_content(
        target=target,
        index=index,
        dimension_id=dimension_id,
        section=section,
    )
    label = _present_label(
        target=target,
        index=index,
        dimension_id=dimension_id,
        data=data,
    )
    if data.get("ok") is False:
        logger.warning("director step skipped speak (present failed): %s", label)
        return data

    await asyncio.sleep(_paint_delay_ms(target) / 1000)
    resolved_anchors = anchors or tuple(_facts_points(data))
    if not resolved_anchors and fallback_speak:
        resolved_anchors = (fallback_speak,)
    resolved_pace: IntroPace = pace or ("spotlight" if brief else "card")
    instructions = _pace_instructions(
        resolved_pace,
        resolved_anchors,
        extra_instructions,
    )
    logger.info("director speak start: %s pace=%s", label, resolved_pace)
    acked = await speak_director_line(
        session,
        segment_id=label,
        instructions=instructions,
        wait_for_playout=wait_for_playout,
    )
    logger.info("director speak done: %s playout_complete=%s", label, acked)
    return data


async def run_present_steps(
    session: AgentSession,
    steps: Iterable[PresentStep],
    *,
    wait_for_playout: bool = True,
    should_continue: Callable[[], Awaitable[bool]] | None = None,
) -> None:
    """Run a scripted UI+voice sequence one step at a time (no pipelining)."""
    for step in steps:
        if should_continue is not None and not await should_continue():
            logger.info("director sequence stopped early before %s", step.target)
            return
        await present_and_speak(
            session,
            target=step.target,
            index=step.index,
            fallback_speak=step.fallback_speak,
            anchors=step.anchors,
            extra_instructions=step.extra_instructions,
            dimension_id=step.dimension_id,
            section=step.section,
            brief=step.brief,
            pace=step.pace,
            wait_for_playout=wait_for_playout,
        )
