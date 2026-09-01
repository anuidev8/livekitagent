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
from narration_barrier import get_session_narration_barrier
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


def _paint_delay_ms(target: str, index: int = 0) -> int:
    if target == "intro_step":
        # Cards 1 and 2 have animated sub-elements (dimension icons, deliverable
        # icons) that need a moment to render before voice starts.
        return 600 if index >= 1 else 450
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
        anchor = spoken or "; ".join(points)
        # When extra_instructions provide detailed per-card guidance (multi-item
        # narration for dimensions / deliverables cards), let them lead and use
        # the anchor only as a factual reference, not as a verbatim constraint.
        if extra:
            return (
                "El foco YA está en pantalla. No uses herramientas. "
                f"Referencia de contenido (no leer literal): «{anchor}». "
                f"Sigue estas instrucciones: {extra} "
                "Tono cálido. Para al terminar."
            ).strip()
        return (
            "El foco YA está en pantalla. No uses herramientas. "
            f"Di este guion en español, sin añadir ideas nuevas ni repetir pantallas anteriores: "
            f"«{anchor}». Tono cálido. Para al terminar."
        ).strip()
    if pace == "transition":
        return (
            "El foco YA está en pantalla. No uses herramientas. "
            f"Di EXACTAMENTE esta frase de puente, sin añadir nada antes ni después: «{spoken}». "
            f"{extra}"
        ).strip()
    return (
        "El icono YA está resaltado. No uses herramientas. "
        f"Di EXACTAMENTE esta frase, sin añadir nada antes ni después: «{spoken}». "
        f"{extra}"
    ).strip()


async def _arm_client_narration_ack(segment_id: str, token: int) -> None:
    await rpc(
        "director_narration_arm",
        {
            "segmentId": segment_id,
            "token": token,
            "timeoutMs": 120_000,
        },
        timeout=5.0,
    )


async def speak_director_line(
    session: AgentSession,
    *,
    segment_id: str,
    instructions: str,
    wait_for_playout: bool = True,
    wait_for_client_ack: bool = True,
) -> bool:
    """Interrupt, speak one director line, wait for kiosk room-audio ack."""
    session.interrupt()
    await wait_for_agent_idle(session)

    # Nova Sonic uses server-side turn detection and ignores allow_interruptions=False.
    # After interrupt() + wait_for_idle(), the server-side VAD still needs time to
    # settle before a new generate_reply fires — otherwise the first audio chunk gets
    # clipped by the server's own reset event.
    # 1.2s puts us past Nova's VAD reset window (~50–600ms observed in production).
    await asyncio.sleep(1.2)

    barrier = get_session_narration_barrier()
    token = barrier.arm(segment_id)
    try:
        await _arm_client_narration_ack(segment_id, token)
    except Exception as exc:
        logger.warning("director_narration_arm failed segment=%s: %s", segment_id, exc)

    await generate_reply_safe(
        session,
        instructions=instructions,
        allow_interruptions=False,
        wait_for_playout=wait_for_playout,
    )
    await wait_for_agent_idle(session)

    if not wait_for_client_ack:
        return True

    acked = await barrier.wait(segment_id, token, timeout=120.0)
    if not acked:
        logger.warning("director client ack timeout segment=%s", segment_id)
    return acked


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
    extra_instructions: str = "",
    dimension_id: str = "",
    section: str = "",
    brief: bool = False,
    pace: IntroPace | None = None,
    wait_for_playout: bool = True,
    wait_for_client_ack: bool = True,
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

    await asyncio.sleep(_paint_delay_ms(target, index) / 1000)
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
    logger.info("director speak start: %s pace=%s", label, resolved_pace)
    acked = await speak_director_line(
        session,
        segment_id=label,
        instructions=instructions,
        wait_for_playout=wait_for_playout,
        wait_for_client_ack=wait_for_client_ack,
    )
    logger.info("director speak done: %s client_ack=%s", label, acked)
    return data


async def run_present_steps(
    session: AgentSession,
    steps: Iterable[PresentStep],
    *,
    wait_for_playout: bool = True,
    wait_for_client_ack: bool = True,
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
            extra_instructions=step.extra_instructions,
            dimension_id=step.dimension_id,
            section=step.section,
            brief=step.brief,
            pace=step.pace,
            wait_for_playout=wait_for_playout,
            wait_for_client_ack=wait_for_client_ack,
        )
