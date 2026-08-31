"""Deterministic «Así funciona» tour — RPC drives UI, generate_reply drives voice.

Storyboard order and copy come from get_session_state.content (processSteps,
dimensionConcepts, deliverableConcepts). No hardcoded Spanish scripts here.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from livekit.agents import AgentSession

from rpc_client import rpc
from tasks.ui_sync import PresentStep, run_present_steps, speak_director_line

logger = logging.getLogger("agent.intro_orchestrator")

_active_task: asyncio.Task[None] | None = None
_run_token = 0


def intro_tour_running() -> bool:
    return _active_task is not None and not _active_task.done()


def cancel_intro_tour() -> None:
    global _active_task, _run_token
    _run_token += 1
    if _active_task and not _active_task.done():
        _active_task.cancel()


def schedule_intro_tour(session: AgentSession) -> bool:
    """Start the intro storyboard once per session (idempotent while running)."""
    global _active_task, _run_token
    if intro_tour_running():
        logger.info("intro orchestrator already running — skip duplicate start")
        return False
    token = _run_token + 1
    _run_token = token
    _active_task = asyncio.create_task(_run_intro_tour(session, token))
    return True


def _ordered_rows(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    rows = [item for item in items if isinstance(item, dict)]
    return sorted(rows, key=lambda row: int(row.get("index", 0)))


def _spotlight_line(item: dict[str, Any]) -> str:
    name = str(item.get("name") or item.get("dimensionName") or item.get("title") or "").strip()
    concept = str(
        item.get("explanation") or item.get("concept") or item.get("copy") or ""
    ).strip()
    if name and concept:
        return f"{name}: {concept}"
    return concept or name


def _points_line(step: dict[str, Any]) -> str:
    points = step.get("points")
    if not isinstance(points, list):
        return ""
    return ". ".join(str(p).strip() for p in points if str(p).strip())


def _card_script(step: dict[str, Any]) -> str:
    voice = str(step.get("voiceScript") or "").strip()
    if voice:
        return voice
    return _points_line(step)


async def _still_on_intro() -> bool:
    try:
        raw = await rpc("get_session_state")
        data = json.loads(raw)
    except Exception:
        return True
    return str(data.get("step") or "") == "intro"


async def _step_ok(token: int) -> bool:
    if token != _run_token:
        return False
    return await _still_on_intro()


async def _load_content() -> dict[str, Any]:
    try:
        raw = await rpc("get_session_state")
        state = json.loads(raw)
    except Exception:
        return {}
    content = state.get("content")
    return content if isinstance(content, dict) else {}


def _dimensions_script(dimensions: list[dict[str, Any]]) -> str:
    """One continuous narration covering all 5 dimension spotlights."""
    lines = []
    for concept in dimensions:
        line = _spotlight_line(concept)
        if line:
            lines.append(line)
    return ". ".join(lines)


def _deliverables_script(deliverables: list[dict[str, Any]]) -> str:
    """One continuous narration covering all 3 deliverables."""
    lines = []
    for item in deliverables:
        line = _spotlight_line(item)
        if line:
            lines.append(line)
    return ". ".join(lines)


def _build_intro_steps(content: dict[str, Any]) -> list[PresentStep]:
    """Build exactly 3 card-level steps — one voice block per card.

    Card 0: gestures/interaction  (intro_step index=0)
    Card 1: all 5 dimensions      (intro_step index=1, voice covers all at once)
    Card 2: all 3 deliverables    (intro_step index=2, voice covers all at once)

    Sub-item spotlights (intro_card_dimension / intro_deliverable) are driven
    client-side from the live transcript via introTourTranscriptSync — not via
    individual orchestrator steps.  This avoids the premature-ack problem where
    a brief inter-chunk silence from Nova was incorrectly interpreted as the
    narration ending, causing the UI card to advance before the voice finished.
    """
    steps_rows = _ordered_rows(content.get("processSteps"))
    dimensions = _ordered_rows(content.get("dimensionConcepts"))
    deliverables = _ordered_rows(content.get("deliverableConcepts"))

    step0 = steps_rows[0] if steps_rows else {}
    step1 = steps_rows[1] if len(steps_rows) > 1 else {}
    step2 = steps_rows[2] if len(steps_rows) > 2 else {}

    dim_names = [
        str(c.get("title") or c.get("name") or "")
        for c in dimensions
        if c.get("title") or c.get("name")
    ]
    dim_fallback = _dimensions_script(dimensions) or _card_script(step1)

    del_fallback = _deliverables_script(deliverables) or _card_script(step2)

    return [
        # ── Card 0: Cómo interactuar ─────────────────────────────────────────
        PresentStep(
            target="intro_step",
            index=0,
            fallback_speak=_card_script(step0),
            pace="card",
            extra_instructions=(
                "BREVE y amable — 4 frases máximo. "
                "Cubre: deslizar derecha avanza, doble izquierda sale, "
                "en análisis deslizar cambia dimensión y pulgar arriba abre detalle, "
                "toque o voz en cualquier momento. "
                "PROHIBIDO dimensiones, entregables, «empezamos el análisis»."
            ),
        ),
        # ── Card 1: Las 5 dimensiones ────────────────────────────────────────
        # Voice narrates ALL five in one continuous block.  The client drives
        # each dimension-icon spotlight from the live transcript keywords, so
        # the UI animation stays in sync without needing individual ack steps.
        PresentStep(
            target="intro_step",
            index=1,
            fallback_speak=dim_fallback,
            pace="card",
            extra_instructions=(
                "Frase de apertura MUY corta (~2 s) — p.ej. «Tu análisis mide cinco "
                "dimensiones de presencia digital» — luego narra CADA dimensión "
                "en este orden exacto: "
                + ", ".join(dim_names)
                + ". Para cada una di su nombre seguido de UNA frase que explique qué mide. "
                "Di los nombres TAL CUAL (incluye «Índice de Venta Social» completo). "
                "Flujo continuo sin pausas largas entre dimensiones. "
                "PROHIBIDO listar con números ni guiones. PROHIBIDO preguntas retóricas."
            ),
        ),
        # ── Card 2: Qué recibirás ────────────────────────────────────────────
        # Same approach: one block covers all 3 deliverables; client spotlights
        # each icon from transcript keywords (radar / informe / correo).
        PresentStep(
            target="intro_step",
            index=2,
            fallback_speak=del_fallback,
            pace="card",
            extra_instructions=(
                "Frase de apertura breve (~2 s) — p.ej. «Al terminar recibirás tres cosas» "
                "— luego narra CADA entregable en orden: Radar, Informe, Correo. "
                "Para cada uno di su nombre y UNA frase de valor. "
                "Flujo continuo. PROHIBIDO preguntas retóricas."
            ),
        ),
    ]


async def _run_intro_tour(session: AgentSession, token: int) -> None:
    logger.info("intro orchestrator start token=%s", token)
    session.interrupt()
    try:
        if not await _step_ok(token):
            return

        content = await _load_content()
        steps = _build_intro_steps(content)
        logger.info("intro orchestrator %d director steps", len(steps))

        await run_present_steps(
            session,
            steps,
            should_continue=lambda: _step_ok(token),
        )

        if not await _step_ok(token):
            return
        try:
            await rpc("intro_tour_finished", {})
        except Exception as exc:
            logger.warning("intro_tour_finished RPC failed: %s", exc)

        await speak_director_line(
            session,
            segment_id="intro_tour:closing_question",
            instructions=(
                "El recorrido «Así funciona» ya terminó. "
                "Pregunta UNA vez, con calma: «¿Empezamos el análisis?» y PARA. "
                "PROHIBIDO repetir gestos, dimensiones o entregables."
            ),
            wait_for_playout=True,
            wait_for_client_ack=True,
        )
        logger.info("intro orchestrator complete token=%s", token)
    except asyncio.CancelledError:
        logger.info("intro orchestrator cancelled token=%s", token)
        raise
    except Exception:
        logger.exception("intro orchestrator failed token=%s", token)
