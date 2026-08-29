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
    return str(
        item.get("explanation") or item.get("concept") or item.get("copy") or ""
    ).strip()


def _points_line(step: dict[str, Any]) -> str:
    points = step.get("points")
    if not isinstance(points, list):
        return ""
    return ". ".join(str(p).strip() for p in points if str(p).strip())


def _step_anchors(
    step: dict[str, Any], *, include_points: bool = True
) -> tuple[str, ...]:
    title = str(step.get("title") or "").strip()
    points = step.get("points")
    rows = (
        [str(item).strip() for item in points]
        if include_points and isinstance(points, list)
        else []
    )
    return tuple(item for item in (title, *rows) if item)


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


def _build_intro_steps(content: dict[str, Any]) -> list[PresentStep]:
    steps_rows = _ordered_rows(content.get("processSteps"))
    dimensions = _ordered_rows(content.get("dimensionConcepts"))
    deliverables = _ordered_rows(content.get("deliverableConcepts"))

    steps: list[PresentStep] = []

    step0 = steps_rows[0] if steps_rows else {}
    steps.append(
        PresentStep(
            target="intro_step",
            index=0,
            anchors=_step_anchors(step0),
            pace="card",
            extra_instructions=(
                "Cubre todos los puntos en facts.points sin omitir ninguno. "
                "PROHIBIDO dimensiones, entregables o «empezamos el análisis»."
            ),
        )
    )

    step1 = steps_rows[1] if len(steps_rows) > 1 else {}
    steps.append(
        PresentStep(
            target="intro_step",
            index=1,
            anchors=_step_anchors(step1, include_points=False),
            pace="transition",
            extra_instructions="Puente breve antes de los iconos de dimensiones.",
        )
    )

    for concept in dimensions:
        dim_id = str(concept.get("id") or "").strip()
        if not dim_id:
            continue
        steps.append(
            PresentStep(
                target="intro_card_dimension",
                index=-1,
                dimension_id=dim_id,
                anchors=tuple(
                    item
                    for item in (
                        str(concept.get("title") or "").strip(),
                        _spotlight_line(concept),
                    )
                    if item
                ),
                pace="spotlight",
                extra_instructions="Solo esta dimensión.",
            )
        )

    step2 = steps_rows[2] if len(steps_rows) > 2 else {}
    steps.append(
        PresentStep(
            target="intro_step",
            index=2,
            anchors=_step_anchors(step2, include_points=False),
            pace="transition",
            extra_instructions="Puente breve antes de los entregables.",
        )
    )

    for item in deliverables:
        del_id = str(item.get("id") or "").strip()
        idx = int(item.get("index", 0))
        steps.append(
            PresentStep(
                target="intro_deliverable",
                index=idx,
                dimension_id=del_id,
                anchors=tuple(
                    anchor
                    for anchor in (
                        str(item.get("title") or "").strip(),
                        _spotlight_line(item),
                    )
                    if anchor
                ),
                pace="spotlight",
                extra_instructions="Solo este entregable.",
            )
        )

    return steps


async def _run_intro_tour(session: AgentSession, token: int) -> None:
    logger.info("intro orchestrator start token=%s", token)
    session.interrupt()
    session.input.set_audio_enabled(False)
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
                "Compón una pregunta breve invitando a empezar el análisis y PARA. "
                "PROHIBIDO repetir gestos, dimensiones o entregables."
            ),
            wait_for_playout=True,
        )
        logger.info("intro orchestrator complete token=%s", token)
    except asyncio.CancelledError:
        logger.info("intro orchestrator cancelled token=%s", token)
        raise
    except Exception:
        logger.exception("intro orchestrator failed token=%s", token)
    finally:
        session.input.set_audio_enabled(True)
