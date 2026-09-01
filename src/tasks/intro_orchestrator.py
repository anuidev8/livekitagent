"""Deterministic «Así funciona» tour — one short voice narration; UI animates independently.

The intro screen now shows a single animated icon reel (gestures → 5 dimensions →
3 deliverables) that loops automatically without any orchestrator sync.
The agent delivers one brief spoken overview (~15-20 s) and then asks
«¿Empezamos el análisis?» — no card-by-card coordination needed.
"""

from __future__ import annotations

import asyncio
import logging

from livekit.agents import AgentSession

from tasks.ui_sync import speak_director_line

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
    """Start the intro narration once per session (idempotent while running)."""
    global _active_task, _run_token
    if intro_tour_running():
        logger.info("intro orchestrator already running — skip duplicate start")
        return False
    token = _run_token + 1
    _run_token = token
    _active_task = asyncio.create_task(_run_intro_tour(session, token))
    return True


def _token_valid(token: int) -> bool:
    return token == _run_token


async def _run_intro_tour(session: AgentSession, token: int) -> None:
    logger.info("intro orchestrator start token=%s", token)
    session.interrupt()
    try:
        # Let VAD settle after interrupt before speaking.
        await asyncio.sleep(1.0)

        if not _token_valid(token):
            return

        # One compact narration covering all three groups: interaction, dimensions,
        # deliverables.  The animated reel on-screen shows the icons; the voice
        # gives a brief orienting overview — no per-card sync required.
        await speak_director_line(
            session,
            segment_id="intro_tour:overview",
            instructions=(
                "BREVE recorrido de bienvenida — máximo 20 segundos en total. "
                "Menciona en un flujo natural: "
                "(1) que pueden navegar con gestos o con su voz, "
                "(2) que el análisis mide cinco dimensiones: Autoridad, LinkedIn SSI, Mensaje, Influencia e Higiene, "
                "(3) que al finalizar recibirán un radar personalizado, un informe y un correo. "
                "Tono cálido y directo. Sin pausas largas. Sin preguntas retóricas. "
                "PROHIBIDO listar con números o guiones. PROHIBIDO explicar cada dimensión en detalle. "
                "Termina la locución aquí — NO preguntes si empezamos todavía."
            ),
            wait_for_playout=True,
            wait_for_client_ack=False,
        )

        if not _token_valid(token):
            return

        # Brief pause so the reel has a moment to breathe before the closing question.
        await asyncio.sleep(0.6)

        if not _token_valid(token):
            return

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

