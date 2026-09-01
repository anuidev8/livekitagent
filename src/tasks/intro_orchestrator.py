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

from tasks.speech import wait_for_agent_idle
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
    try:
        # The intro cue is emitted while the start_experience tool turn may still
        # be closing. Wait for that turn instead of interrupting it: interrupting
        # here clips Nova's first syllable and leaves its server-side VAD unsettled.
        await wait_for_agent_idle(session, timeout=12.0)
        await asyncio.sleep(0.35)

        if not _token_valid(token):
            return

        # Keep the overview and closing question in the same generated reply.
        # A second reply creates a new turn boundary where noise can barge in.
        await speak_director_line(
            session,
            segment_id="intro_tour",
            instructions=(
                "Entrega UNA sola locución natural de máximo 20 segundos, sin herramientas "
                "ni pausas largas. Explica que pueden interactuar con gestos en el aire o "
                "con la voz; que explorarán cinco dimensiones: Autoridad, LinkedIn SSI, "
                "Mensaje, Influencia e Higiene; y que al finalizar recibirán un radar "
                "personalizado, un informe detallado y el resumen en su correo. "
                "No enumeres con números ni expliques cada dimensión. Cierra dentro de la "
                "MISMA locución con «¿Empezamos el análisis?» y PARA."
            ),
            wait_for_playout=True,
            wait_for_client_ack=True,
            skip_interrupt=True,
        )
        logger.info("intro orchestrator complete token=%s", token)
    except asyncio.CancelledError:
        logger.info("intro orchestrator cancelled token=%s", token)
        raise
    except Exception:
        logger.exception("intro orchestrator failed token=%s", token)
