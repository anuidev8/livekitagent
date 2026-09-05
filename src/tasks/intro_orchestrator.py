"""Deterministic «Así funciona» tour — one short voice narration; UI animates independently.

The intro screen now shows a single animated icon reel (gestures → 5 dimensions →
3 deliverables) that loops automatically without any orchestrator sync.
The agent delivers one brief spoken overview (~15-20 s) and then asks
«¿Empezamos el análisis?» — no card-by-card coordination needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from livekit.agents import AgentSession

from narration_barrier import get_session_narration_barrier
from rpc_client import commit_guide_screen
from tasks.speech import kill_agent_speech
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
    # Invalidate in-flight intro narration ack so a late client silence
    # cannot resume / unblock a cancelled tour.
    with contextlib.suppress(Exception):
        get_session_narration_barrier().invalidate()


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
        # Button "Comenzar" often fires while welcome is still speaking.
        # Hard-kill (interrupt force + clear_buffer + idle) so welcome audio
        # and tools cannot bleed into onboarding.
        await kill_agent_speech(session)

        if not _token_valid(token):
            return

        # Reveal intro UI only after welcome buffers are cleared — lockstep
        # with the new overview speech (hold+skeleton on the client).
        await commit_guide_screen("intro:run")

        if not _token_valid(token):
            return

        # Keep the overview and closing question in the same generated reply.
        # A second reply creates a new turn boundary where noise can barge in.
        await speak_director_line(
            session,
            segment_id="intro_tour",
            instructions=(
                "INTRO «CÓMO FUNCIONA» — NO es welcome. "
                "PROHIBIDO ABSOLUTO: Bienvenido/Bienvenida, saludar por nombre, cargo, "
                "empresa, SETI rol, «Es un gusto saludarte», «presencia pública para "
                "entender su posicionamiento», o repetir CUALQUIER frase de la "
                "bienvenida anterior. "
                "Entrega UNA sola locución natural de máximo 30 segundos, sin herramientas "
                "ni pausas largas. PROHIBIDO abrir anunciando lo que vas a hacer «ahora te "
                "explico», «vamos a ver cómo funciona», «te cuento el onboarding» y similares "
                "— PROHIBIDO decir la palabra 'onboarding'. Entra DIRECTO al contenido, "
                "explicando ya mismo que este espejo responde a su toque en la pantalla "
                "o a su voz — pueden tocar cuando quieran, o simplemente hablar. "
                "PROHIBIDO ABSOLUTO mencionar cómo navegar entre resultados en esta "
                "primera frase — esa mención va SOLO más adelante, junto con las "
                "dimensiones, nunca antes. "
                "Antes de nombrarlas, agrega UNA frase muy breve que enmarque qué son "
                "las dimensiones en conjunto — por ejemplo, que vas a medir su presencia "
                "digital en cinco dimensiones distintas — sin explicar el concepto a "
                "fondo, solo dar ese contexto antes de nombrarlas una por una. "
                "Menciona las cinco dimensiones, cada una con una idea MUY corta (3-5 "
                "palabras, no una oración completa) de qué mide — no las enumeres con "
                "números ni las expliques a fondo, solo nómbralas con esa idea breve: "
                "Autoridad (qué tan visible eres en Google), LinkedIn SSI "
                "(tu fuerza en LinkedIn), Mensaje (qué tan claro comunicas), Influencia "
                "(cuánto alcance tiene tu voz más allá de tu organización), e Higiene "
                "(qué tan protegido está tu rastro digital). "
                "INMEDIATAMENTE DESPUÉS de nombrar las cinco dimensiones (no antes, no "
                "mezclado con la frase de interacción inicial): explica que en sus "
                "resultados podrán tocar las flechas para navegar entre ellas, y tocar "
                "«Ver detalle» para profundizar en la que les interese. Esta es la "
                "ÚNICA mención de cómo navegar en toda la locución. "
                "Cierra explicando que al finalizar recibirán un radar personalizado, "
                "un informe detallado y el resumen en su correo. Cierra dentro de la "
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
