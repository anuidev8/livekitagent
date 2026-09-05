"""Welcome greeting on welcome:ready — waits for data consent before intro."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from livekit.agents import AgentSession

from rpc_client import rpc
from tasks.speech import generate_reply_safe, kill_agent_speech, wait_for_agent_idle

logger = logging.getLogger("agent.welcome_orchestrator")

_active_task: asyncio.Task[None] | None = None


def _consent_required(state: dict[str, Any]) -> bool:
    """True when UI still needs accept_data_consent before start_experience."""
    facts = state.get("facts") if isinstance(state.get("facts"), dict) else {}
    if facts.get("dataConsentRequired") is True:
        return True
    if facts.get("dataConsentAccepted") is True:
        return False
    actions = state.get("availableActions") or []
    if isinstance(actions, list) and "accept_data_consent" in actions:
        return True
    if isinstance(actions, list) and "start_experience" in actions:
        return False
    # Default safe: assume consent gate is on until proven otherwise.
    return True


def build_welcome_instructions(state: dict[str, Any]) -> str:
    facts = state.get("facts") if isinstance(state.get("facts"), dict) else {}
    identity = {
        key: str(facts.get(key) or "").strip()
        for key in ("name", "role", "company", "industry")
    }
    base = (
        "Compón un saludo original en español para esta identidad: "
        f"{json.dumps(identity, ensure_ascii=False)}. "
        "Usa el nombre una sola vez; integra cargo y empresa con naturalidad. "
        "En 2 o 3 frases breves explica que Huella Digital explorará su presencia "
        "pública, fortalezas y oportunidades. "
        "No uses herramientas, no leas un guion literal y no repitas saludos anteriores. "
    )
    if _consent_required(state):
        return (
            base
            + "Cierra pidiendo que marque el check de protección de datos en pantalla "
            "(o diga «acepto»). NO digas que vas a avanzar todavía. "
            "PROHIBIDO llamar navigate_journey en este saludo."
        )
    return (
        base
        + "Invita a conocer cómo funciona («¿Vemos cómo funciona?»). "
        "PROHIBIDO llamar navigate_journey en este saludo."
    )


def schedule_welcome(session: AgentSession) -> bool:
    global _active_task
    if _active_task is not None and not _active_task.done():
        return False
    _active_task = asyncio.create_task(_run_welcome(session))
    return True


async def _run_welcome(session: AgentSession) -> None:
    """Greet on welcome:ready, then stop — consent + start are visitor-driven."""
    await kill_agent_speech(session)
    session.input.set_audio_enabled(False)
    spoke = False
    try:
        raw = await rpc("get_session_state")
        state = json.loads(raw)
        if state.get("step") != "welcome" or state.get("phase") != "ready":
            logger.info(
                "welcome skipped for state=%s:%s", state.get("step"), state.get("phase")
            )
            return

        await wait_for_agent_idle(session)
        try:
            handle = await generate_reply_safe(
                session,
                instructions=build_welcome_instructions(state),
                wait_for_playout=False,
            )
            await handle.wait_for_playout()
            spoke = True
        except Exception:
            logger.exception("welcome speech failed — leaving mic open for visitor")

        await wait_for_agent_idle(session)
        # Do NOT auto-call start_experience: SHOW_DATA_CONSENT requires an
        # affirmative checkbox / accept_data_consent before intro.
        logger.info(
            "welcome delivered (spoke=%s); waiting for consent/start via voice or touch",
            spoke,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("welcome orchestrator failed")
    finally:
        session.input.set_audio_enabled(True)
