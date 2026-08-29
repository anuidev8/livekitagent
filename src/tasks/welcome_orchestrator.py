"""Welcome greeting then atomic welcome→intro transition via RPC."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from livekit.agents import AgentSession

from rpc_client import rpc
from tasks.speech import generate_reply_safe, wait_for_agent_idle

logger = logging.getLogger("agent.welcome_orchestrator")

_active_task: asyncio.Task[None] | None = None


def build_welcome_instructions(state: dict[str, Any]) -> str:
    facts = state.get("facts") if isinstance(state.get("facts"), dict) else {}
    identity = {
        key: str(facts.get(key) or "").strip()
        for key in ("name", "role", "company", "industry")
    }
    return (
        "Compón un saludo original en español para esta identidad: "
        f"{json.dumps(identity, ensure_ascii=False)}. "
        "Usa el nombre una sola vez; integra cargo y empresa con naturalidad. "
        "En 2 o 3 frases breves explica que Huella Digital explorará su presencia "
        "pública, fortalezas y oportunidades, e invita a conocer cómo funciona. "
        "No uses herramientas, no leas un guion literal y no repitas saludos anteriores."
    )


def schedule_welcome(session: AgentSession) -> bool:
    global _active_task
    if _active_task is not None and not _active_task.done():
        return False
    _active_task = asyncio.create_task(_run_welcome(session))
    return True


async def _navigate_to_intro() -> None:
    result = json.loads(
        await rpc("navigate_journey", {"action": "start_experience"}, retries=2)
    )
    if result.get("ok") is False:
        raise RuntimeError(f"welcome transition rejected: {result}")


async def _run_welcome(session: AgentSession) -> None:
    session.interrupt()
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
            logger.exception("welcome speech failed — continuing to intro transition")

        await wait_for_agent_idle(session)
        await _navigate_to_intro()
        logger.info("welcome delivered (spoke=%s); navigated to intro", spoke)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("welcome orchestrator failed")
        try:
            await _navigate_to_intro()
            logger.info("welcome fallback navigate_to_intro succeeded after failure")
        except Exception:
            logger.exception("welcome fallback navigate_to_intro also failed")
    finally:
        session.input.set_audio_enabled(True)
