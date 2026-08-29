"""Single-owner welcome narration followed by an atomic intro transition."""

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


async def _run_welcome(session: AgentSession) -> None:
    session.interrupt()
    session.input.set_audio_enabled(False)
    try:
        raw = await rpc("get_session_state")
        state = json.loads(raw)
        if state.get("step") != "welcome" or state.get("phase") != "ready":
            logger.info(
                "welcome skipped for state=%s:%s", state.get("step"), state.get("phase")
            )
            return

        await wait_for_agent_idle(session)
        handle = await generate_reply_safe(
            session,
            instructions=build_welcome_instructions(state),
            wait_for_playout=False,
        )
        await handle.wait_for_playout()
        await wait_for_agent_idle(session)

        # The explicit completion RPC replaces ActiveSpeakers-based inference.
        await rpc("welcome_narration_finished", {}, retries=1)
        result = json.loads(
            await rpc("navigate_journey", {"action": "start_experience"}, retries=1)
        )
        if result.get("ok") is False:
            raise RuntimeError(f"welcome transition rejected: {result}")
        logger.info("welcome delivered; navigated to intro")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("welcome orchestrator failed")
    finally:
        session.input.set_audio_enabled(True)
