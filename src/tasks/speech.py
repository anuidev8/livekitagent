"""Speech helpers for Amazon Nova Sonic (LiveKit AWS realtime).

Nova Sonic does not support mid-session tool_choice / tools swaps.
Passing tool_choice into generate_reply can trigger:
  "updating inference configuration options is not yet supported"
then a Validation error and session close.
Set tool_choice on RealtimeModel at session build time instead.

Kill / clear / fresh-chat_ctx helpers mirror huella-guide-elevenlabs so
touch-nav and hold+commit UI sync behave the same with Nova.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from livekit.agents import AgentSession
from livekit.agents.voice.agent_activity import ActivityClosedError

logger = logging.getLogger("agent.speech")

# Nova server-side VAD needs a short quiet gap after interrupt before the
# next generate_reply — shorter than the old 1.2s director settle, longer
# than ElevenLabs' 0.08s TTS pipeline settle.
_NOVA_KILL_SETTLE_S = 0.35


def build_pantalla_chat_ctx(session: AgentSession) -> Any | None:
    """Fresh chat context for pantalla nav — drop prior-screen history.

    ``generate_reply(instructions=…)`` alone still sends the full attract /
    welcome / intro transcript, so the model keeps saying welcome lines on
    analysis:scanning. Truncating to system instructions only (LiveKit
    ``truncate`` preserves them) forces the reply to follow the new screen
    prompt + tools instead of continuing old turns.
    """
    agent = getattr(session, "current_agent", None)
    if agent is None:
        return None
    chat_ctx = getattr(agent, "chat_ctx", None)
    if chat_ctx is None:
        return None
    try:
        return chat_ctx.copy(exclude_function_call=True).truncate(max_items=0)
    except Exception:
        logger.debug("build_pantalla_chat_ctx failed", exc_info=True)
        return None


def clear_audio_output_buffer(session: AgentSession) -> None:
    """Stop agent-side playout immediately via LiveKit AudioOutput.clear_buffer()."""
    try:
        audio = getattr(getattr(session, "output", None), "audio", None)
    except Exception:
        return
    if audio is None:
        return
    clear = getattr(audio, "clear_buffer", None)
    if not callable(clear):
        return
    try:
        clear()
    except Exception:
        logger.debug("clear_audio_output_buffer raised", exc_info=True)


async def interrupt_force(session: AgentSession, *, timeout: float = 3.0) -> None:
    """Hard-stop current + queued speech and WAIT until teardown completes."""
    try:
        future = session.interrupt(force=True)
    except Exception:
        logger.debug("interrupt(force=True) raised", exc_info=True)
        return
    try:
        await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "interrupt_force: interruption did not complete within %.1fs — continuing anyway",
            timeout,
        )
    except Exception:
        logger.debug("interrupt_force: awaiting interrupt future raised", exc_info=True)


async def kill_agent_speech(
    session: AgentSession,
    *,
    timeout: float = 3.0,
    settle_s: float = _NOVA_KILL_SETTLE_S,
) -> None:
    """Real kill for touch/button screen changes — not frontend mute.

    1. ``interrupt(force=True)`` and await its completion Future
    2. ``output.audio.clear_buffer()`` so RoomIO drops queued PCM
    3. ``wait_for_idle`` so in-flight tool calls from the killed turn drain
    4. Brief settle for Nova Sonic server-side VAD before the next reply
    """
    await interrupt_force(session, timeout=timeout)
    clear_audio_output_buffer(session)
    await wait_for_agent_idle(session, timeout=min(timeout, 2.0))
    clear_audio_output_buffer(session)
    if settle_s > 0:
        await asyncio.sleep(settle_s)


async def generate_reply_safe(
    session: AgentSession,
    *,
    instructions: str,
    allow_interruptions: bool | None = None,
    tool_choice: Any = None,
    tools: Any = None,
    chat_ctx: Any = None,
    wait_for_playout: bool = False,
    **_ignored: Any,
) -> Any:
    """Call generate_reply without unsupported Nova inference-config updates.

    When ``wait_for_playout`` is True, blocks until assistant audio for this
    turn has fully finished — use for director/orchestrator steps so the next
    present_content does not race ahead of speech.

    ``chat_ctx`` is allowed (not an inference-config update) so pantalla nav
    can drop prior-screen history.
    """
    if tool_choice is not None or tools is not None:
        logger.debug(
            "Omitting tool_choice/tools for Nova Sonic generate_reply "
            "(per-reply inference config updates are unsupported)"
        )

    kwargs: dict[str, Any] = {"instructions": instructions}
    if allow_interruptions is not None:
        kwargs["allow_interruptions"] = allow_interruptions
    if chat_ctx is not None:
        kwargs["chat_ctx"] = chat_ctx

    handle = None
    last_exc: BaseException | None = None
    for attempt in range(3):
        try:
            handle = session.generate_reply(**kwargs)
            break
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "draining" not in msg and "pausing" not in msg:
                raise
            last_exc = exc
            logger.warning(
                "generate_reply draining (attempt %s) — wait then retry",
                attempt + 1,
            )
            await interrupt_force(session)
            await wait_for_agent_idle(session, timeout=3.0)
            await asyncio.sleep(0.2 * (attempt + 1))
    if handle is None:
        assert last_exc is not None
        raise last_exc

    await handle

    exc = None
    try:
        exc = handle.exception()
    except Exception:
        exc = None
    if exc is not None:
        logger.warning("generate_reply finished with error: %s", exc)
        raise exc

    if wait_for_playout:
        await handle.wait_for_playout()
        logger.debug(
            "generate_reply playout complete handle=%s", getattr(handle, "id", "?")
        )

    return handle


async def wait_for_agent_idle(session: AgentSession, timeout: float = 2.0) -> None:
    """Wait until the session has no in-flight agent speech or tool work.

    A short timeout guards against Nova Sonic still processing a user turn
    after session.interrupt() — we don't want to block the director for seconds
    just because the model hasn't flushed its internal state yet.
    """
    try:
        await asyncio.wait_for(session.wait_for_idle(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.debug("wait_for_agent_idle timed out after %.1fs — continuing", timeout)
    except ActivityClosedError:
        logger.debug("wait_for_agent_idle: activity already closed — continuing")
