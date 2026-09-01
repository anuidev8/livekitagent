"""Speech helpers for Amazon Nova Sonic (LiveKit AWS realtime).

Nova Sonic does not support mid-session tool_choice / tools swaps.
Passing tool_choice into generate_reply can trigger:
  "updating inference configuration options is not yet supported"
then a Validation error and session close.
Set tool_choice on RealtimeModel at session build time instead.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from livekit.agents import AgentSession

logger = logging.getLogger("agent.speech")


async def generate_reply_safe(
    session: AgentSession,
    *,
    instructions: str,
    allow_interruptions: bool | None = None,
    tool_choice: Any = None,
    tools: Any = None,
    wait_for_playout: bool = False,
    **_ignored: Any,
) -> Any:
    """Call generate_reply without unsupported Nova inference-config updates.

    When ``wait_for_playout`` is True, blocks until assistant audio for this
    turn has fully finished — use for director/orchestrator steps so the next
    present_content does not race ahead of speech.
    """
    if tool_choice is not None or tools is not None:
        logger.debug(
            "Omitting tool_choice/tools for Nova Sonic generate_reply "
            "(per-reply inference config updates are unsupported)"
        )

    kwargs: dict[str, Any] = {"instructions": instructions}
    if allow_interruptions is not None:
        kwargs["allow_interruptions"] = allow_interruptions

    handle = session.generate_reply(**kwargs)
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
        logger.debug("generate_reply playout complete handle=%s", getattr(handle, "id", "?"))

    return handle


async def wait_for_agent_idle(session: AgentSession, timeout: float = 1.5) -> None:
    """Wait until the session has no in-flight agent speech or tool work.

    A short timeout guards against Nova Sonic still processing a user turn
    after session.interrupt() — we don't want to block the director for seconds
    just because the model hasn't flushed its internal state yet.
    """
    try:
        await asyncio.wait_for(session.wait_for_idle(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.debug("wait_for_agent_idle timed out after %.1fs — continuing", timeout)
