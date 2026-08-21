"""Speech helpers for Amazon Nova Sonic (LiveKit AWS realtime).

Nova Sonic does not support mid-session tool_choice / tools swaps.
Passing tool_choice into generate_reply can trigger:
  "updating inference configuration options is not yet supported"
then a Validation error and session close.
Set tool_choice on RealtimeModel at session build time instead.
"""

from __future__ import annotations

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
    **_ignored: Any,
) -> Any:
    """Call generate_reply without unsupported Nova inference-config updates."""
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
    return handle
