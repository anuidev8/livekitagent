"""Client audio ack barrier for director / orchestrator speech."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("agent.narration_barrier")


class NarrationBarrier:
    """One in-flight director segment; kiosk acks when room audio goes quiet."""

    def __init__(self) -> None:
        self._event: asyncio.Event | None = None
        self._segment_id = ""
        self._token = 0

    def arm(self, segment_id: str) -> int:
        self._token += 1
        self._segment_id = segment_id
        self._event = asyncio.Event()
        logger.debug("narration barrier armed segment=%s token=%s", segment_id, self._token)
        return self._token

    def ack(self, segment_id: str, token: int) -> bool:
        if token != self._token or segment_id != self._segment_id:
            logger.debug(
                "narration ack ignored segment=%s token=%s (want %s/%s)",
                segment_id,
                token,
                self._segment_id,
                self._token,
            )
            return False
        if self._event and not self._event.is_set():
            self._event.set()
            logger.info("narration barrier ack segment=%s token=%s", segment_id, token)
        return True

    async def wait(self, segment_id: str, token: int, *, timeout: float = 120.0) -> bool:
        if token != self._token or segment_id != self._segment_id or self._event is None:
            return False
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "narration barrier timeout segment=%s token=%s after %.1fs",
                segment_id,
                token,
                timeout,
            )
            return False


_session_barrier: NarrationBarrier | None = None


def set_session_narration_barrier(barrier: NarrationBarrier) -> None:
    global _session_barrier
    _session_barrier = barrier


def get_session_narration_barrier() -> NarrationBarrier:
    if _session_barrier is None:
        raise RuntimeError("session narration barrier is not initialized")
    return _session_barrier
