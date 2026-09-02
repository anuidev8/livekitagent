"""Compatibility fix for Nova Sonic session continuation in LiveKit AWS 1.7.0.

The upstream recycle timer starts the replacement stream from inside the timer
task itself.  While initializing that stream it starts the next timer and
cancels the "existing" timer, which is the task currently doing the recycle.
The cancellation aborts initialization at its next await, leaving the LiveKit
room connected but the replacement Nova stream unable to receive audio.

Keep this patch small and version-gated so it can be removed when the plugin
ships an upstream fix.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

SUPPORTED_LIVEKIT_AWS_VERSION = "1.7.0"
_PATCH_MARKER = "_huella_avoids_recycle_timer_self_cancel"
_RECONNECT_EVENT_PATCH_MARKER = "_huella_emits_session_reconnected"

logger = logging.getLogger("agent.nova_continuation")


def install_nova_session_continuation_fix() -> bool:
    """Install the LiveKit AWS 1.7.0 recycle-timer fix once.

    Returns ``True`` when the compatibility fix is active. A different plugin
    version fails closed instead of silently patching unknown internals.
    """

    try:
        installed_version = version("livekit-plugins-aws")
    except PackageNotFoundError as exc:  # pragma: no cover - deployment guard
        raise RuntimeError("livekit-plugins-aws is not installed") from exc

    if installed_version != SUPPORTED_LIVEKIT_AWS_VERSION:
        raise RuntimeError(
            "Nova continuation compatibility fix supports "
            f"livekit-plugins-aws=={SUPPORTED_LIVEKIT_AWS_VERSION}; "
            f"found {installed_version}. Review the upstream recycler before upgrading."
        )

    from livekit.plugins.aws.experimental.realtime import realtime_model

    session_class = realtime_model.RealtimeSession
    current_method = session_class._start_session_recycle_timer
    if getattr(current_method, _PATCH_MARKER, False):
        return True

    def _start_session_recycle_timer_without_self_cancel(self: Any) -> None:
        existing_task = self._session_recycle_task
        current_task = asyncio.current_task()

        # Cancel a stale, independent timer, but never cancel the timer that is
        # currently completing a graceful recycle.
        if (
            existing_task is not None
            and not existing_task.done()
            and existing_task is not current_task
        ):
            existing_task.cancel()

        duration = self._calculate_session_duration()
        next_task = asyncio.create_task(
            self._session_recycle_timer(duration),
            name="RealtimeSession._session_recycle_timer",
        )
        self._session_recycle_task = next_task

        if existing_task is current_task:
            logger.info(
                "[SESSION] Armed next Nova recycle timer without cancelling "
                "the active renewal"
            )

    setattr(_start_session_recycle_timer_without_self_cancel, _PATCH_MARKER, True)
    session_class._start_session_recycle_timer = (  # type: ignore[method-assign]
        _start_session_recycle_timer_without_self_cancel
    )
    logger.info(
        "Installed Nova session continuation fix for livekit-plugins-aws %s",
        installed_version,
    )
    return True


def _with_session_reconnected_emit(
    original_recycle: Callable[[Any], Awaitable[None]],
) -> Callable[[Any], Awaitable[None]]:
    """Wrap a ``_graceful_session_recycle``-shaped coroutine so it emits the
    standard ``session_reconnected`` event on ``self`` after ``original_recycle``
    completes. Split out from ``install_nova_session_reconnected_event_fix`` so
    the wrapping behaviour is testable against a fake recycle/session without
    touching the real (heavy, AWS-calling) vendored method.
    """
    from livekit.agents.llm.realtime import RealtimeSessionReconnectedEvent

    async def _graceful_session_recycle_and_emit(self: Any) -> None:
        await original_recycle(self)
        self.emit("session_reconnected", RealtimeSessionReconnectedEvent())
        logger.info("[SESSION] Emitted session_reconnected after recycle")

    setattr(_graceful_session_recycle_and_emit, _RECONNECT_EVENT_PATCH_MARKER, True)
    return _graceful_session_recycle_and_emit


def install_nova_session_reconnected_event_fix() -> bool:
    """Make a completed Nova recycle emit the standard ``session_reconnected``
    event so application code can re-anchor the model after reconnecting.

    ``RealtimeSession.EventTypes`` (livekit-agents core) documents
    ``"session_reconnected"`` as the signal a realtime plugin fires after
    reconnecting — ``RealtimeFallbackAdapter`` emits it on its own reconnects.
    livekit-plugins-aws 1.7.0's ``_graceful_session_recycle`` rebuilds the
    Bedrock stream (see ``initialize_streams(is_restart=True)``) but never
    emits it, so application code has no framework-standard signal that a
    mid-call recycle just happened.

    Regression (2026-09-02, RM_vZnfXrLvRboG logs): a recycle landed while the
    visitor was mid-flow. For the rest of that session the model spoke as if
    it were acting ("vamos a proceder con eso", "colócate frente al espejo
    para tomar la foto") but never called get_session_state, present_content,
    or navigate_journey again — the kiosk UI never moved. Emitting
    session_reconnected lets ``agent.py`` push a reinforcement instruction
    right after reconnect, the same way it already re-anchors the model on
    every ``[pantalla:]`` screen change.

    Returns ``True`` when the fix is active. A different plugin version fails
    closed instead of silently patching unknown internals.
    """

    try:
        installed_version = version("livekit-plugins-aws")
    except PackageNotFoundError as exc:  # pragma: no cover - deployment guard
        raise RuntimeError("livekit-plugins-aws is not installed") from exc

    if installed_version != SUPPORTED_LIVEKIT_AWS_VERSION:
        raise RuntimeError(
            "Nova session_reconnected compatibility fix supports "
            f"livekit-plugins-aws=={SUPPORTED_LIVEKIT_AWS_VERSION}; "
            f"found {installed_version}. Review the upstream recycler before upgrading."
        )

    from livekit.plugins.aws.experimental.realtime import realtime_model

    session_class = realtime_model.RealtimeSession
    original_recycle = session_class._graceful_session_recycle
    if getattr(original_recycle, _RECONNECT_EVENT_PATCH_MARKER, False):
        return True

    session_class._graceful_session_recycle = (  # type: ignore[method-assign]
        _with_session_reconnected_emit(original_recycle)
    )
    logger.info(
        "Installed Nova session_reconnected event fix for livekit-plugins-aws %s",
        installed_version,
    )
    return True
