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
from importlib.metadata import PackageNotFoundError, version
from typing import Any

SUPPORTED_LIVEKIT_AWS_VERSION = "1.7.0"
_PATCH_MARKER = "_huella_avoids_recycle_timer_self_cancel"

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

