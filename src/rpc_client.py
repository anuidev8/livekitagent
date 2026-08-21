"""LiveKit RPC helpers for kiosk client tools."""

from __future__ import annotations

import json
import logging

from livekit.agents import ToolError, get_job_context

logger = logging.getLogger("agent.rpc")


async def rpc(method: str, payload: dict | None = None, timeout: float = 8.0) -> str:
    """Call a client tool registered by the kiosk frontend."""
    try:
        room = get_job_context().room
        participant = next(iter(room.remote_participants.values()), None)
        if participant is None:
            raise ToolError(
                "No hay participante en la sala para ejecutar la herramienta."
            )
        return await room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method=method,
            payload=json.dumps(payload or {}),
            response_timeout=timeout,
        )
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("RPC %s failed", method)
        raise ToolError(f"No se pudo ejecutar {method}: {exc}") from exc
