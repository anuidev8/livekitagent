"""LiveKit RPC helpers for kiosk client tools."""

from __future__ import annotations

import asyncio
import json
import logging

from livekit.agents import ToolError, get_job_context

logger = logging.getLogger("agent.rpc")


async def rpc(
    method: str,
    payload: dict | None = None,
    timeout: float = 8.0,
    retries: int = 2,
) -> str:
    """Call a client tool registered by the kiosk frontend.

    Retries transient failures (UI not ready yet / brief disconnects) with
    short backoff. Missing remote participant is retried as well so the agent
    can join slightly ahead of the browser without hard-failing the turn.
    """
    last_error: Exception | None = None

    for attempt in range(retries + 1):
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
        except ToolError as exc:
            last_error = exc
            if attempt >= retries:
                raise
            delay = 0.45 * (attempt + 1)
            logger.warning(
                "RPC %s attempt %s/%s failed (%s); retry in %.2fs",
                method,
                attempt + 1,
                retries + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            last_error = exc
            logger.exception("RPC %s failed on attempt %s", method, attempt + 1)
            if attempt >= retries:
                raise ToolError(f"No se pudo ejecutar {method}: {exc}") from exc
            await asyncio.sleep(0.45 * (attempt + 1))

    raise ToolError(f"No se pudo ejecutar {method}: {last_error}")
