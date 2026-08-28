import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tasks.speech import generate_reply_safe, wait_for_agent_idle


class _FakeSpeechHandle:
    def __init__(self) -> None:
        self._future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._future.set_result(None)
        self.wait_for_playout = AsyncMock()

    def __await__(self):
        return self._future.__await__()

    def exception(self) -> None:
        return None


@pytest.mark.asyncio
async def test_generate_reply_safe_waits_for_playout_when_requested() -> None:
    handle = _FakeSpeechHandle()
    session = MagicMock()
    session.generate_reply.return_value = handle

    await generate_reply_safe(
        session,
        instructions="Hola",
        wait_for_playout=True,
    )

    session.generate_reply.assert_called_once_with(instructions="Hola")
    handle.wait_for_playout.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_reply_safe_skips_playout_by_default() -> None:
    handle = _FakeSpeechHandle()
    session = MagicMock()
    session.generate_reply.return_value = handle

    await generate_reply_safe(session, instructions="Hola")

    handle.wait_for_playout.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_for_agent_idle_delegates_to_session() -> None:
    session = MagicMock()
    session.wait_for_idle = AsyncMock()

    await wait_for_agent_idle(session)

    session.wait_for_idle.assert_awaited_once()
