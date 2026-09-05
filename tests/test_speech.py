import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.agents.voice.agent_activity import ActivityClosedError

from tasks.speech import (
    clear_audio_output_buffer,
    generate_reply_safe,
    interrupt_force,
    kill_agent_speech,
    wait_for_agent_idle,
)


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
async def test_generate_reply_safe_omits_tool_choice_for_nova() -> None:
    handle = _FakeSpeechHandle()
    session = MagicMock()
    session.generate_reply.return_value = handle

    await generate_reply_safe(
        session,
        instructions="Hola",
        tool_choice="none",
        tools=[],
    )

    session.generate_reply.assert_called_once_with(instructions="Hola")


@pytest.mark.asyncio
async def test_generate_reply_safe_passes_chat_ctx() -> None:
    handle = _FakeSpeechHandle()
    session = MagicMock()
    session.generate_reply.return_value = handle
    fresh = MagicMock(name="fresh_ctx")

    await generate_reply_safe(session, instructions="Hola", chat_ctx=fresh)

    session.generate_reply.assert_called_once_with(
        instructions="Hola",
        chat_ctx=fresh,
    )


@pytest.mark.asyncio
async def test_interrupt_force_awaits_the_interrupt_future_before_returning() -> None:
    resolved = False

    async def _resolve_soon(fut: asyncio.Future) -> None:
        nonlocal resolved
        await asyncio.sleep(0.05)
        resolved = True
        fut.set_result(None)

    future: asyncio.Future = asyncio.get_running_loop().create_future()
    session = MagicMock()
    session.interrupt = MagicMock(return_value=future)

    resolver = asyncio.create_task(_resolve_soon(future))
    await interrupt_force(session)

    session.interrupt.assert_called_once_with(force=True)
    assert resolved is True
    await resolver


@pytest.mark.asyncio
async def test_interrupt_force_is_a_noop_when_nothing_is_speaking() -> None:
    resolved_future: asyncio.Future = asyncio.get_running_loop().create_future()
    resolved_future.set_result(None)
    session = MagicMock()
    session.interrupt = MagicMock(return_value=resolved_future)

    await interrupt_force(session)

    session.interrupt.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_interrupt_force_times_out_gracefully_instead_of_hanging() -> None:
    never_resolves: asyncio.Future = asyncio.get_running_loop().create_future()
    session = MagicMock()
    session.interrupt = MagicMock(return_value=never_resolves)

    await interrupt_force(session, timeout=0.05)

    session.interrupt.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_wait_for_agent_idle_delegates_to_session() -> None:
    session = MagicMock()
    session.wait_for_idle = AsyncMock()

    await wait_for_agent_idle(session)

    session.wait_for_idle.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_for_agent_idle_treats_activity_closed_as_idle() -> None:
    session = MagicMock()
    session.wait_for_idle = AsyncMock(
        side_effect=ActivityClosedError("activity analysis_task is closing")
    )

    await wait_for_agent_idle(session)

    session.wait_for_idle.assert_awaited_once()


def test_clear_audio_output_buffer_calls_sink_clear_buffer() -> None:
    audio = MagicMock()
    session = MagicMock()
    session.output.audio = audio

    clear_audio_output_buffer(session)

    audio.clear_buffer.assert_called_once_with()


def test_clear_audio_output_buffer_is_noop_when_no_audio_sink() -> None:
    session = MagicMock()
    session.output.audio = None

    clear_audio_output_buffer(session)


@pytest.mark.asyncio
async def test_kill_agent_speech_interrupts_clears_buffer_and_waits_idle() -> None:
    resolved: asyncio.Future = asyncio.get_running_loop().create_future()
    resolved.set_result(None)
    audio = MagicMock()
    session = MagicMock()
    session.interrupt = MagicMock(return_value=resolved)
    session.output.audio = audio
    session.wait_for_idle = AsyncMock()

    await kill_agent_speech(session, settle_s=0)

    session.interrupt.assert_called_once_with(force=True)
    assert audio.clear_buffer.call_count == 2
    session.wait_for_idle.assert_awaited_once()


@pytest.mark.asyncio
async def test_kill_agent_speech_settles_before_return() -> None:
    resolved: asyncio.Future = asyncio.get_running_loop().create_future()
    resolved.set_result(None)
    session = MagicMock()
    session.interrupt = MagicMock(return_value=resolved)
    session.output.audio = MagicMock()
    session.wait_for_idle = AsyncMock()

    started = asyncio.get_running_loop().time()
    await kill_agent_speech(session, settle_s=0.05)
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed >= 0.04
