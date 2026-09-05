from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tasks.ui_sync import PresentStep, run_present_steps, speak_director_line


@pytest.mark.asyncio
async def test_run_present_steps_stops_when_should_continue_false() -> None:
    session = object()
    steps = [
        PresentStep(target="intro_step", index=0, fallback_speak="a"),
        PresentStep(target="intro_step", index=1, fallback_speak="b"),
    ]
    calls: list[int] = []

    async def fake_present(*_args, **_kwargs):
        calls.append(1)
        return {"ok": True}

    gate = {"ok": True}

    async def should_continue() -> bool:
        if not gate["ok"]:
            return False
        gate["ok"] = False
        return True

    with patch(
        "tasks.ui_sync.present_and_speak", new=AsyncMock(side_effect=fake_present)
    ):
        await run_present_steps(session, steps, should_continue=should_continue)

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_speak_director_line_omits_allow_interruptions_for_nova() -> None:
    """Nova must never get allow_interruptions=False (WARN + no-op)."""
    session = MagicMock()

    fake_barrier = MagicMock()
    fake_barrier.arm.return_value = 1
    fake_barrier.wait = AsyncMock(return_value=True)

    with (
        patch("tasks.ui_sync.rpc", new=AsyncMock(return_value="{}")),
        patch("tasks.ui_sync.get_session_narration_barrier", return_value=fake_barrier),
        patch("tasks.ui_sync.wait_for_agent_idle", new=AsyncMock()),
        patch("tasks.ui_sync.kill_agent_speech", new=AsyncMock()),
        patch("tasks.ui_sync.build_pantalla_chat_ctx", return_value=None),
        patch("tasks.ui_sync.generate_reply_safe", new=AsyncMock()) as fake_generate,
    ):
        ok = await speak_director_line(
            session,
            segment_id="seg1",
            instructions="Hola",
            skip_interrupt=True,
        )

    assert ok is True
    fake_generate.assert_awaited_once()
    _, kwargs = fake_generate.call_args
    assert "allow_interruptions" not in kwargs
    assert "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_speak_director_line_uses_fresh_chat_ctx_not_welcome_history() -> None:
    session = MagicMock()
    fresh = MagicMock(name="fresh_ctx")
    fake_barrier = MagicMock()
    fake_barrier.arm.return_value = 1
    fake_barrier.wait = AsyncMock(return_value=True)

    with (
        patch("tasks.ui_sync.rpc", new=AsyncMock(return_value="{}")),
        patch("tasks.ui_sync.get_session_narration_barrier", return_value=fake_barrier),
        patch("tasks.ui_sync.wait_for_agent_idle", new=AsyncMock()),
        patch("tasks.ui_sync.kill_agent_speech", new=AsyncMock()),
        patch("tasks.ui_sync.build_pantalla_chat_ctx", return_value=fresh),
        patch("tasks.ui_sync.generate_reply_safe", new=AsyncMock()) as generate,
    ):
        await speak_director_line(
            session,
            segment_id="intro_tour",
            instructions="cómo funciona",
            skip_interrupt=True,
        )

    assert generate.await_args.kwargs["chat_ctx"] is fresh
