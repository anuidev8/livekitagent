import asyncio
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
    """Voice UX recommendation (Implementation §2c): the logged WARN —
    "allow_interruptions cannot be False when using
    VoiceAgent.generate_reply(), disable turn_detection in the RealtimeModel
    and use VAD instead" — fires because Nova Sonic's own server-side turn
    detection ignores the per-reply allow_interruptions flag entirely (it
    was already a dead no-op per the removed comment on this call site).
    Director lines must rely on interrupt() + the VAD-settle sleep already
    in speak_director_line for "uninterruptible" behavior, never on a flag
    Nova silently ignores while still logging a WARN for it.
    """
    session = MagicMock()

    fake_barrier = MagicMock()
    fake_barrier.arm.return_value = 1
    fake_barrier.wait = AsyncMock(return_value=True)

    with (
        patch("tasks.ui_sync.rpc", new=AsyncMock(return_value="{}")),
        patch("tasks.ui_sync.get_session_narration_barrier", return_value=fake_barrier),
        patch("tasks.ui_sync.wait_for_agent_idle", new=AsyncMock()),
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
