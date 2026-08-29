import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tasks.ui_sync import PresentStep, run_present_steps


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

    with patch("tasks.ui_sync.present_and_speak", new=AsyncMock(side_effect=fake_present)):
        await run_present_steps(session, steps, should_continue=should_continue)

    assert len(calls) == 1
