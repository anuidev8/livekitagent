from unittest.mock import AsyncMock, patch

import pytest

from tasks.ui_sync import (
    PresentStep,
    _pace_instructions,
    run_present_steps,
    speak_director_line,
)


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


def test_dynamic_pace_prompt_uses_facts_without_exact_script() -> None:
    prompt = _pace_instructions(
        "spotlight",
        ("Autoridad", "referencia sectorial"),
        "Solo esta dimensión.",
    )

    assert "Autoridad" in prompt
    assert "referencia sectorial" in prompt
    assert "EXACTAMENTE" not in prompt
    assert "¿Google te presenta como referente o solo como cargo?" not in prompt
    assert "parafrasea" in prompt.lower()


@pytest.mark.asyncio
async def test_director_sequence_waits_for_server_playout_without_client_ack() -> None:
    session = object()
    handle = AsyncMock()

    with (
        patch("tasks.ui_sync.wait_for_agent_idle", new=AsyncMock()),
        patch(
            "tasks.ui_sync.generate_reply_safe",
            new=AsyncMock(return_value=handle),
        ),
    ):
        assert await speak_director_line(
            session,
            segment_id="intro_step@0",
            instructions="dynamic prompt",
        )

    handle.wait_for_playout.assert_awaited_once()
