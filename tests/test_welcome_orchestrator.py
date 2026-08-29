import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from tasks.welcome_orchestrator import _run_welcome, build_welcome_instructions


def test_welcome_prompt_uses_identity_facts_not_spoken_fallback() -> None:
    prompt = build_welcome_instructions(
        {
            "spokenContent": "Bienvenido, Bejarano.",
            "facts": {
                "name": "Bejarano",
                "role": "Presidente",
                "company": "Pajonales",
                "industry": "Agroindustria",
            },
        }
    )

    assert "Bejarano" in prompt
    assert "Presidente" in prompt
    assert "Pajonales" in prompt
    assert "Bienvenido, Bejarano." not in prompt
    assert "compón" in prompt.lower()


@pytest.mark.asyncio
async def test_welcome_waits_for_playout_then_marks_and_navigates() -> None:
    session = MagicMock()
    handle = AsyncMock()
    state = {
        "step": "welcome",
        "phase": "ready",
        "facts": {"name": "Ada", "role": "CEO", "company": "SETI"},
    }

    with (
        patch(
            "tasks.welcome_orchestrator.rpc",
            new=AsyncMock(
                side_effect=[
                    json.dumps(state),
                    json.dumps({"ok": True}),
                    json.dumps({"ok": True, "step": "intro"}),
                ]
            ),
        ) as rpc_mock,
        patch(
            "tasks.welcome_orchestrator.generate_reply_safe",
            new=AsyncMock(return_value=handle),
        ),
        patch("tasks.welcome_orchestrator.wait_for_agent_idle", new=AsyncMock()),
    ):
        await _run_welcome(session)

    handle.wait_for_playout.assert_awaited_once()
    assert rpc_mock.await_args_list == [
        call("get_session_state"),
        call("welcome_narration_finished", {}, retries=1),
        call("navigate_journey", {"action": "start_experience"}, retries=1),
    ]
    assert session.input.set_audio_enabled.call_args_list == [call(False), call(True)]
