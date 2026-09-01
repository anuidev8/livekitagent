from unittest.mock import AsyncMock, patch

import pytest

from tasks import intro_orchestrator


@pytest.mark.asyncio
async def test_intro_tour_waits_for_current_turn_then_uses_one_uninterrupted_reply() -> None:
    session = object()
    intro_orchestrator._run_token = 41

    with (
        patch(
            "tasks.intro_orchestrator.wait_for_agent_idle", new=AsyncMock()
        ) as wait_for_idle,
        patch("tasks.intro_orchestrator.asyncio.sleep", new=AsyncMock()),
        patch(
            "tasks.intro_orchestrator.speak_director_line", new=AsyncMock()
        ) as speak,
    ):
        await intro_orchestrator._run_intro_tour(session, token=41)

    wait_for_idle.assert_awaited_once_with(session, timeout=12.0)
    speak.assert_awaited_once()
    kwargs = speak.await_args.kwargs
    assert kwargs["segment_id"] == "intro_tour"
    assert kwargs["skip_interrupt"] is True
    assert kwargs["wait_for_playout"] is True
    assert kwargs["wait_for_client_ack"] is True
    assert "Autoridad" in kwargs["instructions"]
    assert "LinkedIn SSI" in kwargs["instructions"]
    assert "radar personalizado" in kwargs["instructions"]
    assert "¿Empezamos el análisis?" in kwargs["instructions"]


@pytest.mark.asyncio
async def test_cancelled_intro_token_never_starts_speech() -> None:
    session = object()
    intro_orchestrator._run_token = 8

    with (
        patch(
            "tasks.intro_orchestrator.wait_for_agent_idle", new=AsyncMock()
        ),
        patch("tasks.intro_orchestrator.asyncio.sleep", new=AsyncMock()),
        patch(
            "tasks.intro_orchestrator.speak_director_line", new=AsyncMock()
        ) as speak,
    ):
        await intro_orchestrator._run_intro_tour(session, token=7)

    speak.assert_not_awaited()
