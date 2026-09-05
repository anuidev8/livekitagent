import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tasks import intro_orchestrator


@pytest.mark.asyncio
async def test_intro_tour_interrupts_welcome_then_uses_one_uninterrupted_reply() -> None:
    """Button Comenzar during welcome must cut welcome immediately.

    Regression (2026-09-04_12-59-37): wait_for_agent_idle let welcome keep
    talking after the UI was already on intro/onboarding.
    """
    session = MagicMock()
    intro_orchestrator._run_token = 41

    with (
        patch("tasks.intro_orchestrator.asyncio.sleep", new=AsyncMock()),
        patch(
            "tasks.intro_orchestrator.speak_director_line", new=AsyncMock()
        ) as speak,
    ):
        await intro_orchestrator._run_intro_tour(session, token=41)

    session.interrupt.assert_called_once()
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
    idx_frame = kwargs["instructions"].find("cinco dimensiones distintas")
    idx_autoridad = kwargs["instructions"].find("Autoridad")
    assert idx_frame != -1
    assert idx_frame < idx_autoridad


@pytest.mark.asyncio
async def test_cancelled_intro_token_never_starts_speech() -> None:
    session = MagicMock()
    intro_orchestrator._run_token = 8

    with (
        patch("tasks.intro_orchestrator.asyncio.sleep", new=AsyncMock()),
        patch(
            "tasks.intro_orchestrator.speak_director_line", new=AsyncMock()
        ) as speak,
    ):
        await intro_orchestrator._run_intro_tour(session, token=7)

    speak.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_intro_tour_stops_running_orchestrator() -> None:
    """Touch nav past intro must cancel the tour so pantalla cues can speak.

    Regression: 2026-09-04_12-08-50.log — visitor left welcome→analysis→detail
    via taps while intro was still narrating; every [pantalla:] was suppressed
    because cancel_intro_tour was never called from the text_input path.
    """
    intro_orchestrator._active_task = None
    intro_orchestrator._run_token = 0

    session = MagicMock()
    # Park the tour on sleep so cancel() has something to abort.
    idle = asyncio.Event()

    async def block_until_cancelled(*_a, **_k):
        await idle.wait()

    with (
        patch(
            "tasks.intro_orchestrator.asyncio.sleep",
            new=block_until_cancelled,
        ),
        patch(
            "tasks.intro_orchestrator.speak_director_line", new=AsyncMock()
        ) as speak,
    ):
        assert intro_orchestrator.schedule_intro_tour(session) is True
        assert intro_orchestrator.intro_tour_running() is True
        task = intro_orchestrator._active_task
        assert task is not None

        intro_orchestrator.cancel_intro_tour()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert intro_orchestrator.intro_tour_running() is False
        speak.assert_not_awaited()
