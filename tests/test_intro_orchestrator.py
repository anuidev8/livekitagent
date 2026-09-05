from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tasks import intro_orchestrator


@pytest.mark.asyncio
async def test_intro_tour_interrupts_welcome_then_uses_one_uninterrupted_reply() -> (
    None
):
    """Button Comenzar during welcome must cut welcome immediately."""
    session = MagicMock()
    intro_orchestrator._run_token = 41
    order: list[str] = []

    async def _kill(_session):
        order.append("kill")

    async def _commit(key):
        order.append(f"commit:{key}")

    async def _speak(*_a, **_k):
        order.append("speak")

    with (
        patch(
            "tasks.intro_orchestrator.kill_agent_speech",
            new=AsyncMock(side_effect=_kill),
        ),
        patch(
            "tasks.intro_orchestrator.commit_guide_screen",
            new=AsyncMock(side_effect=_commit),
        ),
        patch(
            "tasks.intro_orchestrator.speak_director_line",
            new=AsyncMock(side_effect=_speak),
        ) as speak,
    ):
        await intro_orchestrator._run_intro_tour(session, token=41)

    assert order == ["kill", "commit:intro:run", "speak"]
    kwargs = speak.await_args.kwargs
    assert kwargs["segment_id"] == "intro_tour"
    assert kwargs["skip_interrupt"] is True
    assert kwargs["wait_for_playout"] is True
    assert kwargs["wait_for_client_ack"] is True
    assert "Autoridad" in kwargs["instructions"]
    assert "LinkedIn SSI" in kwargs["instructions"]
    assert "radar personalizado" in kwargs["instructions"]
    assert "¿Empezamos el análisis?" in kwargs["instructions"]
    assert "toque en la pantalla" in kwargs["instructions"]
    assert (
        "Bienvenido" in kwargs["instructions"] or "bienvenida" in kwargs["instructions"]
    )
    assert "PROHIBIDO ABSOLUTO" in kwargs["instructions"]
    idx_frame = kwargs["instructions"].find("cinco dimensiones distintas")
    idx_autoridad = kwargs["instructions"].find("Autoridad")
    assert idx_frame != -1
    assert idx_frame < idx_autoridad


@pytest.mark.asyncio
async def test_cancelled_intro_token_never_starts_speech() -> None:
    session = MagicMock()
    intro_orchestrator._run_token = 8

    with (
        patch("tasks.intro_orchestrator.kill_agent_speech", new=AsyncMock()),
        patch("tasks.intro_orchestrator.commit_guide_screen", new=AsyncMock()),
        patch("tasks.intro_orchestrator.speak_director_line", new=AsyncMock()) as speak,
    ):
        await intro_orchestrator._run_intro_tour(session, token=7)

    speak.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_intro_tour_stops_running_orchestrator() -> None:
    import asyncio

    intro_orchestrator._active_task = None
    intro_orchestrator._run_token = 0

    session = MagicMock()
    idle = asyncio.Event()

    async def block_until_cancelled(*_a, **_k):
        await idle.wait()

    with (
        patch(
            "tasks.intro_orchestrator.kill_agent_speech",
            new=block_until_cancelled,
        ),
        patch("tasks.intro_orchestrator.commit_guide_screen", new=AsyncMock()),
        patch("tasks.intro_orchestrator.speak_director_line", new=AsyncMock()) as speak,
        patch("tasks.intro_orchestrator.get_session_narration_barrier") as barrier,
    ):
        barrier.return_value.invalidate = MagicMock()
        assert intro_orchestrator.schedule_intro_tour(session) is True
        assert intro_orchestrator.intro_tour_running() is True
        task = intro_orchestrator._active_task
        assert task is not None

        intro_orchestrator.cancel_intro_tour()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert intro_orchestrator.intro_tour_running() is False
        speak.assert_not_awaited()
        barrier.return_value.invalidate.assert_called_once()
