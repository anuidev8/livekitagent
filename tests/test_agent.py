import asyncio

import pytest
from livekit.agents import inference, llm

from agent import (
    INSTRUCTIONS,
    MAIN_INSTRUCTIONS,
    NOVA_INSTRUCTIONS,
    NOVA_SESSION_REFRESH_SECONDS,
    NOVA_TURN_DETECTION,
    Assistant,
    NovaAssistant,
    build_on_enter_instructions,
    on_enter_should_defer,
)
from tasks import AnalysisTask, AttractTask


def _judge_llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


def test_on_enter_should_defer_welcome_and_failed_state() -> None:
    assert on_enter_should_defer({"ok": True, "step": "welcome", "phase": "preparing"})
    assert on_enter_should_defer({"ok": True, "step": "welcome", "phase": "ready"})
    assert on_enter_should_defer({"ok": False, "step": "welcome", "phase": "ready"})
    assert not on_enter_should_defer({"ok": True, "step": "intro", "phase": "steps"})


def test_build_on_enter_instructions_are_short_and_state_aware() -> None:
    attract = build_on_enter_instructions({"step": "attract", "phase": "ready"})
    assert "present_content" in attract
    assert "get_session_state" not in attract
    assert len(attract) < 400

    welcome_ready = build_on_enter_instructions(
        {"step": "welcome", "phase": "ready", "title": "Huella Digital"}
    )
    # Only used when defer is false; keep it short if ever called.
    assert len(welcome_ready) < 500

    intro = build_on_enter_instructions(
        {
            "step": "intro",
            "phase": "steps",
            "facts": {"hint": "Narra la tarjeta actual."},
        }
    )
    assert "intro" in intro
    assert "Narra la tarjeta actual." in intro


def test_nova_agent_keeps_stable_tools_without_handoffs() -> None:
    """Nova uses a fixed tool set; AgentTask handoffs are not used."""
    agent = Assistant()
    tool_names = {tool.info.name for tool in agent.tools}

    assert tool_names == {
        "get_session_state",
        "present_content",
        "navigate_journey",
        "fill_search",
    }
    assert NovaAssistant is Assistant
    assert INSTRUCTIONS is MAIN_INSTRUCTIONS is NOVA_INSTRUCTIONS
    assert "on_enter" in Assistant.__dict__

    nova_lower = NOVA_INSTRUCTIONS.lower()
    assert "huella digital" in nova_lower
    assert "seti" in nova_lower
    assert "spokencontent" in nova_lower
    assert "no inventes" in nova_lower or "únicamente" in nova_lower
    assert "nunca digas que no ves" in nova_lower
    assert "livekit" not in nova_lower
    assert "get_session_state first" not in nova_lower
    assert NOVA_TURN_DETECTION in {"LOW", "MEDIUM", "HIGH"}
    assert "onboarding de 3 tarjetas" not in nova_lower
    # Forbidden product framing must not appear as instructions to invent a game.
    assert "bienvenido al juego" not in nova_lower
    assert "qué dice internet" not in nova_lower
    assert any(
        tool.info.description.startswith("Obtiene el estado actual de la pantalla")
        for tool in agent.tools
    )


def test_analysis_task_is_focused() -> None:
    task = AnalysisTask()
    assert "result_dimension" in task.instructions
    assert "scanning" in task.instructions
    assert "attract_tour" not in task.instructions
    tool_names = {tool.info.name for tool in task.tools}
    assert "present_content" in tool_names
    assert "navigate_journey" in tool_names
    assert "return_to_supervisor" in tool_names


def test_attract_task_is_focused() -> None:
    task = AttractTask()
    assert "Gestos" in task.instructions or "interaction-card" in task.instructions
    assert "start_experience" in task.instructions
    assert "recommendation_item" not in task.instructions
    assert "automatically" in task.instructions


def test_ui_sync_attract_scripts_cover_three_cards() -> None:
    from tasks.ui_sync import ATTRACT_CARD_SCRIPTS

    assert len(ATTRACT_CARD_SCRIPTS) == 3
    assert ATTRACT_CARD_SCRIPTS[0]["title"] == "Gestos"
    assert ATTRACT_CARD_SCRIPTS[1]["title"] == "Toque"
    assert ATTRACT_CARD_SCRIPTS[2]["title"] == "Voz"


def test_rpc_client_exposes_retry_knobs() -> None:
    import inspect

    from rpc_client import rpc

    sig = inspect.signature(rpc)
    assert "retries" in sig.parameters
    assert sig.parameters["retries"].default == 2


def test_nova_session_recycles_before_the_aws_timeout() -> None:
    """The SDK must see the app's renewal policy before its module import."""
    from livekit.plugins.aws.experimental.realtime import realtime_model

    assert NOVA_SESSION_REFRESH_SECONDS == 360
    assert realtime_model.MAX_SESSION_DURATION_SECONDS == 360


@pytest.mark.asyncio
async def test_nova_recycle_does_not_cancel_the_active_renewal() -> None:
    """Arming the next timer from a recycle must not cancel that recycle."""
    from livekit.plugins.aws.experimental.realtime import realtime_model

    next_timer_started = asyncio.Event()

    class FakeSession:
        _session_recycle_task: asyncio.Task[None] | None = None

        def _calculate_session_duration(self) -> float:
            return 360.0

        async def _session_recycle_timer(self, duration: float) -> None:
            assert duration == 360.0
            next_timer_started.set()

    fake = FakeSession()
    active_renewal = asyncio.current_task()
    assert active_renewal is not None
    fake._session_recycle_task = active_renewal

    realtime_model.RealtimeSession._start_session_recycle_timer(fake)  # type: ignore[arg-type]

    assert not active_renewal.cancelling()
    assert fake._session_recycle_task is not active_renewal
    await asyncio.wait_for(next_timer_started.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_nova_recycle_cancels_a_stale_independent_timer() -> None:
    """The compatibility fix retains upstream stale-timer cleanup."""
    from livekit.plugins.aws.experimental.realtime import realtime_model

    stale_timer = asyncio.create_task(asyncio.sleep(60))

    class FakeSession:
        _session_recycle_task: asyncio.Task[None] | None = stale_timer

        def _calculate_session_duration(self) -> float:
            return 360.0

        async def _session_recycle_timer(self, duration: float) -> None:
            return None

    fake = FakeSession()
    realtime_model.RealtimeSession._start_session_recycle_timer(fake)  # type: ignore[arg-type]

    await asyncio.sleep(0)
    assert stale_timer.cancelled()
    await fake._session_recycle_task


@pytest.mark.skip(reason="Requires LiveKit Inference credits; Nova is the only voice backend.")
async def test_unused_inference_judge_smoke() -> None:
    llm_inst = _judge_llm()
    assert llm_inst is not None
