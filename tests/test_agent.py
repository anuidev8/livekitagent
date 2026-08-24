import pytest
from livekit.agents import inference, llm

from agent import (
    INSTRUCTIONS,
    MAIN_INSTRUCTIONS,
    NOVA_INSTRUCTIONS,
    NOVA_SESSION_REFRESH_SECONDS,
    Assistant,
    NovaAssistant,
)
from tasks import AnalysisTask, AttractTask


def _judge_llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


def test_nova_agent_keeps_stable_tools_without_handoffs() -> None:
    """Nova uses a fixed tool set; AgentTask handoffs are not used."""
    agent = Assistant()
    tool_names = {tool.info.name for tool in agent.tools}

    assert tool_names == {
        "get_session_state",
        "present_content",
        "navigate_journey",
        "set_control_channel",
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
    # Forbidden product framing must not appear as instructions to invent a game.
    assert "bienvenido al juego" not in nova_lower
    assert "qué dice internet" not in nova_lower
    for term in (
        "profile",
        "scores",
        "report",
        "consent",
        "photo",
        "cancel",
        "authority",
        "identity",
        "roles",
    ):
        assert term not in nova_lower, f"Nova prompt should omit {term!r}"
    assert "Obtiene el estado actual de la pantalla." in {
        tool.info.description for tool in agent.tools
    }


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


@pytest.mark.skip(reason="Requires LiveKit Inference credits; Nova is the only voice backend.")
async def test_unused_inference_judge_smoke() -> None:
    llm_inst = _judge_llm()
    assert llm_inst is not None
