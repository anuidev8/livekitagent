import asyncio
from unittest.mock import MagicMock, patch

import pytest
from livekit.agents import inference, llm

import agent as agent_module
from agent import (
    INSTRUCTIONS,
    MAIN_INSTRUCTIONS,
    NOVA_INSTRUCTIONS,
    NOVA_SESSION_REFRESH_SECONDS,
    NOVA_TURN_DETECTION,
    Assistant,
    NovaAssistant,
    _closing_pantalla_instructions,
    _deliver_pantalla_reply,
    _PantallaGuard,
    _pending_pantalla_replies,
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
        "answer_seti_question",
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
    # Regression: a real session (2026-09-02 logs) had the guide narrate the
    # SETI knowledge-base tool's perceived gaps out loud ("aunque no se
    # proporcionaron detalles específicos...") instead of speaking
    # confidently from what the tool did return.
    assert "respuesta de la herramienta es incompleta" in nova_lower
    assert "answer_seti_question" in nova_lower
    # The visitor may ask broadly ("cuéntame de SETI") or narrowly ("qué
    # bancos son clientes"); the guide should lead with the six-area summary
    # only for the broad case, and go straight to detail for the narrow one.
    assert "resumen general" in nova_lower


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


@pytest.mark.skip(
    reason="Requires LiveKit Inference credits; Nova is the only voice backend."
)
async def test_unused_inference_judge_smoke() -> None:
    llm_inst = _judge_llm()
    assert llm_inst is not None


def test_deliver_pantalla_reply_speaks_immediately_when_idle() -> None:
    """No in-flight speech (None, or already done) — speak right away, no task."""
    session = MagicMock()
    session.current_speech = None
    _deliver_pantalla_reply(session, "hola")
    session.generate_reply.assert_called_once_with(instructions="hola")
    assert not _pending_pantalla_replies

    session2 = MagicMock()
    done_handle = MagicMock()
    done_handle.done.return_value = True
    session2.current_speech = done_handle
    _deliver_pantalla_reply(session2, "hola de nuevo")
    session2.generate_reply.assert_called_once_with(instructions="hola de nuevo")
    assert not _pending_pantalla_replies


@pytest.mark.asyncio
async def test_deliver_pantalla_reply_waits_for_current_speech_then_speaks() -> None:
    """Speech in flight — defer to a background task, don't cut it off, don't
    interrupt once it finishes naturally within the grace period."""
    session = MagicMock()
    handle = MagicMock()
    handle.done.return_value = False

    async def _finishes_quickly() -> None:
        return None

    handle.wait_for_playout = _finishes_quickly
    session.current_speech = handle

    _deliver_pantalla_reply(session, "nuevo contenido")

    # Deferred, not fired synchronously — the whole point is not to chop off
    # the in-flight utterance.
    session.generate_reply.assert_not_called()
    assert len(_pending_pantalla_replies) == 1

    await asyncio.gather(*_pending_pantalla_replies)

    handle.interrupt.assert_not_called()
    session.generate_reply.assert_called_once_with(instructions="nuevo contenido")


@pytest.mark.asyncio
async def test_deliver_pantalla_reply_force_interrupts_after_grace_period() -> None:
    """Speech that runs long — bounded wait, then force-interrupt, then speak.
    This is the safety net so a stuck/long utterance can't indefinitely delay
    new screen content, mirroring the on_enter timeout fix."""
    session = MagicMock()
    handle = MagicMock()
    handle.done.return_value = False

    async def _never_finishes() -> None:
        await asyncio.sleep(10)

    handle.wait_for_playout = _never_finishes
    session.current_speech = handle

    with patch.object(agent_module, "_PANTALLA_INTERRUPT_GRACE_S", 0.05):
        _deliver_pantalla_reply(session, "contenido urgente")
        assert len(_pending_pantalla_replies) == 1
        await asyncio.gather(*_pending_pantalla_replies)

    handle.interrupt.assert_called_once()
    session.generate_reply.assert_called_once_with(instructions="contenido urgente")


def test_photo_consent_question_voices_the_decline_option() -> None:
    """User feedback: the agent only asked about taking a photo out loud —
    visitors who wanted to skip only found out that was possible from the
    on-screen buttons, never from what the agent said. The spoken question
    must mention BOTH options, not just the "yes, take a photo" branch."""
    consent = _closing_pantalla_instructions(
        "[pantalla:closing:photo_consent] step=closing phase=photo_consent"
    )
    assert consent is not None
    idx_question = consent.find("Pregunta")
    idx_no_branch = consent.find("no / omitir")
    assert idx_question != -1 and idx_no_branch != -1
    question_guidance = consent[idx_question:idx_no_branch]
    assert "sin foto" in question_guidance or "prefiere" in question_guidance.lower()

    idx_q = NOVA_INSTRUCTIONS.find("¿Quieres tomarte")
    assert idx_q != -1
    quoted_end = NOVA_INSTRUCTIONS.find("»", idx_q)
    quoted_question = NOVA_INSTRUCTIONS[idx_q:quoted_end]
    assert "sin foto" in quoted_question


def test_closing_thanks_finish_tool_called_before_farewell_speech() -> None:
    """Regression: a real session (2026-09-02 08:16 logs) showed the visitor
    say "finalizar" four separate times in a row and navigate_journey(finish)
    never fired once. Root cause: both instruction sites told the model to
    speak the farewell line THEN call the tool. Nova Sonic's own barge-in
    detection kept cutting the generation off mid-farewell (the impatient
    visitor talking over it, since nothing visibly happened yet) before the
    trailing tool call was ever reached. The fix mirrors the existing
    photo_consent "LLAMA ... PRIMERO" pattern: tool call before speech, so a
    barge-in after the tool call already fired can no longer swallow it.
    """
    thanks = _closing_pantalla_instructions(
        "[pantalla:closing:thanks] step=closing phase=thanks"
    )
    assert thanks is not None
    assert "navigate_journey(finish)" in thanks
    assert "PRIMERO" in thanks
    # The old buggy ordering must not reappear.
    assert "despedida y navigate_journey(finish)" not in thanks

    nova = NOVA_INSTRUCTIONS
    assert "despedida y navigate_journey(finish)" not in nova
    idx = nova.find("confirmen salir")
    assert idx != -1
    snippet = nova[idx : idx + 260]
    assert "navigate_journey(finish)" in snippet
    assert "PRIMERO" in snippet


def test_retake_photo_resets_generating_and_delivered_guards() -> None:
    """Regression (2026-09-02, RM_SpsHnphyUjch logs): a visitor retook their
    card photo, then said "quiero enviar el reporte" / "enviar" many times
    while the agent kept repeating "se están enviando a tu correo" without
    ever calling navigate_journey(advance). Root cause: retake_photo only
    forgot the "closing:photo" once-only pantalla guard, so the second pass
    through closing:generating and closing:delivered (after the retake) was
    silently swallowed by the once-only guard — the model never received
    fresh CLOSING DELIVERED instructions telling it to call
    navigate_journey(advance), so it just improvised stalling narration.
    """
    guard = _PantallaGuard()

    # First pass through the closing cycle: both cues get narrated once.
    guard.mark_narrated("closing:photo")
    guard.mark_narrated("closing:generating")
    guard.mark_narrated("closing:delivered")
    assert guard.already_narrated("closing:generating")
    assert guard.already_narrated("closing:delivered")

    # Visitor asks to retake the photo — this must reset every closing
    # once-key so the second generating/delivered pass narrates again,
    # not just the pose screen.
    guard.on_navigate_action("retake_photo")

    assert not guard.already_narrated("closing:photo")
    assert not guard.already_narrated("closing:generating")
    assert not guard.already_narrated("closing:delivered")


@pytest.mark.asyncio
async def test_answer_seti_question_tool_delegates_to_knowledge_base() -> None:
    agent = Assistant()
    result = await agent.answer_seti_question(
        context=None, query="¿qué servicios ofrece SETI?"
    )
    assert "Desarrollo" in result or "PRIME" in result
