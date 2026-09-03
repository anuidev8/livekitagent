import asyncio
from unittest.mock import MagicMock, patch

import pytest
from livekit.agents import inference, llm

import agent as agent_module
from agent import (
    _GENERATING_SETI_FACTS,
    _PANTALLA_INTERRUPT_GRACE_S,
    _SESSION_RECONNECTED_INSTRUCTIONS,
    _USER_VOICE_TOOL_HINT,
    INSTRUCTIONS,
    MAIN_INSTRUCTIONS,
    NOVA_INSTRUCTIONS,
    NOVA_SESSION_REFRESH_SECONDS,
    NOVA_TURN_DETECTION,
    Assistant,
    NovaAssistant,
    _closing_pantalla_instructions,
    _deliver_pantalla_reply,
    _generating_keepalive_instructions,
    _PantallaGuard,
    _pending_pantalla_replies,
    build_on_enter_instructions,
    on_enter_should_defer,
)
from nova_session_continuation import (
    _RECONNECT_EVENT_PATCH_MARKER,
    _with_session_reconnected_emit,
    install_nova_session_reconnected_event_fix,
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
        "select_search_result",
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


@pytest.mark.asyncio
async def test_reconnect_emit_wrapper_calls_original_then_emits() -> None:
    """Regression (2026-09-02, RM_vZnfXrLvRboG logs): a mid-call Nova recycle
    left the model speaking as if it were acting ("vamos a proceder con eso",
    "colócate frente al espejo para tomar la foto") but it never called
    get_session_state / present_content / navigate_journey again for the
    rest of the session — livekit-plugins-aws 1.7.0's recycle never emitted
    RealtimeSession's own documented "session_reconnected" event, so
    application code had no signal to react to. The wrapper must run the
    real recycle to completion, THEN emit session_reconnected exactly once —
    never before the recycle finishes, never on failure.
    """
    call_order: list[str] = []

    async def fake_original_recycle(self: object) -> None:
        call_order.append("recycle")
        await asyncio.sleep(0)

    class FakeSession:
        def __init__(self) -> None:
            self.emitted: list[tuple[str, object]] = []

        def emit(self, event: str, payload: object) -> None:
            call_order.append("emit")
            self.emitted.append((event, payload))

    wrapped = _with_session_reconnected_emit(fake_original_recycle)
    fake = FakeSession()
    await wrapped(fake)

    assert call_order == ["recycle", "emit"]
    assert len(fake.emitted) == 1
    event_name, payload = fake.emitted[0]
    assert event_name == "session_reconnected"
    from livekit.agents.llm.realtime import RealtimeSessionReconnectedEvent

    assert isinstance(payload, RealtimeSessionReconnectedEvent)


@pytest.mark.asyncio
async def test_reconnect_emit_wrapper_skips_emit_when_recycle_fails() -> None:
    """A failed recycle must propagate its error, not silently emit
    session_reconnected as if reconnection succeeded."""

    async def failing_recycle(self: object) -> None:
        raise RuntimeError("bedrock stream init failed")

    class FakeSession:
        def __init__(self) -> None:
            self.emitted: list[tuple[str, object]] = []

        def emit(self, event: str, payload: object) -> None:
            self.emitted.append((event, payload))

    wrapped = _with_session_reconnected_emit(failing_recycle)
    fake = FakeSession()
    with pytest.raises(RuntimeError, match="bedrock stream init failed"):
        await wrapped(fake)

    assert fake.emitted == []


def test_reconnect_event_fix_is_installed_and_idempotent() -> None:
    """agent.py installs this at import time (see agent_module import above).
    Calling it again must be a no-op that still reports success, matching
    the sibling _start_session_recycle_timer fix's idempotency contract."""
    from livekit.plugins.aws.experimental.realtime import realtime_model

    current = realtime_model.RealtimeSession._graceful_session_recycle
    assert getattr(current, _RECONNECT_EVENT_PATCH_MARKER, False)

    assert install_nova_session_reconnected_event_fix() is True
    # Re-installing must not wrap an already-wrapped method a second time.
    assert realtime_model.RealtimeSession._graceful_session_recycle is current


def test_session_reconnected_instructions_force_resync_before_anything_else() -> None:
    """The reinforcement instruction fired on session_reconnected must force
    a fresh get_session_state before the model reacts to anything else, ban
    assuming/improvising the screen from pre-reconnect memory, tell it to
    actually call the tool for anything it only claimed to do before the
    reconnect, and never surface the reconnect to the visitor."""
    instructions = _SESSION_RECONNECTED_INSTRUCTIONS
    assert "get_session_state" in instructions
    assert "PRIMERO" in instructions
    idx_get_state = instructions.find("get_session_state")
    idx_primero = instructions.find("PRIMERO")
    assert idx_get_state < idx_primero
    assert "PROHIBIDO asumir" in instructions or "PROHIBIDO" in instructions
    assert "nunca la des por hecha" in instructions.lower()
    assert "nunca lo menciones" in instructions.lower()


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


def test_generating_keepalive_never_repeats_and_never_claims_completion() -> None:
    """Feedback: while the card is generating, visitors heard the agent
    either go dead silent for long stretches or repeat the same filler
    phrase. Each keepalive tick must (a) carry a distinct SETI fact so
    consecutive ticks never sound the same, (b) explicitly forbid repeating
    prior phrasing, and (c) never claim the card/report is ready — that
    would resurrect the original "sending" hallucination bug.
    """
    assert len(_GENERATING_SETI_FACTS) >= 4
    assert len(set(_GENERATING_SETI_FACTS)) == len(_GENERATING_SETI_FACTS)

    seen_facts = set()
    for tick in range(len(_GENERATING_SETI_FACTS)):
        instructions = _generating_keepalive_instructions(tick)
        fact = _GENERATING_SETI_FACTS[tick]
        assert fact in instructions
        seen_facts.add(fact)
        assert "PROHIBIDO repetir" in instructions
        assert "PROHIBIDO ABSOLUTO" in instructions
        assert "listos" in instructions or "listo" in instructions

    # Every tick in one full cycle used a different fact.
    assert len(seen_facts) == len(_GENERATING_SETI_FACTS)

    # Cycling past the end of the list wraps around rather than crashing.
    wrapped = _generating_keepalive_instructions(len(_GENERATING_SETI_FACTS))
    assert _GENERATING_SETI_FACTS[0] in wrapped


def test_generating_keepalive_grace_period_covers_longest_fact() -> None:
    """Regression (2026-09-02, RM_3HK2n8CFPegT logs): the closing:delivered
    handoff force-interrupted an in-flight SETI fact mid-word
    ("...bajo el propósito «Crecemos para") because the shared pantalla
    grace period (8s) was sized for the old one-line filler, not these
    longer grounded facts. Guards against the grace period regressing
    below what the longest fact needs at a conservative spoken pace, and
    against reintroducing the verbose transition preamble that pushed the
    original line over budget.
    """
    # Conservative: slower than the ~3.25 words/s implied by the incident
    # (26 words spoken in the 8s before the cut), so this is a safety
    # margin check, not a tight fit.
    words_per_second = 2.5
    longest_fact_words = max(len(fact.split()) for fact in _GENERATING_SETI_FACTS)
    # +6 words of headroom for whatever short lead-in the model adds.
    estimated_seconds = (longest_fact_words + 6) / words_per_second
    assert estimated_seconds <= _PANTALLA_INTERRUPT_GRACE_S

    # The instructions must no longer suggest the long transition preamble
    # that ate into the speaking budget in the incident.
    instructions = _generating_keepalive_instructions(0)
    assert "mientras se termina de armar tu tarjeta" not in instructions.lower()
    assert "sin preámbulos largos" in instructions


def test_generating_keepalive_requires_short_transition_not_cold_open() -> None:
    """Feedback (2026-09-02, RM_MeiLzPnKgbwA logs): the SETI fact landed as a
    cold, near-verbatim recitation with no lead-in at all — it read as a
    company ad interrupting the wait, not as a "keeping you company" aside.
    The earlier fix banned the old long preamble ("mientras se termina de
    armar tu tarjeta...") for eating into the speaking budget and getting the
    fact cut off mid-word. This asserts the replacement guidance requires a
    much shorter transition (2-4 words) instead of swinging to "no
    transition at all" — with a concrete example so the model isn't guessing.
    """
    instructions = _generating_keepalive_instructions(0)
    assert "transici" in instructions.lower()
    assert "2" in instructions and "4" in instructions
    assert "mientras tanto" in instructions.lower()
    # Must not regress into the long banned preamble, and must keep the
    # "no long preambles" ceiling from the earlier fix.
    assert "mientras se termina de armar tu tarjeta" not in instructions.lower()
    assert "sin preámbulos largos" in instructions


def test_delivered_voice_hint_covers_first_time_photo_request() -> None:
    """Regression (2026-09-02, RM_vZnfXrLvRboG logs): a visitor who had
    skipped the photo earlier said "quiero tomarme una foto" at
    closing:delivered. The per-turn voice hint's retake_photo trigger list
    only recognized "repeat" phrasing (repetir, otra foto, retake...), so
    the model verbally agreed ("Entiendo que quieres tomarte una foto... "
    "Colócate frente al espejo...") without ever calling
    navigate_journey(retake_photo) — the screen never moved, the visitor
    was left staring at the same delivered card. The hint must also
    recognize wanting a first photo (not just a repeat) as a trigger for
    the same tool.
    """
    hint = _USER_VOICE_TOOL_HINT
    idx = hint.find("phase=delivered y quiere una foto")
    assert idx != -1
    window = hint[idx : idx + 400]
    assert "quiero tomarme una foto" in window
    assert "navigate_journey(retake_photo)" in window


def test_delivered_narration_does_not_claim_photo_when_skipped() -> None:
    """Regression (same session as above): with facts.photoSkipped true,
    the agent said "tu informe, junto con la imagen, viajarán juntos a tu
    correo" — false, no photo was ever taken. The old opening line
    unconditionally claimed "informe y foto van juntos"; it must instead
    branch on facts.photoSkipped and say nothing about a photo/image when
    none was taken.
    """
    delivered = _closing_pantalla_instructions(
        "[pantalla:closing:delivered] step=closing phase=delivered"
    )
    assert delivered is not None
    idx_skipped = delivered.find("photoSkipped es true")
    idx_not_skipped = delivered.find("photoSkipped es false")
    assert idx_skipped != -1
    assert idx_not_skipped != -1
    assert idx_skipped < idx_not_skipped

    intro = delivered[:idx_skipped]
    assert "informe y foto" not in intro.lower()
    assert "informe e imagen" not in intro.lower()

    skipped_block = delivered[idx_skipped:idx_not_skipped]
    assert "sin mencionar foto" in skipped_block.lower()


@pytest.mark.asyncio
async def test_answer_seti_question_tool_delegates_to_knowledge_base() -> None:
    agent = Assistant()
    result = await agent.answer_seti_question(
        context=None, query="¿qué servicios ofrece SETI?"
    )
    assert "Desarrollo" in result or "PRIME" in result
