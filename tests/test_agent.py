import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit.agents import inference, llm

from agent import (
    _GENERATING_SETI_FACTS,
    _PANTALLA_INTERRUPT_GRACE_NAV_S,
    _PANTALLA_INTERRUPT_GRACE_S,
    _SESSION_RECONNECTED_INSTRUCTIONS,
    _USER_VOICE_TOOL_HINT,
    DIMENSION_LABELS,
    INSTRUCTIONS,
    MAIN_INSTRUCTIONS,
    NOVA_INSTRUCTIONS,
    NOVA_SESSION_REFRESH_SECONDS,
    NOVA_TURN_DETECTION,
    Assistant,
    NovaAssistant,
    _analysis_pantalla_instructions,
    _closing_pantalla_instructions,
    _deliver_pantalla_reply,
    _generating_keepalive_instructions,
    _log_rpc_state_summary,
    _pantalla_dedupe_key,
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


@pytest.mark.asyncio
async def test_analysis_task_is_focused() -> None:
    task = AnalysisTask()
    assert "result_dimension" in task.instructions
    assert "scanning" in task.instructions
    assert "attract_tour" not in task.instructions
    tool_names = {tool.info.name for tool in task.tools}
    assert "present_content" in tool_names
    assert "navigate_journey" in tool_names
    assert "return_to_supervisor" in tool_names


@pytest.mark.asyncio
async def test_attract_task_is_focused() -> None:
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


def test_session_reconnected_instructions_reanchor_seti_question_handling() -> None:
    """Regression (2026-09-08, RM_NouknjNQXa5s logs): a Nova recycle landed
    right as the visitor entered closing:thanks and kept asking about SETI
    ("qué hace SETI", "dime los servicios"). Post-reconnect, the guide
    answered from stale memory a couple of times, then started refusing with
    banned phrases ("Lo siento, pero no puedo responder preguntas sobre mis
    propias funciones o capacidades") — misreading a 2nd-person "qué haces"
    about the company as a question about itself. The reconnect reinforcement
    only re-anchored screen/tool state, never the SETI-question contract or
    the anti-refusal rule, so both silently lapsed for the rest of the
    session. Both must be re-asserted every time the session reconnects, not
    just once at the start of the call — SETI context has to hold up
    anywhere in the flow, not only where it first came up."""
    instructions = _SESSION_RECONNECTED_INSTRUCTIONS
    assert "answer_seti_question" in instructions
    assert "segunda persona" in instructions.lower()
    assert "no sobre ti" in instructions.lower()
    assert "lo siento" in instructions.lower()
    assert "no puedo responder" in instructions.lower()
    # Must still come after the get_session_state-first reinforcement, not
    # replace it.
    assert instructions.index("get_session_state") < instructions.index(
        "answer_seti_question"
    )


@pytest.mark.skip(
    reason="Requires LiveKit Inference credits; Nova is the only voice backend."
)
async def test_unused_inference_judge_smoke() -> None:
    llm_inst = _judge_llm()
    assert llm_inst is not None


class _FakeSpeechHandle:
    def __init__(self, *, done: bool = True) -> None:
        self._done = done
        self.interrupt = MagicMock()
        self.wait_for_playout = AsyncMock()

    def done(self) -> bool:
        return self._done


def _resolved_future() -> asyncio.Future:
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    fut.set_result(None)
    return fut


def _mock_session(*, current_speech=None) -> MagicMock:
    session = MagicMock()
    session.current_speech = current_speech
    session.interrupt = MagicMock(return_value=_resolved_future())
    session.output.audio = MagicMock()
    session.wait_for_idle = AsyncMock()
    session.current_agent = MagicMock()
    session.current_agent.chat_ctx = MagicMock()
    session.current_agent.chat_ctx.copy.return_value.truncate.return_value = MagicMock(
        name="fresh_ctx"
    )
    return session


@pytest.mark.asyncio
async def test_deliver_pantalla_reply_speaks_immediately_when_idle() -> None:
    session = _mock_session(current_speech=None)

    with patch("agent._commit_guide_screen", new=AsyncMock()) as commit:
        _deliver_pantalla_reply(session, "hola", dedupe_key="analysis:results")
        await asyncio.sleep(0.4)

    session.interrupt.assert_called_once_with(force=True)
    session.output.audio.clear_buffer.assert_called()
    session.wait_for_idle.assert_awaited()
    commit.assert_awaited_once_with("analysis:results")
    session.generate_reply.assert_called_once()
    assert session.generate_reply.call_args.kwargs["instructions"] == "hola"
    assert "chat_ctx" in session.generate_reply.call_args.kwargs


@pytest.mark.asyncio
async def test_deliver_pantalla_reply_uses_fresh_chat_ctx() -> None:
    session = _mock_session(current_speech=None)
    fresh = MagicMock(name="fresh_ctx")

    with (
        patch("agent._commit_guide_screen", new=AsyncMock()),
        patch("agent.build_pantalla_chat_ctx", return_value=fresh),
    ):
        _deliver_pantalla_reply(
            session, "ANALYSIS SCAN only", dedupe_key="analysis:scanning"
        )
        await asyncio.sleep(0.4)

    session.generate_reply.assert_called_once_with(
        instructions="ANALYSIS SCAN only",
        chat_ctx=fresh,
    )


@pytest.mark.asyncio
async def test_deliver_pantalla_reply_commits_ui_before_speech() -> None:
    session = _mock_session(current_speech=None)
    order: list[str] = []

    async def _commit(key: str | None) -> None:
        order.append(f"commit:{key}")

    def _gen(**kwargs):
        order.append("speak")
        return MagicMock()

    session.generate_reply = MagicMock(side_effect=_gen)

    with patch("agent._commit_guide_screen", new=AsyncMock(side_effect=_commit)):
        _deliver_pantalla_reply(
            session, "dim higiene", dedupe_key="detail:continuous:higiene"
        )
        await asyncio.sleep(0.4)

    assert order == ["commit:detail:continuous:higiene", "speak"]


@pytest.mark.asyncio
async def test_deliver_pantalla_reply_drops_stale_reply_when_idle() -> None:
    session = _mock_session(current_speech=None)
    guard = _PantallaGuard()
    guard.is_duplicate("newer")

    with patch("agent._commit_guide_screen", new=AsyncMock()) as commit:
        _deliver_pantalla_reply(
            session, "stale", pantalla_guard=guard, dedupe_key="older"
        )
        await asyncio.sleep(0.4)

    session.generate_reply.assert_not_called()
    commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_pantalla_reply_nav_grace_kills_current_speech() -> None:
    current = _FakeSpeechHandle(done=False)
    session = _mock_session(current_speech=current)

    with patch("agent._commit_guide_screen", new=AsyncMock()):
        _deliver_pantalla_reply(session, "nueva pantalla", grace_s=0.0)
        await asyncio.sleep(0.4)

    session.interrupt.assert_called_once_with(force=True)
    session.output.audio.clear_buffer.assert_called()
    current.interrupt.assert_not_called()
    session.generate_reply.assert_called_once()
    assert session.generate_reply.call_args.kwargs["instructions"] == "nueva pantalla"


@pytest.mark.asyncio
async def test_deliver_pantalla_reply_cancels_superseded_pending_tasks() -> None:

    session = _mock_session(current_speech=None)
    blocker = asyncio.Event()

    async def _never() -> None:
        await blocker.wait()

    stale = asyncio.create_task(_never())
    _pending_pantalla_replies.add(stale)

    with patch("agent._commit_guide_screen", new=AsyncMock()):
        _deliver_pantalla_reply(session, "fresh", dedupe_key="analysis:results")
        await asyncio.sleep(0.4)

    assert stale.cancelled()
    assert stale not in _pending_pantalla_replies
    session.generate_reply.assert_called_once()
    assert session.generate_reply.call_args.kwargs["instructions"] == "fresh"
    blocker.set()


@pytest.mark.asyncio
async def test_deliver_pantalla_reply_waits_out_grace_then_speaks() -> None:
    current = _FakeSpeechHandle(done=False)
    current.wait_for_playout = AsyncMock(return_value=None)
    session = _mock_session(current_speech=current)

    with patch("agent._commit_guide_screen", new=AsyncMock()):
        _deliver_pantalla_reply(session, "filler", grace_s=1.0)
        await asyncio.sleep(0.4)

    session.interrupt.assert_called_once_with(force=True)
    session.generate_reply.assert_called_once()
    assert session.generate_reply.call_args.kwargs["instructions"] == "filler"


@pytest.mark.asyncio
async def test_commit_guide_screen_calls_rpc() -> None:
    with patch("rpc_client.rpc", new=AsyncMock(return_value='{"ok":true}')) as rpc_mock:
        from rpc_client import commit_guide_screen

        await commit_guide_screen("intro:run")

    rpc_mock.assert_awaited_once()
    assert rpc_mock.await_args.args[0] == "commit_guide_screen"
    assert rpc_mock.await_args.args[1] == {"key": "intro:run"}


def test_deliver_pantalla_reply_defaults_to_responsive_nav_grace() -> None:
    """Touch navigation defaults to immediate interrupt (grace 0).
    Long grace is opt-in only for generating filler."""
    import inspect

    default = inspect.signature(_deliver_pantalla_reply).parameters["grace_s"].default
    assert default == _PANTALLA_INTERRUPT_GRACE_NAV_S
    assert _PANTALLA_INTERRUPT_GRACE_NAV_S == 0.0
    assert _PANTALLA_INTERRUPT_GRACE_NAV_S < _PANTALLA_INTERRUPT_GRACE_S


def test_analysis_scanning_instructions_forbid_welcome_replay() -> None:
    """Button-nav past welcome must not re-greet on analysis:scanning
    (2026-09-04_12-44-40 logs)."""
    text = _analysis_pantalla_instructions("analysis:scanning")
    assert text is not None
    assert "PROHIBIDO ABSOLUTO" in text
    assert "bienvenida" in text.lower() or "welcome" in text.lower()
    assert "lector" in text.lower() or "NFC" in text
    assert "SCAN" in text


def test_analysis_results_return_is_brief() -> None:
    text = _analysis_pantalla_instructions("analysis:results")
    assert text is not None
    assert "De vuelta" in text or "volver" in text.lower()
    assert "PROHIBIDO" in text


def test_pantalla_dedupe_key_closing_review() -> None:
    key = _pantalla_dedupe_key("[pantalla:closing:review] step=closing phase=review.")
    assert key == "closing:review"


def test_pantalla_dedupe_key_analysis_scanning() -> None:
    key = _pantalla_dedupe_key(
        "[pantalla:analysis:scanning] UI step=analysis phase=scanning."
    )
    assert key == "analysis:scanning"


def test_detail_continuous_dedupe_includes_dimension() -> None:
    cue = (
        "[pantalla:detail:continuous] DETAIL_CONTINUOUS_TOUR — "
        "UNA sola respuesta CONTINUA DIM_ID=ssi"
    )
    assert _pantalla_dedupe_key(cue) == "detail:continuous:ssi"


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


def test_closing_delivered_advance_tool_called_before_confirmation_speech() -> None:
    """Regression (2026-09-04 13:14 logs): a visitor said "enviar" on the
    closing:delivered card and the agent replied "Gracias, tu tarjeta e
    informe ya estan en camino a tu correo" — but no navigate_journey call
    was ever emitted that turn, so /api/session/send-report never fired and
    nothing was actually sent. Root cause: unlike the photo_consent and
    thanks instructions (which already say "LLAMA ... PRIMERO, antes de
    decir cualquier palabra"), the closing:delivered instructions only said
    *when* to call navigate_journey(advance), never that it must happen
    before any confirmation speech. Mirrors the existing finish/back
    "tool call moves the screen, your voice alone does not" pattern.
    """
    delivered = _closing_pantalla_instructions(
        "[pantalla:closing:delivered] step=closing phase=delivered"
    )
    assert delivered is not None
    assert "navigate_journey(advance)" in delivered
    idx = delivered.find("CUANDO CONFIRME ENVIAR")
    assert idx != -1
    snippet = delivered[idx : idx + 500]
    assert "PRIMERO" in snippet
    assert "ok:true" in snippet

    nova = NOVA_INSTRUCTIONS
    idx_nova = nova.find("CUANDO CONFIRME ENVIAR")
    assert idx_nova != -1
    nova_snippet = nova[idx_nova : idx_nova + 500]
    assert "navigate_journey(advance)" in nova_snippet
    assert "PRIMERO" in nova_snippet
    assert "ok:true" in nova_snippet


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


def test_closing_review_asks_once_and_never_claims_to_see_the_photo() -> None:
    """The frontend's `review` phase (huella-digital ClosingView.tsx) shows the
    just-captured photo and waits for confirm_portrait/retake_photo before the
    card is built. The agent cannot see the photo — it must ask a single
    yes/no question and route the answer to the right tool, never describe
    or judge the image itself (that would be a hallucination)."""
    review = _closing_pantalla_instructions(
        "[pantalla:closing:review] step=closing phase=review"
    )
    assert review is not None
    assert "navigate_journey(confirm_portrait)" in review
    assert "navigate_journey(retake_photo)" in review
    assert "no puedes verla" in review
    assert "UNA SOLA VEZ" in review

    hint = _USER_VOICE_TOOL_HINT
    idx = hint.find("phase=review")
    assert idx != -1
    snippet = hint[idx : idx + 260]
    assert "navigate_journey(confirm_portrait)" in snippet
    assert "navigate_journey(retake_photo)" in snippet


def test_retake_photo_resets_review_guard_too() -> None:
    """Same regression class as test_retake_photo_resets_generating_and_delivered_guards,
    but for the new `review` step inserted between capture and generating: a
    visitor who retakes their photo from `delivered` walks pose → capture →
    review → generating → delivered again. If "closing:review" isn't forgotten
    on retake_photo, the once-only guard would silently skip asking about the
    NEW photo the second time through review.
    """
    guard = _PantallaGuard()

    guard.mark_narrated("closing:photo")
    guard.mark_narrated("closing:review")
    guard.mark_narrated("closing:generating")
    guard.mark_narrated("closing:delivered")

    guard.on_navigate_action("retake_photo")

    assert not guard.already_narrated("closing:photo")
    assert not guard.already_narrated("closing:review")
    assert not guard.already_narrated("closing:generating")
    assert not guard.already_narrated("closing:delivered")


def test_generating_keepalive_never_claims_completion() -> None:
    """While the card generates, speak the fixed SETI purpose line and never
    claim the card/report is ready.
    """
    assert len(_GENERATING_SETI_FACTS) >= 1
    assert "crecemos para nuestros clientes" in _GENERATING_SETI_FACTS[0].lower()

    for tick in range(len(_GENERATING_SETI_FACTS)):
        instructions = _generating_keepalive_instructions(tick)
        assert _GENERATING_SETI_FACTS[tick] in instructions
        assert "PROHIBIDO ABSOLUTO" in instructions
        assert "listos" in instructions or "listo" in instructions

    wrapped = _generating_keepalive_instructions(len(_GENERATING_SETI_FACTS))
    assert _GENERATING_SETI_FACTS[0] in wrapped


def test_generating_keepalive_grace_period_covers_longest_fact() -> None:
    """Grace must cover the full SETI purpose line + short lead-in."""
    words_per_second = 2.5
    longest_fact_words = max(len(fact.split()) for fact in _GENERATING_SETI_FACTS)
    estimated_seconds = (longest_fact_words + 6) / words_per_second
    assert estimated_seconds <= _PANTALLA_INTERRUPT_GRACE_S

    instructions = _generating_keepalive_instructions(0)
    assert "mientras se termina de armar tu tarjeta" not in instructions.lower()


def test_generating_keepalive_requires_mientras_lead_in() -> None:
    """Keep a «mientras tanto» framing so the SETI line is not a cold open."""
    instructions = _generating_keepalive_instructions(0)
    assert "mientras tanto" in instructions.lower()
    assert "mientras se genera tu tarjeta" in instructions.lower()
    assert "mientras se termina de armar tu tarjeta" not in instructions.lower()


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


def test_detail_revisit_always_narrates_with_varied_closing() -> None:
    """User feedback (2026-09-04): revisiting an already-toured dimension
    used to skip straight to "quieres el informe, volver, u otra dimensión?"
    with zero narration — a past fix to avoid tedious verbatim repeats, but
    the visitor found a bare question with no acknowledgment of the
    dimension jarring. Every detail entry, first visit or not, must narrate
    — the old "PROHIBIDO repetir el detalle" rule must be gone, replaced by
    "always narrate again, but vary the wording" — and the closing question
    must vary instead of repeating the identical phrase every dimension.
    """
    assert (
        "PROHIBIDO repetir el detalle de una dimensión ya narrada"
        not in NOVA_INSTRUCTIONS
    )
    idx = NOVA_INSTRUCTIONS.find("SIEMPRE narra de nuevo")
    assert idx != -1
    snippet = NOVA_INSTRUCTIONS[idx : idx + 700]
    assert "ángulo" in snippet or "angulo" in snippet
    assert "VARÍA esa pregunta" in NOVA_INSTRUCTIONS

    # Regression (2026-09-04 17:16 logs): even with the "always narrate again"
    # rule in place, the model composed "Ya hemos cubierto esta dimensión.
    # Quieres el informe..." on a real SSI revisit — using its own memory of
    # the conversation as an excuse to skip narration despite the explicit
    # instruction. The exact phrase it used must be named as forbidden, not
    # just "vary the wording" (which it was already told and ignored).
    assert "ya cubrimos/cubierto esta dimensión" in snippet
    assert "ya revisamos/revisado esta dimensión" in snippet


def test_gate_rejections_always_produce_speech() -> None:
    """Voice UX recommendation (docs/huella-guide-voice-ux-recommendation.md,
    Codebase §2): navigate_journey/present_content gate rejections
    (must_speak_welcome_first, ui_owns_detail_section_advance,
    action_not_available, present_failed) carry a `hint` in the ok:false
    tool result, but nothing previously forced Nova to say anything back —
    a rejected command looked, from the visitor's side, identical to an
    unresponsive agent. The prompt must force a short spoken line derived
    from the hint before silently retrying."""
    nova_lower = NOVA_INSTRUCTIONS.lower()
    idx = nova_lower.find("ok:false")
    assert idx != -1
    window = NOVA_INSTRUCTIONS[idx : idx + 500]
    window_lower = window.lower()
    assert "nunca" in window_lower and "silencio" in window_lower
    assert "hint" in window_lower
    # Must not instruct reading the raw hint field verbatim — compose it.
    assert "literal" in window_lower


def test_stale_swipe_gesture_block_removed() -> None:
    """Voice UX recommendation (Codebase §4): frontend swipe/gesture
    navigation was already removed; the UI never sends
    SWIPE_CONTINUE_REQUEST anymore, so this block was dead prompt weight
    describing input the UI no longer sends."""
    assert "SWIPE_CONTINUE_REQUEST" not in NOVA_INSTRUCTIONS
    assert "GESTOS (si preguntan" not in NOVA_INSTRUCTIONS


def test_dimension_trigger_labels_are_centralized() -> None:
    """Voice UX recommendation (Codebase §4): dimension names were typed
    twice as literal trigger phrases (NOVA_INSTRUCTIONS re-explain rule +
    _USER_VOICE_TOOL_HINT). A rename or a 6th dimension would silently go
    stale in one spot. Both trigger blocks must be built from one shared
    DIMENSION_LABELS constant instead of hand-typed strings in each place."""
    assert DIMENSION_LABELS == ("Autoridad", "SSI", "Mensaje", "Influencia", "Higiene")
    for label in DIMENSION_LABELS:
        assert label in NOVA_INSTRUCTIONS
        assert label in _USER_VOICE_TOOL_HINT


def test_log_rpc_state_summary_extracts_focus_fields(caplog) -> None:
    """The whole point of this helper is to make "what did the tool call
    resolve to" visible in logs — assert it actually surfaces the fields
    that identify which dimension/screen a response is about, and that it
    never raises on malformed input (called from real RPC round-trips)."""
    import json as _json
    import logging

    with caplog.at_level(logging.INFO, logger="agent"):
        _log_rpc_state_summary(
            "present_content",
            _json.dumps(
                {
                    "ok": True,
                    "step": "analysis",
                    "phase": "results",
                    "focusDimensionId": "arquitectura",
                    "analysisDimIndex": 2,
                }
            ),
        )
    assert any(
        "focusDimensionId=arquitectura" in r.message
        and "analysisDimIndex=2" in r.message
        for r in caplog.records
    )

    # Malformed / non-dict / non-JSON input must never raise.
    _log_rpc_state_summary("get_session_state", "not json")
    _log_rpc_state_summary("get_session_state", _json.dumps([1, 2, 3]))


def test_result_dimension_requires_explicit_dimension_target() -> None:
    """Regression (2026-09-04 15:49 logs): a visitor said "mensaje" on
    analysis:results; present_content(result_dimension) was called without a
    dimension_id, and the frontend's fallback resolved it to whichever
    dimension was already focused (left over from the visitor's previous
    stop) instead of Mensaje — so the guide narrated a different dimension's
    facts than the one the visitor named. NOVA_INSTRUCTIONS must forbid
    omitting dimensionId/index for result_dimension whenever the visitor
    named a dimension, since the frontend has no reliable way to recover
    the intended target from a bare "result_dimension" call.
    """
    idx = NOVA_INSTRUCTIONS.find("result_dimension SIEMPRE necesita")
    assert idx != -1
    snippet = NOVA_INSTRUCTIONS[idx : idx + 400]
    assert "dimensionId" in snippet
    assert "NUNCA lo omitas" in snippet


@pytest.mark.asyncio
async def test_answer_seti_question_tool_delegates_to_knowledge_base() -> None:
    agent = Assistant()
    result = await agent.answer_seti_question(
        context=None, query="¿qué servicios ofrece SETI?"
    )
    assert "Desarrollo" in result or "PRIME" in result


def test_welcome_ready_requires_data_consent_before_start() -> None:
    """Welcome ready must gate start_experience behind accept_data_consent."""
    assert "accept_data_consent" in NOVA_INSTRUCTIONS
    assert "PROHIBIDO: start_experience sin consentimiento" in NOVA_INSTRUCTIONS
    assert "accept_data_consent" in _USER_VOICE_TOOL_HINT
    assert "protección de datos" in _USER_VOICE_TOOL_HINT

    from tasks.welcome_orchestrator import build_welcome_instructions

    with_consent = build_welcome_instructions(
        {
            "step": "welcome",
            "phase": "ready",
            "availableActions": ["accept_data_consent", "back", "cancel"],
            "facts": {
                "name": "Ana Pérez",
                "role": "CTO",
                "company": "SETI",
                "dataConsentRequired": True,
            },
        }
    )
    assert "protección de datos" in with_consent
    assert "navigate_journey" in with_consent  # forbidden wording present
    assert "PROHIBIDO llamar navigate_journey" in with_consent

    after_consent = build_welcome_instructions(
        {
            "step": "welcome",
            "phase": "ready",
            "availableActions": ["start_experience", "back", "cancel"],
            "facts": {
                "name": "Ana Pérez",
                "role": "CTO",
                "company": "SETI",
                "dataConsentAccepted": True,
                "dataConsentRequired": False,
            },
        }
    )
    assert "¿Vemos cómo funciona?" in after_consent or "cómo funciona" in after_consent
    assert "protección de datos" not in after_consent


def test_generic_continue_word_is_never_treated_as_consent() -> None:
    """Regression: logs showed the model call accept_data_consent right after
    hearing "sí eh comenzar" — a generic continue phrase with no explicit
    "acepto" — then immediately call start_experience in the same turn,
    skipping real consent entirely. The prompt must explicitly rule this out.
    """
    for blob in (NOVA_INSTRUCTIONS, _USER_VOICE_TOOL_HINT):
        assert "CONFUSIÓN PROHIBIDA" in blob or "NUNCA equivalen a consentimiento" in blob
        assert "sí eh comenzar" in blob or "«sí, comencemos»" in blob or "sí, comencemos" in blob
        assert "acepto" in blob.lower()
        assert "autorizo" in blob.lower()
