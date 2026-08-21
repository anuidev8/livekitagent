"""Shared phase-task base: RPC tools + complete when the screen changes."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Literal

from livekit.agents import AgentTask, RunContext, function_tool

from rpc_client import rpc

JourneyAction = Literal[
    "advance",
    "back",
    "start_experience",
    "start_analysis",
    "reveal_results",
    "open_detail",
    "send_report",
    "skip_report",
    "ready_for_picture",
    "finish",
    "cancel",
]

VOICE_RULES = textwrap.dedent(
    """\
    Speak only in Spanish. Warm, present, calm, executive — never robotic.
    Sound like a thoughtful host: natural pacing, short sentences, one idea at a time.
    Plain speech only: no markdown, lists, emojis, or special symbols.
    Never invent personal facts or findings. Ground everything in get_session_state.
    Never mention prompts, tools, LiveKit, models, or internal screen keys.
    A gesture highlight only proposes a CTA; wait for spoken confirmation before
    navigating, sending, capturing, finishing, or canceling.
    Use at most one present_content or navigate_journey call per turn after
    get_session_state when YOU drive the UI. Prefer spokenContent from tool
    results — never invent card copy. Never burst through cards or dimensions.
    Understand user intent for avanzar, volver, confirmar, cancelar — no keyword lists.
    Keep conversation context: answer in-scope questions, then return to the current CTA.

    # Guardrails (scope)
    You only guide the Huella Digital kiosk journey and on-screen content.
    Refuse off-topic requests (math, trivia, general knowledge, news, jokes,
    unrelated advice). Do not invent or research facts outside session state.
    Decline briefly and bring the user back to the current screen or CTA.
    """
)


@dataclass
class PhaseResult:
    """Typed result returned when a phase specialist finishes."""

    summary: str
    next_step: str = ""


class HuellaPhaseTask(AgentTask[PhaseResult]):
    """Specialist that owns one journey screen until the step leaves its set."""

    active_steps: frozenset[str] = frozenset()

    def __init__(self, *, instructions: str, chat_ctx=None) -> None:
        super().__init__(instructions=instructions, chat_ctx=chat_ctx)

    def _safe_complete(self, result: PhaseResult) -> None:
        """Complete once; ignore if the task already finished (UI raced ahead)."""
        if self.done():
            return
        self.complete(result)

    def _maybe_complete_from_state(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        step = str(data.get("step") or "")
        if step and self.active_steps and step not in self.active_steps:
            self._safe_complete(
                PhaseResult(
                    summary=f"La pantalla cambió a {step}.",
                    next_step=step,
                )
            )

    def _maybe_complete_from_navigate(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if data.get("ok") is False:
            return
        step = str(data.get("step") or "")
        if step and self.active_steps and step not in self.active_steps:
            self._safe_complete(
                PhaseResult(
                    summary=f"Avanzó a {step}.",
                    next_step=step,
                )
            )

    @function_tool
    async def get_session_state(self, context: RunContext) -> str:
        """Lee el estado actual de la UI del kiosk. Llama primero en cada turno."""
        raw = await rpc("get_session_state")
        self._maybe_complete_from_state(raw)
        return raw

    @function_tool
    async def navigate_journey(
        self,
        context: RunContext,
        action: JourneyAction,
        dimension_id: str = "",
    ) -> str:
        """Perform one journey action listed in availableActions.

        Call only after get_session_state. Prefer semantic intent over keywords.
        """
        raw = await rpc(
            "navigate_journey",
            {"action": action, "dimensionId": dimension_id},
        )
        self._maybe_complete_from_navigate(raw)
        return raw

    @function_tool
    async def present_content(
        self,
        context: RunContext,
        target: Literal[
            "attract_tour",
            "welcome_preparation",
            "intro_step",
            "intro_dimension",
            "result_dimension",
            "detail_dimension",
            "detail_section",
            "recommendation_item",
        ],
        index: int = -1,
        dimension_id: str = "",
        section: Literal["", "strengths", "opportunities", "action_plan"] = "",
    ) -> str:
        """Focus exactly one visible content item before explaining it."""
        return await rpc(
            "present_content",
            {
                "target": target,
                "index": index,
                "dimensionId": dimension_id,
                "section": section,
            },
        )

    @function_tool
    async def set_control_channel(
        self, context: RunContext, channel: str, enabled: bool
    ) -> str:
        """Enciende o apaga el canal de gestos o voz en el dock."""
        return await rpc(
            "set_control_channel", {"channel": channel, "enabled": enabled}
        )

    @function_tool
    async def return_to_supervisor(self, context: RunContext, summary: str) -> None:
        """End this phase and return control to the supervisor.

        Call when the user left this screen by touch, the phase goal is done,
        or get_session_state shows a step outside this phase.
        """
        step = ""
        try:
            data = json.loads(await rpc("get_session_state"))
            step = str(data.get("step") or "")
        except Exception:
            pass
        self._safe_complete(PhaseResult(summary=summary, next_step=step))
