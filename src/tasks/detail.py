"""Dimension detail deep-dive."""

from __future__ import annotations

import json
import textwrap

from rpc_client import rpc
from tasks.base import VOICE_RULES, HuellaPhaseTask
from tasks.ui_sync import present_and_speak


class DetailTask(HuellaPhaseTask):
    active_steps = frozenset({"detail"})

    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=textwrap.dedent(
                f"""\
                {VOICE_RULES}

                You own the detail screen only.

                Always ground speech in content.activeDimension / presentation.
                The runtime focuses the spider and section before you speak on enter.
                When the user switches dimension or section, call present_content
                detail_dimension / detail_section BEFORE narrating so UI stays synced.

                CTAs — Guide advance / back / send_report from availableActions by
                intent when they want the full plan or to leave detail.

                If the step is no longer detail, call return_to_supervisor.
                """
            ),
        )

    async def on_enter(self) -> None:
        raw = await rpc("get_session_state")
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            state = {}

        content = state.get("content") if isinstance(state.get("content"), dict) else {}
        active = content.get("activeDimension") if isinstance(content, dict) else None
        active = active if isinstance(active, dict) else {}
        dimension_id = str(active.get("id") or state.get("focusDimensionId") or "serp")
        section = str(state.get("detailSection") or "strengths")
        if section not in {"strengths", "opportunities", "action_plan"}:
            section = "strengths"

        await present_and_speak(
            self.session,
            target="detail_dimension",
            index=int(active.get("index") or 0),
            dimension_id=dimension_id,
            fallback_speak=str(active.get("summary") or dimension_id),
            extra_instructions=(
                "El spider ya enfoca esta dimensión. Resume score y contexto breve."
            ),
        )
        await present_and_speak(
            self.session,
            target="detail_section",
            index=0,
            section=section,  # type: ignore[arg-type]
            fallback_speak=section,
            extra_instructions=(
                "Narra solo la sección enfocada (fortalezas, oportunidades o plan) "
                "con el texto exacto del estado."
            ),
        )
