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

                Audience: C-level executives — credible, consultative, grounded in
                real report data (facts.evidence / facts.gaps / facts.tactics).

                Always ground speech in content.activeDimension / presentation.
                The UI auto-advances sections after each narration — do NOT call
                navigate_journey(advance) between sections.

                First entry: present_content(detail_section, section=strengths) once.
                Later [pantalla:detail] cues: get_session_state only — no present_content.

                PROHIBITED section labels: «Fortalezas», «Oportunidades», «Plan de acción».
                Synthesize 2–3 paced sentences; never enumerate items verbatim.

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
            target="detail_section",
            index=0,
            dimension_id=dimension_id,
            section=section,  # type: ignore[arg-type]
            fallback_speak=active.get("summary") or dimension_id,
            extra_instructions=(
                "Audiencia C-level. Compón desde facts.hint — evidencia concreta del "
                "informe, tono de consultor senior. PROHIBIDO rótulos de sección. "
                "2–3 oraciones pausadas; no enumeres."
            ),
        )
