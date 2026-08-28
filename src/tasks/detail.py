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

                Paraphrase for facts.role and facts.company — explain WHY each
                observation matters; never read facts.items verbatim.

                DETAIL_CONTINUOUS_TOUR — ONE uninterrupted locution for evidence →
                gaps → tactics. First entry: present_content(detail_section,
                section=strengths) once. Weave facts in flowing C-level prose —
                NO get_session_state or present_content between blocks.
                PROHIBITED: silence, pauses, or stopping between blocks OR items.
                Chain with connectors; never read facts.items one-by-one.
                UI highlights Fortalezas → Oportunidades → Plan alone — keep speaking.

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
                "DETAIL_CONTINUOUS_TOUR — UNA locución SIN silencios: evidencia → brechas → tácticas. "
                "Parafrasea facts para facts.role — explica POR QUÉ; PROHIBIDO pausas entre bloques/ítems. "
                "Encadena con «Además…», «Donde veo margen…», «En concreto…». "
                "PROHIBIDO rótulos de sección. UI resalta secciones sola — sigue hablando."
            ),
        )
