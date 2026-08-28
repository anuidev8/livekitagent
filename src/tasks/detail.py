"""Dimension detail deep-dive."""

from __future__ import annotations

import textwrap

from tasks.base import VOICE_RULES, HuellaPhaseTask


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
                gaps → tactics. First entry ONLY: present_content(detail_section,
                section=strengths) once via [pantalla:detail:continuous]. Weave facts
                in flowing C-level prose — NO get_session_state or present_content
                between blocks.
                PROHIBITED: silence, pauses, or stopping between blocks OR items.
                Chain with connectors; never read facts.items one-by-one.
                UI highlights Fortalezas → Oportunidades → Plan alone — keep speaking.
                End with ONE choice question: report, back to globe, or another
                dimension — then STOP and WAIT.

                DETAIL_REVISIT (facts.detailTourComplete): PROHIBITED repeating the
                detail monologue. Offer the three choices in one brief sentence only.

                CTAs — send_report / back / open_detail from availableActions by
                visitor intent. PROHIBIDO re-narrate a dimension already covered.

                If the step is no longer detail, call return_to_supervisor.
                """
            ),
        )

    async def on_enter(self) -> None:
        # Speech is driven by client [pantalla:detail:continuous|revisit] cues —
        # avoid duplicate present_and_speak here (double narration).
        return
