"""Report consent (optional) + closing capture phase."""

from __future__ import annotations

import textwrap

from tasks.base import VOICE_RULES, HuellaPhaseTask


class ClosingTask(HuellaPhaseTask):
    active_steps = frozenset({"report", "closing"})

    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=textwrap.dedent(
                f"""\
                {VOICE_RULES}

                You own report (if active) and closing — guide the full end of the
                experience naturally, including preload, pose, capture, generation,
                delivery, and thanks. Always get_session_state first and follow phase.

                report — Obtain explicit consent before navigate_journey send_report or
                skip_report. Answer short questions about the report, then return to CTA.

                closing — Narrate each phase (prep, pose, capture, shutter, generating,
                delivered, thanks) in sync with what is on screen. Capture only after
                clear confirmation: navigate_journey ready_for_picture. Count slowly.
                Guide every CTA by intent (confirm pose, capture, finish, back when
                available). On thanks: warm thank-you by name, invite them to scan
                the QR to learn more about SETI, then navigate_journey finish when
                done. Do NOT re-narrate analysis, scores, or delivery details.

                If the step leaves report/closing (for example back to recommendations),
                call return_to_supervisor.
                """
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Llama get_session_state y guía report o closing según el paso y phase "
                "activos. Narra lo visible, guía el CTA por intención, y no captures "
                "sin confirmación."
            )
        )
