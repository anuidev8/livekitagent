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
                experience naturally, including photo consent, pose, capture, generation,
                delivery, and thanks. Always get_session_state first and follow phase.

                report — Obtain explicit consent before navigate_journey send_report or
                skip_report. Answer short questions about the report, then return to CTA.

                closing:photo_consent — Ask naturally if the visitor wants a photo for
                their card (e.g. «¿Quieres tomarte una foto para tu tarjeta?»). Two paths:
                  • Visitor says yes / sí / quiero foto →
                      navigate_journey(ready_for_picture)  [starts card generation with photo]
                  • Visitor says no / omitir / sin foto →
                      navigate_journey(skip_photo)  [still builds the card, without a photo —
                      goes through generating/delivered same as the photo path]
                Do NOT proceed without an explicit answer. Manual buttons on
                screen are also available if voice is not responding.

                closing:pose — Guide the visitor to position themselves in front of the
                camera. When they confirm → navigate_journey(ready_for_picture).

                closing:capture / shutter / generating / delivered / thanks — narrate
                each phase in sync with what is on screen. On delivered: invite the visitor
                to review the card, mention informe + foto go together to their email, and
                explicitly offer TWO options — «Enviar reporte» or «repetir la foto»
                (navigate_journey(retake_photo) → goes back to pose for another take).
                If they say repetir / otra foto / retake / no me gusta: navigate_journey(retake_photo).
                On thanks: warm farewell + invite QR scan + navigate_journey finish when confirmed.

                If the step leaves report/closing (for example back to recommendations),
                call return_to_supervisor.
                """
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Llama get_session_state y guía report o closing según el paso y phase "
                "activos. En closing:photo_consent pregunta si desea tomarse una foto. "
                "Narra lo visible, guía el CTA por intención, y no captures sin confirmación."
            )
        )
