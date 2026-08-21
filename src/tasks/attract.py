"""Attract / NFC / validation opening phase."""

from __future__ import annotations

import asyncio
import json
import textwrap

from rpc_client import rpc
from tasks.base import VOICE_RULES, HuellaPhaseTask, PhaseResult
from tasks.speech import generate_reply_safe
from tasks.ui_sync import ATTRACT_CARD_SCRIPTS, present_and_speak


class AttractTask(HuellaPhaseTask):
    active_steps = frozenset({"attract", "nfc", "validation"})

    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=textwrap.dedent(
                f"""\
                {VOICE_RULES}

                You own the opening screens: attract, and optionally nfc or validation.

                attract — Warm human opening. Do NOT sound robotic or scripted.
                The interaction-card tour (Gestos, Toque, Voz) is driven by the
                runtime: present_content runs in code BEFORE you speak each card.
                When you speak about a card, it is already on screen — never invent
                cards that are not focused. Do NOT wait for the user to ask how to
                interact; the tour starts automatically after the title greeting.

                After the three cards, guide the CTA (empezar / avanzar / confirmar).
                When they want to begin, navigate_journey start_experience.

                Never invent a personal name on attract. Never hard-match keywords —
                understand intent.

                nfc — One clear instruction to approach the NFC band. No name.

                validation — Credential is being validated. No name.

                If get_session_state shows a step outside attract/nfc/validation, call
                return_to_supervisor.
                """
            ),
        )
        self._tour_started = False

    async def on_enter(self) -> None:
        raw = await rpc("get_session_state")
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            state = {}
        step = str(state.get("step") or "attract")

        if step in {"nfc", "validation"}:
            await generate_reply_safe(
                self.session,
                instructions=(
                    "En nfc: una instrucción clara para acercar la manilla. "
                    "En validation: confirma que la credencial se valida. Sin nombre. "
                    "No llames herramientas en este turno."
                ),
                tool_choice="none",
            )
            return

        # 1) Title-only greeting (no cards yet).
        await generate_reply_safe(
            self.session,
            instructions=(
                "Saludo cálido y humano solo sobre el título ¿Sabe qué dice "
                "internet de usted? Invita curiosidad en una o dos frases. "
                "Sin nombre. NO describas gestos, toque ni voz todavía. "
                "No llames herramientas en este turno."
            ),
            tool_choice="none",
        )

        # 2) Auto interaction tour — UI first, then voice.
        await self._run_attract_tour()

        # 3) CTA after cards.
        still = await self._still_on_attract()
        if not still:
            return
        await generate_reply_safe(
            self.session,
            instructions=(
                "Ya mostraste gestos, toque y voz en pantalla. Invita a empezar "
                "cuando quiera. Si pide empezar o avanzar, usa navigate_journey "
                "start_experience."
            ),
            tool_choice="auto",
        )

    async def _still_on_attract(self) -> bool:
        if self.done():
            return False
        try:
            data = json.loads(await rpc("get_session_state"))
        except Exception:
            return True
        step = str(data.get("step") or "")
        if step and step not in self.active_steps:
            self._safe_complete(
                PhaseResult(
                    summary=f"La pantalla cambió a {step}.",
                    next_step=step,
                )
            )
            return False
        return True

    async def _run_attract_tour(self) -> None:
        if self._tour_started:
            return
        self._tour_started = True

        for card in ATTRACT_CARD_SCRIPTS:
            if not await self._still_on_attract():
                return
            try:
                await present_and_speak(
                    self.session,
                    target="attract_card",
                    index=int(card["index"]),
                    fallback_speak=str(card["speak"]),
                    extra_instructions=(
                        f"Enfócate solo en {card['title']}. Una idea. Luego pausa corta."
                    ),
                )
            except Exception:
                # Session/realtime may have dropped; stop the tour cleanly.
                return
            await asyncio.sleep(0.45)
