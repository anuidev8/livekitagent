"""Welcome preparing / ready phase."""

from __future__ import annotations

import asyncio
import json
import textwrap

from rpc_client import rpc
from tasks.base import VOICE_RULES, HuellaPhaseTask, PhaseResult
from tasks.speech import generate_reply_safe
from tasks.ui_sync import present_and_speak

_WELCOME_PREP = [
    {
        "index": 0,
        "speak": "Confirmando tu identidad con la credencial del evento.",
    },
    {
        "index": 1,
        "speak": "Personalizando tu perfil a partir de fuentes públicas.",
    },
    {
        "index": 2,
        "speak": "Preparando las cinco dimensiones de tu huella digital.",
    },
]


class WelcomeTask(HuellaPhaseTask):
    active_steps = frozenset({"welcome"})

    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=textwrap.dedent(
                f"""\
                {VOICE_RULES}

                You own the welcome screen only.

                phase preparing — The checklist is synced by the runtime with
                present_content before each line you speak. Narrate gently without
                saying the person's name, role, or company until ready.

                phase ready — NOW greet using the dynamic profile name, role, and
                company from get_session_state. One executive purpose sentence.
                Guide CTAs from availableActions (empezar / avanzar / volver) by
                intent, not keywords. navigate_journey start_experience only when
                they want to proceed; back/cancel when they want to leave.

                If the step is no longer welcome, call return_to_supervisor.
                """
            ),
        )
        self._prep_started = False

    async def on_enter(self) -> None:
        raw = await rpc("get_session_state")
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            state = {}

        phase = str(state.get("phase") or "preparing")
        if phase == "ready":
            await generate_reply_safe(
                self.session,
                instructions=(
                    "phase ready: saluda con el perfil dinámico de get_session_state "
                    "(nombre, cargo, empresa) y guía el CTA de empezar."
                ),
                tool_choice="auto",
                tools=["get_session_state", "navigate_journey"],
            )
            return

        await self._run_preparation_tour()

        if not await self._still_on_welcome():
            return
        # UI may already have flipped to ready via welcome_preparation index 2.
        await generate_reply_safe(
            self.session,
            instructions=(
                "Si phase es ready, saluda ya con nombre, cargo y empresa del "
                "perfil dinámico y guía el CTA. Si aún preparing, tranquiliza "
                "brevemente y espera."
            ),
            tool_choice="auto",
            tools=["get_session_state", "navigate_journey", "present_content"],
        )

    async def _still_on_welcome(self) -> bool:
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

    async def _run_preparation_tour(self) -> None:
        if self._prep_started:
            return
        self._prep_started = True

        for item in _WELCOME_PREP:
            if not await self._still_on_welcome():
                return
            await present_and_speak(
                self.session,
                target="welcome_preparation",
                index=int(item["index"]),
                fallback_speak=str(item["speak"]),
                extra_instructions=(
                    "Sin nombre, cargo ni empresa. Una frase corta. Tono tranquilizador."
                ),
            )
            await asyncio.sleep(0.35)
