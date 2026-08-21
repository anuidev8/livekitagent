"""Intro process steps + dimension explainers."""

from __future__ import annotations

import asyncio
import json
import textwrap

from rpc_client import rpc
from tasks.base import VOICE_RULES, HuellaPhaseTask, PhaseResult
from tasks.speech import generate_reply_safe
from tasks.ui_sync import present_and_speak


class IntroTask(HuellaPhaseTask):
    active_steps = frozenset({"intro"})

    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=textwrap.dedent(
                f"""\
                {VOICE_RULES}

                You own the intro screen ("Así funciona") only.

                The runtime drives the UI before you speak:
                1) intro_step 0→1→2 (three process cards)
                2) intro_dimension 0→4 (spider / radar + dimension copy)

                Speak only presentation / processSteps / dimensionConcepts text.
                Never invent a different process or dimension count.

                After the automatic walkthrough, allow the user to: repeat a card
                (present_content intro_step), open the spider again
                (present_content intro_dimension), jump, or ask questions by intent.

                CTAs — Guide "Comenzar análisis" / back / cancel from availableActions
                by intent. Only navigate_journey start_analysis when they clearly want
                to begin the analysis.

                If the step is no longer intro, call return_to_supervisor.
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

        phase = str(state.get("phase") or state.get("introPhase") or "steps")
        if phase == "explain" and self._tour_started:
            await generate_reply_safe(
                self.session,
                instructions=(
                    "El spider ya está visible. Si hace falta, present_content "
                    "intro_dimension para enfocar una dimensión y explica solo esa."
                ),
                tool_choice="auto",
                tools=["get_session_state", "present_content", "navigate_journey"],
            )
            return

        await self._run_intro_tour(state)

        if not await self._still_on_intro():
            return
        await generate_reply_safe(
            self.session,
            instructions=(
                "Ya recorriste las tres tarjetas y las cinco dimensiones en el radar. "
                "Ofrece repetir, volver a una dimensión, o empezar el análisis según "
                "availableActions (start_analysis)."
            ),
            tool_choice="auto",
            tools=["get_session_state", "present_content", "navigate_journey"],
        )

    async def _still_on_intro(self) -> bool:
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

    async def _run_intro_tour(self, state: dict) -> None:
        if self._tour_started:
            return
        self._tour_started = True

        content = state.get("content") if isinstance(state.get("content"), dict) else {}
        raw_steps = content.get("processSteps") if isinstance(content, dict) else None
        steps = raw_steps if isinstance(raw_steps, list) else []
        raw_concepts = (
            content.get("dimensionConcepts") if isinstance(content, dict) else None
        )
        concepts = raw_concepts if isinstance(raw_concepts, list) else []

        # 1) Three "Así funciona" process cards
        for i in range(3):
            if not await self._still_on_intro():
                return
            item = steps[i] if i < len(steps) and isinstance(steps[i], dict) else {}
            title = str(item.get("title") or f"Paso {i + 1}")
            narration = str(item.get("narration") or item.get("speak") or title)
            await present_and_speak(
                self.session,
                target="intro_step",
                index=i,
                fallback_speak=narration,
                extra_instructions=(
                    f"Lee solo esta tarjeta del proceso ({title}). "
                    "No inventes otros pasos."
                ),
            )
            await asyncio.sleep(0.4)

        # 2) Spider / radar — five dimensions (switches UI to explain mode)
        dim_count = max(len(concepts), 5)
        for i in range(min(5, dim_count)):
            if not await self._still_on_intro():
                return
            concept = (
                concepts[i]
                if i < len(concepts) and isinstance(concepts[i], dict)
                else {}
            )
            title = str(concept.get("title") or f"Dimensión {i + 1}")
            explanation = str(
                concept.get("explanation") or concept.get("copy") or title
            )
            dimension_id = str(concept.get("id") or "")
            await present_and_speak(
                self.session,
                target="intro_dimension",
                index=i,
                dimension_id=dimension_id,
                fallback_speak=explanation,
                extra_instructions=(
                    f"El radar ya enfoca {title}. Explica solo esa dimensión. "
                    "No listes las cinco de golpe."
                ),
            )
            await asyncio.sleep(0.35)
