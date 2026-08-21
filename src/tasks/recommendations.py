"""Recommendations / plan board phase."""

from __future__ import annotations

import json
import textwrap

from rpc_client import rpc
from tasks.base import VOICE_RULES, HuellaPhaseTask
from tasks.speech import generate_reply_safe
from tasks.ui_sync import present_and_speak


class RecommendationsTask(HuellaPhaseTask):
    active_steps = frozenset({"recommendations"})

    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=textwrap.dedent(
                f"""\
                {VOICE_RULES}

                You own the recommendations / plan board only.

                On enter the runtime focuses recommendation_item lines before you
                speak. Summarize content.plan first when asked, then one focused
                line at a time. Allow repeat/jump by intent. Guide CTAs from
                availableActions only.

                If the step is no longer recommendations, call return_to_supervisor.
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
        plan = content.get("plan") if isinstance(content, dict) else None
        plan = plan if isinstance(plan, dict) else {}
        mission = str(plan.get("missionTitle") or "tu plan")
        focus = str(plan.get("focusTitle") or "")
        items = plan.get("items") if isinstance(plan.get("items"), list) else []
        start_index = int(
            plan.get("activeIndex") or state.get("recommendationIndex") or 0
        )
        start_index = max(0, min(start_index, max(0, len(items) - 1)))

        await generate_reply_safe(
            self.session,
            instructions=(
                f"Resume brevemente el plan en pantalla: misión {mission}"
                f"{f' enfocando {focus}' if focus else ''}. "
                "Luego las líneas de acción se enfocan una a una. "
                "No llames herramientas en este turno."
            ),
            tool_choice="none",
        )

        # Present the active line (and up to two more) so the board UI moves with voice.
        for offset in range(min(3, max(1, len(items)))):
            index = (start_index + offset) % max(1, len(items) or 1)
            item = (
                items[index]
                if index < len(items) and isinstance(items[index], dict)
                else {}
            )
            title = str(item.get("title") or f"Acción {index + 1}")
            await present_and_speak(
                self.session,
                target="recommendation_item",
                index=index,
                fallback_speak=title,
                extra_instructions="Lee solo esta línea del plan visible en pantalla.",
            )
