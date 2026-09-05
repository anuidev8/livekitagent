"""Analysis scanning → complete → results."""

from __future__ import annotations

import asyncio
import json
import textwrap

from rpc_client import rpc
from tasks.base import VOICE_RULES, HuellaPhaseTask, PhaseResult
from tasks.speech import generate_reply_safe
from tasks.ui_sync import present_and_speak


class AnalysisTask(HuellaPhaseTask):
    active_steps = frozenset({"analysis"})

    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=textwrap.dedent(
                f"""\
                {VOICE_RULES}

                You own the analysis screen only.

                phase scanning — Narrate public-source analysis in progress. Never invent
                findings absent from state. Reassure briefly; wait for completion.

                phase complete — Paraphrase facts.uiStandingLine warmly (role,
                standing, strongest dimension). In the same flow add ONE strength
                from facts.strengths/coverLines and ONE gap from facts.opportunities
                or facts.weakestDimension — report data only, consultative C-level.
                State exact global score from content.overallScore. Offer
                navigate_journey reveal_results or dimension detail when ready.

                phase results — The runtime focuses each result_dimension before you
                speak. Narrate only the focused card: title, score, summary,
                Oportunidades from state/presentation. Map intent to availableActions
                (open_detail, advance, send_report, back). No cancel CTA.

                If the step is no longer analysis, call return_to_supervisor.
                """
            ),
        )
        self._results_tour_started = False

    async def on_enter(self) -> None:
        raw = await rpc("get_session_state")
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            state = {}

        phase = str(state.get("phase") or "scanning")
        facts = state.get("facts") if isinstance(state.get("facts"), dict) else {}
        has_report = facts.get("hasReport")
        if has_report is None:
            has_report = bool(facts.get("hasRealData")) if "hasRealData" in facts else None
        # Explicit false → no package; missing keys with empty content also treat as no report
        # when overallScore is 0 and dimensions list is empty.
        content = state.get("content") if isinstance(state.get("content"), dict) else {}
        dims = content.get("dimensions") if isinstance(content, dict) else None
        dim_list = dims if isinstance(dims, list) else []
        if has_report is None:
            score = content.get("overallScore") if isinstance(content, dict) else None
            has_report = bool(dim_list) or (isinstance(score, (int, float)) and score > 0)
        no_report = has_report is False

        if phase == "results":
            if no_report:
                await generate_reply_safe(
                    self.session,
                    instructions=(
                        "Habla con calidez en español, unas 2 frases cortas: "
                        "por ahora no hemos encontrado un informe de huella listo "
                        "para esta persona; cuando esté disponible, podrán "
                        "revisarlo juntos aquí. "
                        "PROHIBIDO inventar fuentes, scores o dimensiones. "
                        "PROHIBIDO invitar a continuar, detalle, reporte o otro análisis. "
                        "Cierra sin CTA de avance."
                    ),
                    tool_choice="none",
                )
                return
            await self._run_results_tour(state)
            if not await self._still_on_analysis():
                return
            await generate_reply_safe(
                self.session,
                instructions=(
                    "Ya recorriste las dimensiones reveladas. Guía detalle, otra "
                    "dimensión o envío según availableActions."
                ),
                tool_choice="auto",
                tools=[
                    "get_session_state",
                    "present_content",
                    "navigate_journey",
                ],
            )
            return

        if phase == "complete":
            if no_report:
                await generate_reply_safe(
                    self.session,
                    instructions=(
                        "Habla con calidez (~2 frases): por ahora no hemos "
                        "encontrado un informe de huella listo para ti; cuando "
                        "esté disponible, lo revisamos juntos aquí. "
                        "PROHIBIDO inventar puntuaciones, fuentes, fortalezas o brechas. "
                        "PROHIBIDO open_detail, send_report, reveal_results o invitar a continuar."
                    ),
                    tool_choice="none",
                )
                return
            await generate_reply_safe(
                self.session,
                instructions=(
                    "El análisis terminó. Paráfrasis cálida de facts.uiStandingLine "
                    "(rol, standing, dimensión más fuerte) + UNA fortaleza de "
                    "facts.strengths/coverLines y UNA brecha de facts.opportunities/"
                    "weakestDimension — solo datos del informe, tono consultor C-level. "
                    "Menciona el score global exacto (content.overallScore). "
                    "Invita a abrir detalle de dimensión o navigate_journey(reveal_results)."
                ),
                tool_choice="auto",
                tools=["get_session_state", "navigate_journey"],
            )
            return

        if no_report:
            await generate_reply_safe(
                self.session,
                instructions=(
                    "Habla con calidez (~2 frases): por ahora no hemos encontrado "
                    "un informe de huella listo para ti; cuando esté disponible, "
                    "lo revisamos juntos aquí. "
                    "PROHIBIDO inventar LinkedIn, prensa, redes, sitios, hallazgos "
                    "o puntuaciones. PROHIBIDO invitar a continuar. No llames herramientas."
                ),
                tool_choice="none",
            )
            return

        await generate_reply_safe(
            self.session,
            instructions=(
                "Narra el análisis en curso usando SOLO facts.sourceGroups / "
                "facts.narrationAnchors / facts.searchFindings del estado. "
                "Sin inventar hallazgos. Breve y tranquilizador. No llames herramientas."
            ),
            tool_choice="none",
        )

    async def _still_on_analysis(self) -> bool:
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

    async def _run_results_tour(self, state: dict) -> None:
        if self._results_tour_started:
            return
        self._results_tour_started = True

        content = state.get("content") if isinstance(state.get("content"), dict) else {}
        dims = content.get("dimensions") if isinstance(content, dict) else None
        dimensions = dims if isinstance(dims, list) else []
        count = min(5, len(dimensions) or 5)

        for i in range(count):
            if not await self._still_on_analysis():
                return
            dim = (
                dimensions[i]
                if i < len(dimensions) and isinstance(dimensions[i], dict)
                else {}
            )
            title = str(dim.get("title") or f"Dimensión {i + 1}")
            score = dim.get("score")
            summary = str(dim.get("summary") or "")
            opps = dim.get("opportunities")
            opp_text = ""
            if isinstance(opps, list) and opps:
                opp_text = " Oportunidades: " + "; ".join(str(x) for x in opps[:3])
            fallback = f"{title}. Score {score}. {summary}{opp_text}".strip()
            dimension_id = str(dim.get("id") or "")
            await present_and_speak(
                self.session,
                target="result_dimension",
                index=i,
                dimension_id=dimension_id,
                fallback_speak=fallback,
                extra_instructions=(
                    f"Narra solo la tarjeta de {title} visible ahora. "
                    "Usa score, resumen y oportunidades del estado."
                ),
            )
            await asyncio.sleep(0.35)
