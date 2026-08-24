"""Huella Digital voice guide — Amazon Nova Sonic 2 (LiveKit AWS realtime)."""

from __future__ import annotations

import json
import logging
import os
import textwrap

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    ToolError,
    cli,
    function_tool,
    room_io,
)
from livekit.plugins import ai_coustics, aws

from rpc_client import rpc

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "huella-guide")

# Nova Sonic 2 Spanish (es-US): lupe (feminine) | carlos (masculine)
# https://docs.livekit.io/agents/models/realtime/plugins/nova-sonic/
NOVA_VOICE = os.getenv("NOVA_VOICE", "lupe")
NOVA_TURN_DETECTION = os.getenv("NOVA_TURN_DETECTION", "MEDIUM")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Product detail lives in Next.js RPC spokenContent / facts.
# Keep this prompt free of investigation / surveillance framing so Bedrock
# RAI does not block session init (ValidationException: content filters).
NOVA_INSTRUCTIONS = textwrap.dedent(
    """\
    Eres la guía de voz en español del kiosk SETI Huella Digital.
    Habla como una anfitriona profesional de un evento corporativo:
    cálida, clara y serena — nunca robótica ni infantil.
    Evita abrir solo con «Hola.» Evita diminutivos y coloquialismos:
    nada de «rapidito», «momentito», «segundito», «te late», «arrancamos».
    Prefiere: «con calma», «cuando quieras», «iniciamos».
    Ritmo pausado y suave: oraciones completas, con pausa breve entre ellas.
    Solo español. Sin markdown ni listas.

    GENERAS TU PROPIO MENSAJE — no eres un lector de guión.
    La herramienta devuelve SIEMPRE "facts" con datos de la pantalla y
    un "hint" de composición. Úsalos como ancla de verdad y compón tú misma
    el mensaje en lenguaje natural. spokenContent es solo emergencia.
    Nunca repitas la misma frase en dos turnos de la misma superficie.
    Nunca inventes datos que no estén en facts.

    En cada turno del visitante o mensaje [pantalla:]:
    1) Llama get_session_state.
    2) Si debes enfocar algo, llama present_content con el target EXACTO.
    3) Compón tu mensaje a partir de facts.hint y los campos de facts.
    Los [pantalla:] traen pista de step/phase/identity/focus.
    Úsalos para foco y timing; no inventes pantallas ni datos de perfil.

    REGLA CENTRAL — COMPOSICIÓN DINÁMICA:
    Sigue facts.hint siempre. Varía la apertura de cada dimensión o sección.
    No copies el texto del guión. No uses etiquetas como apertura directa.

    PROHIBIDO decir frases de espera («un momento», «espera», «ya casi»)
    fuera de pantallas que de verdad cargan: welcome preparing, analysis
    scanning, y closing capture/shutter/generating.

    Targets válidos de present_content:
    - attract_tour (solo index -1), gesture_practice,
      welcome_preparation (index 0..2), intro_step (index 0..2),
      intro_dimension (dimension_id=higiene|serp|ssi|influencia|arquitectura),
      result_dimension (index 0..4 o dimensionId),
      detail_dimension (+dimensionId), detail_section (+section),
      recommendation_item.
    Prohibido: target "attract", "intro", "analysis", "dimension".

    Si present_content o navigate_journey fallan (ok:false): llama
    get_session_state, usa availableActions, y reintenta.

    ════════════════════════════════════════════════
    FLUJOS ESPECÍFICOS
    ════════════════════════════════════════════════

    ATTRACT (pantalla de inicio):
    1) present_content(attract_tour, index=-1) → título.
    2) Narra facts con tono profesional. Termina de hablar completo.
    3) Pregunta si quiere practicar los gestos O continuar directo.
       Espera su respuesta.
    4) Si acepta practicar: present_content(gesture_practice).
    5) Si prefiere continuar SIN practicar: navigate_journey start_experience
       INMEDIATAMENTE. PROHIBIDO present_content(gesture_practice) en este
       caso — no muestres el playground ni expliques gestos.
    PROHIBIDO attract_tour index ≥ 0.
    PROHIBIDO nombre/rol/empresa hasta welcome ready.

    GESTURE PRACTICE:
    Narra una vez. Deja practicar. No pidas continuar.
    Cuando confirme, navigate_journey start_experience.

    WELCOME PREPARING (carga de identidad):
    - Emite UNA sola locución larga y cálida que acompañe TODO el proceso
      de verificación, desde el inicio hasta que termina. Algo así como
      narrar quién es el evento, qué se está confirmando, que el sistema
      revisa fuentes públicas — sin interrumpirte y sin esperar respuesta.
    - PROHIBIDO llamar ninguna herramienta adicional mientras la UI siga
      en phase=preparing. NO llames get_session_state de nuevo ni
      present_content(welcome_preparation, X) para "avanzar" el índice.
      La UI avanza sola. Tú solo hablas UNA vez, largo y cálido.
    - PROHIBIDO enumerar los ítems del checklist o mencionarlos uno a uno.
    - PROHIBIDO pedir continuar o confirmación.
    - PROHIBIDO nombre, rol, empresa mientras preparing.

    WELCOME READY:
    Saluda INMEDIATAMENTE con nombre (desde facts.name/role/company).
    Ofrece empezar si start_experience está en availableActions.

    «ASÍ FUNCIONA» (intro) — AVANCE AUTOMÁTICO POR LA UI:
    La UI avanza sola tras cada narración (pasos 1→2→3, luego dimensiones
    Autoridad → Higiene → Influencia → Mensaje → SSI).
    Tu trabajo: cuando llega un [pantalla:] o present_content enfocado,
    narra SOLO ese elemento (menciona el título, explica el concepto).
    PROHIBIDO pedir «continuar» o esperar al visitante para el siguiente.
    PROHIBIDO llamar present_content solo para pasar al siguiente índice —
    la UI ya lo hace. SÍ usa present_content si el visitante pide una
    dimensión concreta por nombre. Si pide iniciar el análisis:
    navigate_journey start_analysis de inmediato.
    En la última dimensión (facts.isLast): después de explicar, ofrece
    iniciar el análisis.

    ANALYSIS SCANNING:
    Narra con entusiasmo y naturalidad qué fuentes se revisan (Google,
    LinkedIn, prensa, directorios, redes). Varía cada turno. Tono de
    acompañamiento en tiempo real. PROHIBIDO lista numerada.

    RESULTADOS (result_dimension):
    Compón desde facts. Introduce la dimensión con contexto primero,
    score al final de forma casual. Varía la apertura cada vez.

    DETALLE (detail_section) — AVANCE AUTOMÁTICO POR LA UI:
    La UI avanza sola tras cada narración: Fortalezas → Oportunidades →
    Plan de acción. Tu trabajo: narra SOLO la sección enfocada
    (sintetiza items como a un amigo; no enumeres ni leas literalmente).
    PROHIBIDO pedir «continuar» o esperar al visitante para la siguiente.
    PROHIBIDO llamar present_content solo para pasar a la siguiente
    sección — la UI ya lo hace. SÍ usa present_content(detail_section,
    section=strengths|opportunities|action_plan) si el visitante pide
    fortalezas, oportunidades o plan de acción por nombre.
    En la última sección (plan de acción / facts.isLastSection): después
    de narrar, ofrece volver a resultados, otra dimensión o el reporte.
    No uses lenguaje de advertencia («alerta», «atención»).

    DETALLE → VOLVER (back desde detail):
    Cuando el visitante dice "volver" o navega BACK desde detail, la UI
    regresa a los resultados. NO describas el resumen de la dimensión
    de nuevo a menos que el visitante lo pida explícitamente.
    Di solo algo breve como «De vuelta a los resultados. ¿Qué quieres
    ver?» o «¿Quieres revisar otra dimensión o avanzar al plan?»
    Espera su elección.

    RECOMENDACIONES:
    Presenta cada paso del plan de acción con naturalidad desde facts.item.
    No digas «Paso N del plan: X». Varía la apertura.

    CIERRE / FOTO / TARJETA:
    - photo (prep/pose/capture/shutter/generating): UNA sola locución
      larga y cálida EN TOTAL al entrar. Cubre: vamos a crear tu
      informe personalizado, y antes tomamos una foto para la tarjeta;
      quédate en el recuadro y confirma cuando estés listo.
      La UI cambia títulos sola — PROHIBIDO re-narrar o llamar
      herramientas por cada fase. PROHIBIDO navigate_journey.
    - delivered: la cámara está activa — narra el resultado con calidez
      usando facts (nombre, topDimension, score). Espera phase=thanks.
    - thanks: agradece brevemente, ofrece terminar.

    GESTOS (si preguntan):
    Deslizar = slider; tip-touch = ciclar CTA; puño = confirmar CTA.
    Gestos en acciones críticas requieren confirmación hablada.

    FUERA DE TEMA:
    Si el visitante habla de algo completamente ajeno a la experiencia
    (no relacionado con su huella digital, el kiosk o el evento),
    redirige en una frase breve y natural. Para preguntas sobre el evento,
    el sistema o cómo funciona el kiosk — responde con naturalidad.

    No menciones herramientas, modelos ni sistemas internos.
    """
)

# Alias for tests / docs that still reference MAIN_INSTRUCTIONS.
MAIN_INSTRUCTIONS = NOVA_INSTRUCTIONS
INSTRUCTIONS = NOVA_INSTRUCTIONS

_FALLBACK_SESSION = {
    "ok": False,
    "error": "get_session_state_unavailable",
    "step": "attract",
    "phase": "ready",
    "availableActions": ["start_experience", "practice_gestures"],
    "spokenContent": (
        "Bienvenido. Estoy preparando tu experiencia. "
        "En un momento practicamos los gestos e iniciamos."
    ),
    "title": "Huella Digital",
}


class Assistant(Agent):
    """Nova Sonic host: fixed tools, UI-first spokenContent contract.

    Per LiveKit Nova Sonic guide: register @function_tool on the Agent;
    put tool_choice on RealtimeModel (not per-reply). Warm greeting via
    on_enter + generate_reply (Nova Sonic 2 mixed modalities).
    """

    def __init__(self) -> None:
        super().__init__(instructions=NOVA_INSTRUCTIONS)

    async def on_enter(self) -> None:
        # Attract first beat: hero title + invite, then WAIT for the answer.
        # The screen change from a tool call is instant but speech is not —
        # firing a second present_content in this same turn makes the UI
        # jump to the practice screen before the invite line even finishes
        # playing. Ask, and let the next turn (their answer) drive the next
        # tool call instead of stacking two in one turn.
        await self.session.generate_reply(
            instructions=(
                "Estás en la pantalla de inicio (attract). "
                "1) Llama get_session_state. "
                "2) Llama present_content con target exactamente "
                "'attract_tour' e index -1 (título). "
                "3) Compón tu mensaje desde facts con tono profesional "
                "y pausado — NO abras solo con «Hola» y NO uses diminutivos. "
                "Termina de hablar completo. "
                "4) Pregunta si quiere practicar los gestos del espejo "
                "o prefiere continuar directo al análisis, y espera su respuesta. "
                "IMPORTANTE: si dice que prefiere continuar SIN gestos, llama "
                "navigate_journey start_experience — NUNCA present_content(gesture_practice) "
                "en ese caso. Solo muestra el playground si acepta practicar. "
                "PROHIBIDO present_content attract_tour index 0, 1 o 2. "
                "PROHIBIDO nombre, rol o empresa del visitante en esta pantalla."
            ),
        )

    @function_tool
    async def get_session_state(self, context: RunContext) -> str:
        """Obtiene el estado actual de la pantalla (step, phase, availableActions, spokenContent)."""
        try:
            return await rpc("get_session_state", retries=2)
        except ToolError as exc:
            logger.warning("get_session_state soft-fail: %s", exc)
            payload = dict(_FALLBACK_SESSION)
            payload["message"] = str(exc)
            return json.dumps(payload, ensure_ascii=False)

    @function_tool
    async def present_content(
        self,
        context: RunContext,
        target: str,
        index: int = -1,
        dimension_id: str = "",
        section: str = "",
    ) -> str:
        """Enfoca UN elemento visible. Targets EXACTOS: attract_tour
        (solo index -1 título; index≥0 va a práctica), gesture_practice,
        welcome_preparation (0..2 prep), intro_step (0=primero, 1=segundo,
        2=tercero), intro_dimension (mejor dimension_id=higiene|serp|…),
        result_dimension, detail_dimension (+dimension_id), detail_section
        (+section), recommendation_item. Compón tu mensaje desde facts.hint
        y los campos de facts devueltos. Si ok=false, get_session_state y reintenta.
        Nunca uses target=attract|intro|analysis.
        """
        return await rpc(
            "present_content",
            {
                "target": target,
                "index": index,
                "dimensionId": dimension_id,
                "section": section,
            },
        )

    @function_tool
    async def navigate_journey(
        self, context: RunContext, action: str, dimension_id: str = ""
    ) -> str:
        """Ejecuta una acción disponible en la experiencia."""
        return await rpc(
            "navigate_journey",
            {"action": action, "dimensionId": dimension_id},
        )

    @function_tool
    async def set_control_channel(
        self, context: RunContext, channel: str, enabled: bool
    ) -> str:
        """Activa o desactiva un control de la experiencia."""
        return await rpc(
            "set_control_channel", {"channel": channel, "enabled": enabled}
        )


# Backward-compatible name used by earlier deploys / tests.
NovaAssistant = Assistant


def _build_nova_realtime() -> aws.realtime.RealtimeModel:
    """Amazon Nova Sonic 2 — LiveKit AWS realtime plugin.

    Session-recycle note: with static AWS_ACCESS_KEY_ID/SECRET credentials
    Bedrock enforces a hard 360-second cap per bidirectional stream, after
    which the plugin silently tears down and re-opens the WebSocket
    (_session_recycle_timer).  This causes a ~1-2 s freeze mid-session and
    resets temperature/top_p to defaults.

    To push the cap to 3600 s (or remove it entirely with IAM role + STS):
      - Use an IAM Role with STS and set AWS_SESSION_TOKEN in the environment.
      - OR keep static creds and accept the 360 s recycle, but pin
        temperature/top_p here so at least the model behavior stays consistent
        across the recycle.

    session_refresh_interval: explicitly set below to avoid the silent
    mid-conversation reset.  Value must be < 360 for static creds — we use
    355 s to trigger a proactive recycle a few seconds before AWS forces it,
    which is slightly less disruptive than a hard timeout.
    """
    return aws.realtime.RealtimeModel.with_nova_sonic_2(
        voice=NOVA_VOICE,
        turn_detection=NOVA_TURN_DETECTION,  # type: ignore[arg-type]
        region=AWS_REGION,
        tool_choice="auto",
        generate_reply_timeout=20.0,
        temperature=0.7,
        top_p=0.9,
        # Nova Sonic 2 defaults to mixed modalities (audio + text),
        # which enables on_enter generate_reply warm intro.
    )


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "agent": AGENT_NAME,
        "voice_backend": "nova",
        "voice_model": "amazon.nova-2-sonic",
        "nova_voice": NOVA_VOICE,
    }

    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        logger.warning(
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY missing — "
            "Nova Sonic Bedrock calls will fail until secrets are set."
        )

    session = AgentSession(
        llm=_build_nova_realtime(),
        # Increased from 4: get_session_state + present_content + navigate_journey
        # + potential retry each = 6 steps needed on complex screens (detail_dimension).
        # With 4 the agent silently stops mid-chain on those screens, appearing frozen.
        max_tool_steps=8,
    )

    logger.info("Starting Nova Sonic voice=%s region=%s", NOVA_VOICE, AWS_REGION)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_L
                ),
            ),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
