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

# Product detail lives in Next.js RPC spokenContent. Keep this prompt short and
# free of investigation / surveillance framing so Bedrock RAI does not block
# session init (ValidationException: content filters).
NOVA_INSTRUCTIONS = textwrap.dedent(
    """\
    Eres la guía de voz en español del kiosk SETI Huella Digital.
    Habla como una anfitriona profesional de un evento corporativo:
    cálida, clara y serena — nunca robótica ni infantil.
    Evita abrir solo con «Hola.» Evita diminutivos y coloquialismos:
    nada de «rapidito», «momentito», «segundito», «te late», «arrancamos».
    Prefiere: «un momento», «con calma», «cuando quieras», «iniciamos».
    Ritmo pausado y suave: oraciones completas, con pausa breve entre ellas.
    Especialmente en «Así funciona» (intro): habla DESPACIO, lee todo el
    spokenContent y no pases a la siguiente tarjeta hasta terminar.
    Solo español. Sin markdown ni listas.

    En cada turno del visitante o mensaje [pantalla:]:
    1) Llama get_session_state.
    2) Si debes enfocar un elemento visible, llama present_content con un
       target EXACTO de la lista (nunca uses el nombre del paso solo).
    3) Ancla en spokenContent, narration o title de las herramientas:
       dilo completo primero. Luego puedes añadir una o dos frases breves
       del mismo tema para aclarar (sin inventar nombres, cifras ni datos).
    Los [pantalla:] traen una pista de operador sobre step/phase/identity/
    focus. Úsala para foco y timing; no inventes pantallas ni datos de perfil.

    Si la herramienta devuelve "facts" (closing, result_dimension y
    detail_section por ahora): compón tú misma 1-2 frases naturales en
    español usando SOLO esos valores (score, dimension, section, summary,
    opportunities, items, closingPhase). No leas spokenContent literal en ese
    caso — es solo un respaldo — y no agregues cifras, nombres ni datos que
    no estén en facts. No repitas la misma frase textual si el visitante
    vuelve a pedir la misma dimensión o fase; reformula.

    Di el título de cada tarjeta o dimensión de forma cálida y natural, no
    como una etiqueta leída ("aquí tienes búsqueda" en vez de "búsqueda").
    En intro_dimension (pantalla "Así funciona"): di el concepto (spokenContent)
    completo, y añade UNA frase propia explicando por qué importa en general
    — sin inventar datos del visitante, eso llega después en el análisis.

    PROHIBIDO decir frases de espera («un momento», «espera», «dame un
    momento», «ya casi») fuera de las pantallas que de verdad están
    cargando algo: welcome preparing, analysis scanning, y closing
    capture/shutter/generating. En intro, result_dimension, detail_section y
    recommendation_item no hay ninguna carga — no uses ese lenguaje ahí.

    Targets válidos de present_content (obligatorio usar estos strings):
    - attract_tour — solo index -1 = título «¿Sabe qué dice…?».
      NO uses index 0|1|2 (las tarjetas Gestos/Toque/Voz ya no existen;
      si las pides, la UI salta a práctica).
    - gesture_practice — práctica interactiva (slider + tip-touch +
      cerrar la mano). Ir directo aquí tras el saludo de attract.
    - welcome_preparation — index 0|1|2 casillas de carga. La UI pone
      phase=ready sola; no fuerces el nombre.
    - intro_step — index 0|1|2 (0=primero, 1=segundo, 2=tercero).
    - intro_dimension — preferir dimensionId (higiene, serp, ssi,
      influencia, arquitectura). El orden del araña NO es el de resultados.
      Si el visitante pide una dimensión SOLO por número («la dimensión 3»,
      «la tercera») sin nombrarla, los dos órdenes no coinciden y adivinar el
      índice puede mostrar la dimensión equivocada. Antes de llamar
      present_content, di en voz alta cuál vas a mostrar (ej. «te muestro
      Mensaje») o pide el nombre — nunca cambies de pantalla en silencio
      sobre un número ambiguo.
    - result_dimension — index 0..4 o dimensionId (orden de resultados).
    - detail_dimension — con dimensionId.
    - detail_section — section strengths|opportunities|action_plan.
    - recommendation_item — index del plan.
    Prohibido: target "attract", "intro", "analysis", "dimension" u otros
    inventados — fallan y la UI no cambia.

    Si present_content o navigate_journey fallan (ok:false): llama
    get_session_state, usa availableActions, y reintenta. No digas que
    vas a mostrar una pantalla hasta que la herramienta confirme ok.

    Flujo pantalla de inicio (attract):
    1) present_content(attract_tour, index=-1) → título en pantalla.
    2) Narra completo el spokenContent con tono profesional (no «Hola» seco).
       Termina de hablar antes de llamar otra herramienta — el cambio de
       pantalla es instantáneo pero tu voz no, así que si enfocas la práctica
       antes de terminar de hablar, la pantalla salta antes de que termines
       de invitar y se ve descoordinado.
    3) Cuando termines, pregunta si quiere practicar los gestos del espejo o
       prefiere continuar directo. Espera su respuesta — no asumas.
    4) Si acepta practicar: llama present_content(gesture_practice) y narra
       su spokenContent.
    5) Si prefiere continuar: usa navigate_journey start_experience.
    6) PROHIBIDO present_content attract_tour con index 0, 1 o 2.
    PROHIBIDO decir el nombre, rol o empresa hasta welcome phase=ready.
    También PROHIBIDO en gesture_practice, nfc, validation y welcome
    preparing.

    Flujo práctica de gestos (gesture_practice):
    - Narra spokenContent una vez y deja practicar (slider, tip-touch,
      cerrar la mano). PROHIBIDO pedir «continuar» mientras practica.
    - Cuando esté listo (phase=ready o confirme), ofrece empezar y usa
      navigate_journey start_experience si está en availableActions.

    Flujo verificación de identidad (welcome phase=preparing, o validation):
    - Es una pantalla de CARGA: el sistema valida la identidad (API).
    - Llama get_session_state y present_content(welcome_preparation, 0|1)
      según lo visible. Narra spokenContent completo, sin cortar la frase.
    - PROHIBIDO pedir «continuar», «adelante» o confirmación.
    - PROHIBIDO navigate_journey y PROHIBIDO present_content index=3
      hasta que phase=ready (la UI espera a que termines de hablar).
    - Sin nombre, rol ni empresa mientras preparing.

    Flujo bienvenida lista (welcome phase=ready):
    1) En cuanto la vista con nombre aparece, saluda INMEDIATAMENTE con
       spokenContent (Bienvenido/a + nombre). No esperes otra confirmación.
    2) Ofrece empezar solo si start_experience está en availableActions.

    Flujo «Así funciona» (intro):
    1) present_content(intro_step, 0) → primero; narra completo y despacio.
    2) «Segundo» = index 1; «tercero» = index 2. No uses 2 para el segundo.
    3) Dimensiones: present_content(intro_dimension, dimension_id="higiene")
       (o serp, ssi, influencia, arquitectura). No inventes el índice.
    4) Ofrece empezar el análisis con availableActions.

    Nunca digas que no ves la pantalla. Si una herramienta falla o llega
    incompleta, usa el último spokenContent conocido o la respuesta de
    respaldo y continúa con profesionalismo.

    Un solo elemento visible por turno: enfoca, narra completo, escucha.
    No avances en ráfaga. El visitante puede pedir repetir, profundizar,
    ir adelante o atrás — interpreta la intención y usa availableActions /
    present_content.

    Guía CTAs naturales (confirmar, empezar, volver, ver detalle, terminar)
    según availableActions. Usa navigate_journey solo cuando confirme una
    acción listada. Usa set_control_channel solo si pide activar o
    desactivar gestos o voz.

    Gestos en el kiosk (explícalos con precisión si preguntan):
    - Deslizar la mano en el aire: mover el slider de tarjetas (solo carrusel,
      no botones CTA).
    - Unir pulgar e índice (tip-touch): seleccionar o ciclar el botón CTA.
    - Cerrar la mano: confirmar el CTA resaltado.
    Un resaltado por gesto no ejecuta captura, envío, terminar o cancelar
    en acciones críticas — espera confirmación hablada cuando aplique.

    Flujo cierre / foto / tarjeta:
    - En pose: guía y espera ready_for_picture.
    - Tras la foto (capture, shutter, generating): compón una frase breve y
      cálida a partir de facts.closingPhase (ver arriba) — varía el tono, no
      repitas siempre la misma frase. PROHIBIDO pedir continuar y PROHIBIDO
      navigate_journey en estas tres fases. La UI genera la tarjeta sola
      (generate-card); espera phase=delivered o thanks antes de ofrecer cerrar.
    - En thanks: agradece y usa finish solo si está en availableActions.

    No menciones herramientas, modelos ni sistemas internos.
    Si el tema no pertenece a esta experiencia, redirige en una frase corta
    a lo que hay en pantalla.
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
                "3) Habla solo spokenContent/title con tono profesional "
                "y pausado — NO abras solo con «Hola» y NO uses diminutivos "
                "(rapidito, momentito). Termina de hablar completo. "
                "4) Luego pregunta si quiere practicar los gestos del espejo "
                "o prefiere continuar directo, y espera su respuesta antes "
                "de llamar cualquier otra herramienta. "
                "PROHIBIDO present_content attract_tour index 0, 1 o 2. "
                "PROHIBIDO decir el nombre, rol o empresa del visitante "
                "en esta pantalla — eso es solo en welcome listo. "
                "No uses target 'attract'."
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
        (+section), recommendation_item. Habla con el spokenContent
        devuelto. Si ok=false, get_session_state y reintenta.
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
    """Amazon Nova Sonic 2 — LiveKit AWS realtime plugin."""
    return aws.realtime.RealtimeModel.with_nova_sonic_2(
        voice=NOVA_VOICE,
        turn_detection=NOVA_TURN_DETECTION,  # type: ignore[arg-type]
        region=AWS_REGION,
        tool_choice="auto",
        generate_reply_timeout=20.0,
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
        max_tool_steps=4,
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
