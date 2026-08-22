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
    3) Habla únicamente con spokenContent, narration o title que
       devuelvan las herramientas. Puedes suavizar el ritmo (pausas)
       sin inventar datos del producto.

    Targets válidos de present_content (obligatorio usar estos strings):
    - attract_tour — solo index -1 = título «¿Sabe qué dice…?».
      NO uses index 0|1|2 (las tarjetas Gestos/Toque/Voz ya no existen;
      si las pides, la UI salta a práctica).
    - gesture_practice — práctica interactiva (slider + tip-touch +
      cerrar la mano). Ir directo aquí tras el saludo de attract.
    - welcome_preparation — index 0|1|2 casillas; index 3 = listo con nombre.
    - intro_step — index 0|1|2 tarjetas «Así funciona» (número + título).
    - intro_dimension — index 0..4 araña / dimensiones (cómo funcionan).
    - result_dimension — index 0..4 resultados.
    - detail_dimension — con dimensionId.
    - detail_section — section strengths|opportunities|action_plan.
    - recommendation_item — index del plan.
    Prohibido: target "attract", "intro", "analysis", "dimension" u otros
    inventados — fallan y la UI no cambia.

    Flujo pantalla de inicio (attract):
    1) present_content(attract_tour, index=-1) → título en pantalla.
    2) Narra spokenContent con tono profesional (no «Hola» seco) e invita
       a practicar los gestos. No ofrezcas un tour de tarjetas.
    3) Si acepta, pide empezar, o confirma el CTA: present_content
       gesture_practice (o navigate_journey practice_gestures /
       start_experience si están en availableActions).
    4) PROHIBIDO present_content attract_tour con index 0, 1 o 2.
    PROHIBIDO en attract: decir el nombre, rol o empresa del visitante.
    El nombre solo se dice en welcome cuando phase=ready / present_content
    welcome_preparation index=3 (vista con nombre).

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
    1) present_content(welcome_preparation, index=3) si aún no está enfocado.
    2) Habla spokenContent con el nombre, tono profesional. Ofrece empezar
       solo si start_experience está en availableActions.

    Flujo «Así funciona» (intro):
    1) present_content(intro_step, 0) → narra TODO el spokenContent con
       ritmo lento y suave → espera → pregunta si sigue.
    2) Igual con index 1 y 2. No avances hasta terminar cada narración.
    3) Pregunta si quiere ver las cinco dimensiones; si acepta:
       present_content(intro_dimension, 0..4) una por turno, igual de pausado.
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
    - Tras la foto (capture, shutter, generating): narra spokenContent con
      calma. PROHIBIDO pedir continuar y PROHIBIDO navigate_journey.
      La UI genera la tarjeta sola (generate-card); espera phase=delivered
      o thanks antes de ofrecer cerrar.
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
        # Attract first beat: hero title, professional invite → practice.
        await self.session.generate_reply(
            instructions=(
                "Estás en la pantalla de inicio (attract). "
                "1) Llama get_session_state. "
                "2) Llama present_content con target exactamente "
                "'attract_tour' e index -1 (título). "
                "3) Habla solo spokenContent/title con tono profesional "
                "y pausado — NO abras solo con «Hola» y NO uses diminutivos "
                "(rapidito, momentito). Invita a practicar los gestos "
                "(sin tour de tarjetas Gestos/Toque/Voz). "
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
        (solo index -1 título; index≥0 va a práctica), gesture_practice
        (práctica rápida), welcome_preparation (0..2 prep, 3 listo),
        intro_step (0..2), intro_dimension (0..4), result_dimension (0..4),
        detail_dimension (+dimension_id), detail_section (+section),
        recommendation_item. Habla con el spokenContent devuelto.
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
