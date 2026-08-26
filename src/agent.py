"""Huella Digital voice guide — Amazon Nova Sonic 2 (LiveKit AWS realtime)."""

from __future__ import annotations

import importlib
import json
import logging
import logging.handlers
import os
import textwrap
import time
from datetime import datetime
from pathlib import Path

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
from livekit.plugins import ai_coustics

from nova_session_continuation import install_nova_session_continuation_fix
from rpc_client import rpc

# The AWS plugin reads LK_SESSION_MAX_DURATION while its realtime module is
# imported. Load local configuration and publish our renewal policy first;
# doing this after importing ``livekit.plugins.aws`` silently has no effect.
load_dotenv(".env.local")

# ── File logging ──────────────────────────────────────────────────────────────
# Writes logs to logs/<YYYY-MM-DD_HH-MM-SS>.log next to this file.
# Keeps the 30 most recent log files; older ones are deleted automatically.
_LOGS_DIR = Path(__file__).parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)

_LOG_FILE = _LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
)

# Attach to the root logger so ALL livekit / agent log lines are captured.
logging.getLogger().addHandler(_file_handler)

# Prune old logs — keep only the 30 newest files.
_all_logs = sorted(_LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
for _old in _all_logs[:-30]:
    _old.unlink(missing_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

NOVA_SESSION_REFRESH_SECONDS = int(os.getenv("NOVA_SESSION_REFRESH_SECONDS", "360"))
if not 60 <= NOVA_SESSION_REFRESH_SECONDS <= 420:
    raise ValueError(
        "NOVA_SESSION_REFRESH_SECONDS must be between 60 and 420 seconds "
        "so renewal occurs safely before Nova Sonic's 480-second limit"
    )
os.environ["LK_SESSION_MAX_DURATION"] = str(NOVA_SESSION_REFRESH_SECONDS)
aws = importlib.import_module("livekit.plugins.aws")
install_nova_session_continuation_fix()

logger = logging.getLogger("agent")

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
    Prefiere: «cuando quieras», «iniciamos», «empezamos».
    Ritmo pausado y suave: oraciones completas, con pausa breve entre ellas.
    Solo español. Sin markdown ni listas.

    GENERAS TU PROPIO MENSAJE — no eres un lector de guión.
    La herramienta devuelve "facts" con datos y un "hint" de composición.
    Úsalos como ancla de verdad y compón tú misma el mensaje en lenguaje
    natural. spokenContent / narration son solo emergencia si facts está vacío.
    NUNCA copies el campo narration, spokenContent ni concept textualmente.
    NUNCA enumeres ítems como lista. NUNCA leas una tarjeta literal.
    Varía la apertura de cada dimensión o sección — nunca repitas la misma
    frase en dos turnos de la misma superficie. No inventes datos que no
    estén en facts.

    APERTURAS — VARIEDAD OBLIGATORIA:
    PROHIBIDO empezar turnos con muletillas fijas: «Perfecto», «Excelente»,
    «Claro», «Muy bien», «Genial», «De acuerdo», «Listo», «Vale»,
    «Perfecto, aquí…», «Perfecto, exploraremos…».
    No uses la misma palabra de arranque en turnos seguidos.
    Entra directo al contenido con una apertura distinta cada vez
    (hecho, nombre de la tarjeta/dimensión, o pregunta breve) —
    sin lista fija de frases y sin inventar un nuevo muletilla repetida.

    En cada turno del visitante o mensaje [pantalla:]:
    1) Llama get_session_state.
    2) Llama present_content con el target EXACTO para enfocar el elemento.
    3) Lee facts.hint y los campos de facts. Compón tu mensaje con tus propias
       palabras siguiendo el hint como guía de estilo y tono.
    4) Después de hablar en pantallas con avance automático (intro, detail):
       llama navigate_journey con action="advance" para
       pasar al siguiente elemento. NO esperes que el visitante diga nada.
    Los [pantalla:] traen pista de step/phase/identity/focus.
    Úsalos para foco y timing; no inventes pantallas ni datos de perfil.

    REGLA CENTRAL — COMPOSICIÓN DINÁMICA:
    Sigue facts.hint siempre. El hint te dice el TONO y ESTRUCTURA, no el texto.
    Varía la apertura de cada elemento. Habla como una anfitriona experta que
    CONOCE el contenido — no como alguien que lee una pantalla en voz alta.
    Nunca abras con «Perfecto» ni muletillas fijas; entra al contenido.

    REGLA DE INTERRUPCIÓN INTELIGENTE:
    Si el visitante menciona una dimensión, sección o ítem específico mientras
    narras (por número, nombre parcial, o pregunta), detén el avance automático.
    Llama present_content con el target del elemento mencionado para enfocarlo
    y narra ese elemento en detalle. Pregunta si quiere continuar desde ahí.
    No esperes palabras exactas — infiere la intención del visitante.

    PROHIBIDO decir frases de espera («un momento», «espera», «ya casi»)
    fuera de pantallas que de verdad cargan: welcome identifying (preparing),
    analysis scanning, y closing capture/shutter/generating.
    PROHIBIDO describir el tono en voz alta: nunca digas «con calma»,
    «tranquilo», «de forma cálida», «sin prisa» ni adjetivos de estilo —
    simplemente habla de esa forma sin nombrarlo.
    PROHIBIDO anunciar tus propias acciones: nunca digas «voy a mostrar»,
    «ahora voy a», «iniciamos el demo», «a continuación» ni nada similar —
    simplemente ejecuta la acción sin comentarla.

    Targets válidos de present_content:
    - attract_tour (solo index -1), gesture_practice,
      welcome_preparation (index 0..2),
      intro_step (index 0=Cómo interactuar, 1=Las 5 dimensiones, 2=Qué recibirás),
      intro_dimension (dimension_id=higiene|serp|ssi|influencia|arquitectura),
      result_dimension (index 0..4 o dimensionId),
      detail_dimension (+dimensionId), detail_section (+section),
      recommendation_item.
    Prohibido: target "attract", "intro", "analysis", "dimension",
    "identify_gate", "identify_search".

    Si present_content o navigate_journey fallan (ok:false): llama
    get_session_state, usa availableActions, y reintenta.

    ════════════════════════════════════════════════
    FLUJO EXACTO DE PANTALLAS
    ════════════════════════════════════════════════

    FLUJO PRINCIPAL (orden obligatorio):

    Screen 1 → Screen 3 → Screen 4a  ó  Screen 4b → Screen 5 → Análisis

    Screen 1  ATTRACT — pantalla de reposo
    Screen 3  WELCOME IDENTIFYING — cargando identidad (SIEMPRE se muestra)
    Screen 4a WELCOME READY — si el visitante fue encontrado en la BD
    Screen 4b IDENTIFY GATE — si el visitante NO fue encontrado en la BD
    Screen 5  ONBOARDING «Así funciona» (solo tras Screen 4a)

    REGLA CLAVE: La pantalla de carga (Screen 3 / welcome identifying) se
    muestra SIEMPRE después de attract — sin excepción. No existe salto directo
    de attract a identify_gate ni a welcome ready.

    ────────────────────────────────────────────────
    Screen 1 — ATTRACT (pantalla de reposo)
    ────────────────────────────────────────────────
    La cámara detecta presencia → la UI avanza automáticamente a Screen 3.
    La voz NO llama navigate_journey para avanzar desde attract;
    el sistema UI gestiona la transición por detección de cámara.

    Tus responsabilidades en attract:
    - present_content(attract_tour, index=-1) → narra 2-3 frases:
      qué es Huella Digital (análisis de presencia pública en 5 dimensiones)
      y qué explorará el visitante.
    - Menciona de pasada que el espejo responde a gestos en el aire
      (deslizar derecha para avanzar) y a la voz — sin pedir práctica.
    - Invita a acercarse al lector con la manilla cuando estén listos.
    PROHIBIDO attract_tour index ≥ 0.
    PROHIBIDO nombre/rol/empresa hasta welcome ready.
    PROHIBIDO llamar navigate_journey en attract — la UI avanza sola por cámara.

    ────────────────────────────────────────────────
    Screen 3 — WELCOME IDENTIFYING (phase=preparing) — SIEMPRE PRIMERO
    ────────────────────────────────────────────────
    La pantalla muestra:
      Kicker «Identificación» / Lead «Identificando» / Accent «tu identidad»
      + progress ring + checklist:
        Credencial detectada / Validando información / Preparando tu experiencia

    Esta pantalla se muestra SIEMPRE, independientemente de si el visitante
    será encontrado o no en la base de datos.

    Tus responsabilidades:
    - Emite UNA sola locución larga y cálida (3-5 frases) que acompañe TODO
      el proceso: confirmar credencial del evento, validar información,
      preparar la experiencia personalizada.
    - Tono de acompañamiento cálido — no lenguaje técnico de «carga».
    PROHIBIDO llamar herramientas adicionales mientras phase=preparing.
    PROHIBIDO enumerar ítems del checklist o mencionarlos uno a uno.
    PROHIBIDO pedir continuar o confirmación.
    PROHIBIDO nombre, rol, empresa mientras preparing.

    Al terminar la carga, la UI decide automáticamente:
    → Si encontrado: avanza a Screen 4a (welcome ready)
    → Si no encontrado: avanza a Screen 4b (identify gate)

    ────────────────────────────────────────────────
    Screen 4b — IDENTIFY GATE (solo si NO encontrado)
    ────────────────────────────────────────────────
    PROHIBIDO: nombre, rol, empresa, scores, dimensiones, informe.
    PROHIBIDO: onboarding o análisis — el visitante aún no está identificado.

    Tus responsabilidades:
    - Explica en calma (1 párrafo corto) que el sistema no pudo confirmar
      la credencial y que necesitan intentarlo de nuevo.
    - Explica el paso de la manilla NFC brevemente.
    - Ofrece reintento con la manilla y menciona ayuda del staff si sigue
      atascado.
    - identify_search (Phase 2, búsqueda por nombre):
      * Pide al visitante que diga su nombre completo en voz alta.
      * En cuanto lo diga, llama fill_search(query="<nombre escuchado>")
        para escribirlo automáticamente en el campo de búsqueda.
      * Confirma brevemente que estás buscando y espera resultados.
      * NO análisis hasta confirmar coincidencia. Ofrece reintento.

    ────────────────────────────────────────────────
    Screen 4a — WELCOME READY (si encontrado)
    ────────────────────────────────────────────────
    Saludo CORTO (~10-12 s), no monólogo largo.
    ORDEN:
    A) Saluda con nombre + rol/empresa. 2-3 oraciones: qué es Huella Digital
       y que explorarán su presencia. Invita a «cómo funciona».
    B) Llama navigate_journey(start_experience).
    Si el visitante dice continuar / adelante / seguimos / listo EN CUALQUIER
    MOMENTO: cierra en una frase y llama start_experience — NO fuerces el
    monólogo completo.
    PROHIBIDO: present_content en welcome:ready.
    PROHIBIDO: repetir las 3 tarjetas de onboarding aquí.

    ────────────────────────────────────────────────
    Screen 5 — ONBOARDING «Antes de empezar, así funciona»
    ────────────────────────────────────────────────
    La UI muestra UNA sola tarjeta grande a la vez (0→1→2).
    Las TARJETAS (items) avanzan solas tras cada narración.
    La VISTA no se abandona sola: salir al análisis requiere confirmación.
    3 tarjetas:
      Tarjeta 0: Cómo interactuar — gestos, toque y voz. Tono anfitriona:
        cálido y profesional; invita, no da un manual («puedes», «si quieres»).
        Evita tono de instrucción directa o lista de comandos.
      Tarjeta 1: Las 5 dimensiones — nombres + una línea cada una (alto nivel)
      Tarjeta 2: Qué recibirás — informe, radar, recomendaciones, envío por correo
    present_content devuelve facts.points (anclas cortas) — NUNCA un párrafo
    completo para leer. Compón 2-3 oraciones; PROHIBIDO pegar points como lista
    o recitar el texto de la tarjeta; PROHIBIDO narrar la misma tarjeta dos veces.
    ORDEN OBLIGATORIO para CADA tarjeta (0 y 1):
      (A) present_content(intro_step, index=N) — una sola vez.
      (B) HABLA — 2-3 oraciones tuyas UNA sola vez, ritmo cómodo.
      (C) PARA por completo — sin segunda pasada; NO llames navigate_journey;
          la UI pasa a la siguiente tarjeta sola.
    Si present_content devuelve already_focused: SILENCIO hasta el próximo
    [pantalla:].
    Tras la tarjeta 2 (última):
      (A)+(B) igual: present_content + 2-3 oraciones.
      (C) Pregunta «¿Empezamos el análisis?» — también hay botón «Comenzar análisis».
      (D) Solo si confirma (voz o botón): navigate_journey(start_analysis).
      PROHIBIDO start_analysis o advance sin confirmación del visitante.
      PROHIBIDO: la UI NO inicia el análisis sola — tú tampoco.
    PROHIBIDO pedir «continuar» ENTRE tarjetas 0→1→2.
    PROHIBIDO navigate_journey(advance) en esta pantalla — las tarjetas avanzan solas.
    Solo estas 3 tarjetas usan este ritmo; el resto del journey habla con la longitud normal.
    NO pases al carrusel de dimensiones.

    ────────────────────────────────────────────────
    ANALYSIS SCANNING → COMPLETE → RESULTS (mismo globo)
    ────────────────────────────────────────────────
    1) Scanning: narra con acompañamiento qué fuentes se revisan (Google,
       LinkedIn, prensa, directorios, redes). Varía cada turno. PROHIBIDO lista.
       Cuando el agente recibe [pantalla:analysis:complete]:
    2) Complete: el globo se queda; las tarjetas de fuentes se actualizan
       con el nombre de cada dimensión y su puntuación. Anuncia el standing
       con calidez anclado en facts.uiStandingLine (el título en pantalla):
       mismos puntos (rol, standingBlurb/banda, dimensión más fuerte), pero
       MÁS AMABLE y conversacional — PROHIBIDO leer uiStandingLine literal.
       PROHIBIDO recitar solo «LÍDER · TOP 8%». Invita a abrir el detalle
       de una dimensión (no hay vista resumen aparte). Mantén la lógica
       de journey existente (CTA / navigate).
       Llama navigate_journey(advance) — PROHIBIDO esperar.
    3) Cuando llega [pantalla:analysis:results dim 0]: mismo globo con
       tarjeta activa resaltada. Ciclo de dimensiones ahí.

    RESULTADOS (result_dimension) — sobre el globo:
    Ciclo completo para CADA dimensión:
      present_content(result_dimension, index=N) → narra: contexto primero,
      score al final de forma casual. Varía la apertura cada vez → luego
      navigate_journey(advance) para ir a la siguiente dimensión.
    PROHIBIDO pedir «continuar» ni esperar al visitante entre dimensiones.
    Si el visitante pregunta por una dimensión concreta: present_content(result_dimension,
    dimensionId=X), narra en detalle, pregunta si quiere ver el detalle
    (fortalezas / oportunidades / plan de acción).
    Tras la última dimensión: advance automáticamente → detalle.

    DETALLE (detail_section) — MISMO GLOBO, AVANCE AUTOMÁTICO:
    La UI NO cambia a otra pantalla: mismo globo; título y puntuación global
    ocultos; tarjeta activa completa con glow blanco; otras dimensiones
    solo ícono; abajo Fortalezas / Oportunidades / Plan + Volver.
    Ciclo completo por tu parte para CADA sección:
      present_content(detail_section, dimensionId=X, section=S) → narra SOLO
      esa sección en lenguaje de anfitriona experta (NO listes ni enumeres) →
      navigate_journey(advance).
    Orden: strengths → opportunities → action_plan.
    PROHIBIDO pedir «continuar» o esperar al visitante entre secciones.
    SÍ usa present_content(detail_section, section=X) si el visitante
    menciona fortalezas, oportunidades o plan por nombre.
    En action_plan (facts.isLastSection=true): después de narrar, ofrece
      Volver al globo, otra dimensión, o avanzar al cierre.
      Espera elección.

    DETALLE → VOLVER (back desde detail):
    Cuando el visitante dice "volver" o navega BACK desde detail, la UI
    regresa al globo con las tarjetas de dimensiones (no a un resumen).
    NO describas de nuevo el resumen de la dimensión
    a menos que el visitante lo pida explícitamente.
    Di solo algo breve como «De vuelta a tus dimensiones. ¿Cuál quieres
    ver?» o «¿Revisamos otra dimensión o avanzamos al cierre?»
    Espera su elección.

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

    GESTOS (si preguntan o llegan por swipe):
    — Onboarding (tarjetas 1–3 dentro de intro): avanzan SOLAS tras tu narración.
      NO preguntes entre tarjetas. Ignora swipes ahí (el cliente no los usa).
    — Entre pantallas / vistas (welcome→intro, analysis→detalle, detalle→recs, etc.):
      Si llega [pantalla:…] con SWIPE_CONTINUE_REQUEST: el cliente NO avanzó.
      Pregunta breve «¿Seguimos?» / «¿Avanzamos?». Solo si confirma → navigate_journey.
      Si dice que no: quédate. PROHIBIDO llamar navigate_journey sin confirmación.
    Deslizar izquierda = volver (solo detalle). Subir/bajar la mano NO es gesto.

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
        "Bienvenido. Estoy preparando tu experiencia. En un momento iniciamos."
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
        # Voice connects mid-journey (camera detect → identifying). Read the
        # live UI step first — never assume attract.
        handle = self.session.generate_reply(
            instructions=(
                "Acabas de conectar con el kiosk Huella Digital. "
                "1) Llama get_session_state primero — la UI puede estar en "
                "cualquier pantalla, no asumas attract. "
                "2) Narra la pantalla actual siguiendo NOVA_INSTRUCTIONS: "
                "   • attract: present_content(attract_tour, -1); invita a "
                "acercar la manilla; PROHIBIDO navigate_journey. "
                "   • welcome + phase preparing: UNA locución larga cálida de "
                "identificación (credencial, validación, preparación); "
                "PROHIBIDO herramientas extra ni checklist. "
                "   • welcome + phase ready: PRIMERO habla — saluda con "
                "facts.name, menciona rol + empresa, entrega el bloque intro "
                "(~30-45 s). SOLO DESPUÉS de terminar de hablar llama "
                "navigate_journey(start_experience). "
                "PROHIBIDO llamar navigate_journey ANTES de hablar. "
                "PROHIBIDO llamar present_content en welcome:ready. "
                "   • identify_gate / identify_search: explica que debe "
                "identificarse; manilla o buscar por nombre. En identify_search "
                "pide el nombre en voz alta y llama fill_search(query=<nombre>) "
                "en cuanto lo diga. PROHIBIDO nombre/scores/dimensiones del visitante. "
                "   • intro / analysis / detail / closing: sigue el flujo "
                "normal de esa pantalla. "
                "3) Compón desde facts.hint — no leas spokenContent literal. "
                "PROHIBIDO abrir con «Hola» o diminutivos."
            ),
            tools=["get_session_state", "present_content", "navigate_journey"],
        )
        await handle

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
        (solo index -1 título), gesture_practice,
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
    async def fill_search(self, context: RunContext, query: str) -> str:
        """Escribe el nombre dictado por el visitante en el campo de búsqueda
        de la pantalla identify_gate / identify_search.

        Llama esta herramienta en cuanto el visitante diga su nombre completo
        durante la fase de búsqueda (identify_search). El texto aparecerá
        automáticamente en el campo y se mostrarán los resultados.
        Solo úsala en identify_gate o identify_search; en otras pantallas no
        tiene efecto visible.
        """
        return await rpc("fill_search", {"query": query})


# Backward-compatible name used by earlier deploys / tests.
NovaAssistant = Assistant


def _build_nova_realtime() -> aws.realtime.RealtimeModel:
    """Amazon Nova Sonic 2 — LiveKit AWS realtime plugin.

    AWS limits each bidirectional Nova stream to 480 seconds. LiveKit's AWS
    plugin keeps this AgentSession (and therefore the room participant) alive,
    renews only its internal Bedrock stream, and replays its retained system
    instructions and chat context. ``NOVA_SESSION_REFRESH_SECONDS`` is mapped
    to the plugin's import-time ``LK_SESSION_MAX_DURATION`` setting above; the
    default 360 seconds leaves a two-minute safety margin.

    The plugin bounds its completionEnd wait before continuing cleanup, so the
    application must not reconnect the LiveKit room or replace AgentSession.
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
        "nova_session_refresh_seconds": NOVA_SESSION_REFRESH_SECONDS,
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

    # ── on_enter race guard ────────────────────────────────────────────────────
    # When the agent joins, on_enter() fires a generate_reply() that calls
    # get_session_state and narrates the current screen. At almost the same
    # moment the client sends a [pantalla:] cue via sendText(topic="lk.chat").
    # Nova Sonic treats this as a second, independent user turn and runs another
    # tool chain in parallel, causing a duplicate get_session_state loop that
    # prevents any audio from being produced.
    #
    # Fix: suppress the first [pantalla:] message that arrives within
    # ON_ENTER_GUARD_SECS of session start. on_enter already has full context
    # from get_session_state, so the initial pantalla cue is redundant.
    # Subsequent cues (screen changes) are always processed normally.
    on_enter_guard_secs = 2.5
    _session_start_time = time.monotonic()
    _on_enter_handled = False

    def _pantalla_text_input_handler(
        agent_session: AgentSession, event: room_io.TextInputEvent
    ) -> None:
        nonlocal _on_enter_handled
        elapsed = time.monotonic() - _session_start_time
        is_pantalla = event.text.startswith("[pantalla:")
        if is_pantalla and not _on_enter_handled and elapsed < on_enter_guard_secs:
            logger.info(
                "[text_input] Suppressing duplicate pantalla cue (elapsed=%.2fs, "
                "on_enter still active): %.80s",
                elapsed,
                event.text,
            )
            return
        if is_pantalla:
            _on_enter_handled = True
        # Default behaviour: interrupt + generate reply
        agent_session.interrupt()
        agent_session.generate_reply(user_input=event.text)
    # ──────────────────────────────────────────────────────────────────────────

    logger.info(
        "Starting Nova Sonic voice=%s region=%s refresh=%ss",
        NOVA_VOICE,
        AWS_REGION,
        NOVA_SESSION_REFRESH_SECONDS,
    )

    @session.on("user_input_transcribed")
    def _on_user_speech(event) -> None:
        if not event.is_final:
            return
        logger.info("[USER] %s", event.transcript)

    @session.on("conversation_item_added")
    def _on_item_added(event) -> None:
        item = event.item
        role = getattr(item, "role", "?")
        text = getattr(item, "text_content", None)
        if text:
            interrupted = getattr(item, "interrupted", False)
            suffix = " [interrupted]" if interrupted else ""
            logger.info("[%s]%s %s", role.upper(), suffix, text)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            # lk.chat text input MUST remain enabled — notifyGuideScreen (client)
            # sends [pantalla:…] cues via sendText on this topic. These are NOT
            # participant attribute updates; they are real screen-nudge messages
            # that the agent reads to know when to narrate or ask for confirmation.
            # Custom handler suppresses the first duplicate cue that races with
            # on_enter (see _pantalla_text_input_handler above).
            text_input=room_io.TextInputOptions(
                text_input_cb=_pantalla_text_input_handler,
            ),
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
