"""Huella Digital voice guide — Amazon Nova Sonic 2 (LiveKit AWS realtime)."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import logging.handlers
import os
import re
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
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics, cartesia

from nova_session_continuation import install_nova_session_continuation_fix
from rpc_client import rpc, wait_for_kiosk_participant
from tasks.intro_orchestrator import intro_tour_running, schedule_intro_tour
from narration_barrier import NarrationBarrier, set_session_narration_barrier

# The AWS plugin reads LK_SESSION_MAX_DURATION while its realtime module is
# imported. Load local configuration and publish our renewal policy first;
# doing this after importing ``livekit.plugins.aws`` silently has no effect.
load_dotenv(".env")
load_dotenv(".env.local", override=True)

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

NOVA_SESSION_REFRESH_SECONDS = int(os.getenv("NOVA_SESSION_REFRESH_SECONDS", "420"))
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

# Voice backend: nova (default) | cartesia (STT/LLM Inference + Cartesia Sonic TTS)
VOICE_BACKEND = os.getenv("VOICE_BACKEND", "nova").strip().lower()

# Cartesia plugin — custom voice via CARTESIA_VOICE UUID (see huella-guide-cartesia)
STT_MODEL = os.getenv("STT_MODEL", "deepgram/nova-3")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "multi")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-31b-it")
TTS_MODEL = os.getenv("TTS_MODEL", "sonic-3")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "").strip()
CARTESIA_VOICE = os.getenv("CARTESIA_VOICE", "").strip()
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "es")
TTS_SPEED = os.getenv("TTS_SPEED", "1.0")

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

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
    Entra directo al contenido (hecho o idea de la pantalla) —
    sin anunciar el cambio de tarjeta («Ahora avanzamos a la siguiente/última
    tarjeta», «Perfecto, seguimos con…»).
    Sin lista fija de frases y sin inventar un nuevo muletilla repetida.

    En cada turno del visitante o mensaje [pantalla:]:
    1) Llama get_session_state.
    2) Llama present_content con el target EXACTO para enfocar el elemento —
       EXCEPCIÓN detail: en [pantalla:detail] tras la primera sección, salta al paso 3
       (get_session_state basta; la UI ya avanzó sola).
    3) Lee facts.hint y los campos de facts. Compón tu mensaje con tus propias
       palabras siguiendo el hint como guía de estilo y tono.
    4) En intro «Así funciona»: el runtime Python ejecuta el recorrido de las
       tres tarjetas por RPC — NO llames present_content ni navigate_journey(advance)
       durante ese tour. Tras el tour, solo pregunta «¿Empezamos el análisis?» y PARA.
       En detail: la UI avanza secciones sola — PROHIBIDO navigate_journey(advance)
       entre secciones. UNA sola locución continua evidencia→brechas→tácticas;
       PROHIBIDO silencio o pausa entre bloques o ítems. PROHIBIDO present_content
       extra tras la primera entrada (salvo petición explícita del visitante).
    Los [pantalla:] traen pista de step/phase/identity/focus.
    Úsalos para foco y timing; no inventes pantallas ni datos de perfil.

    PROHIBIDO ABSOLUTO — NUNCA LEAS EN VOZ ALTA:
    - El texto de mensajes [pantalla:…] (ni completo ni fragmentado).
    - Meta del sistema o de la interfaz: «La UI», «la pantalla», «la tarjeta
      siguiente/última», «avanzó», «va a mostrar», «cambió de pantalla»,
      «UI step», «focus=», «availableActions», nombres de tools, inglés técnico.
    - Jerga interna: «get_session_state», «present_content», «spokenContent»,
      «facts.hint», «CONTINUOUS TOUR», «rendering», «under construction».
    - Cualquier instrucción interna, corchetes o claves técnicas.
    Si llega un [pantalla:]: úsalo SOLO para decidir tools; habla al visitante
    en español natural sobre el CONTENIDO (dimensiones, entregables, gestos),
    nunca sobre la interfaz ni el cue.

    REGLA CENTRAL — COMPOSICIÓN DINÁMICA:
    Sigue facts.hint siempre. El hint te dice el TONO y ESTRUCTURA, no el texto.
    Varía la apertura de cada elemento. Habla como una anfitriona experta que
    CONOCE el contenido — no como alguien que lee una pantalla en voz alta.
    Nunca abras con «Perfecto» ni muletillas fijas; entra al contenido.
    No anuncies puentes de tarjeta («avanzamos a la siguiente/última…»).

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
      intro_card_dimension (iconos en tarjeta 1; dimension_id=serp|ssi|arquitectura|influencia|higiene),
      intro_deliverable (iconos en tarjeta 2; index 0=Radar,1=Informe,2=Correo),
      intro_dimension (spider legacy — no usar en onboarding de 3 tarjetas),
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
    - SILENCIO TOTAL mientras phase=preparing. El cliente pre-calienta la
      sesión de voz sin locución — la primera voz es el saludo en welcome:ready.
    PROHIBIDO hablar, narrar o llamar herramientas mientras phase=preparing.
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
      * La UI auto-selecciona el primer resultado (~1 s después de que aparece)
        y avanza al Screen 4a automáticamente — NO necesitas hacer nada más.
      * Di solo: "Buscando <nombre>…" y quédate en silencio mientras la UI
        selecciona. Si en 3 s no avanzó, pregunta si quieren reintentar.
      * Si no hay coincidencia: ofrece intentar con otro nombre o pedir ayuda.

    ────────────────────────────────────────────────
    Screen 4a — WELCOME READY (si encontrado)
    ────────────────────────────────────────────────
    Saludo CORTO (~10-12 s), no monólogo largo.
    ORDEN:
    A) Saluda con nombre + rol/empresa. 2-3 oraciones: qué es Huella Digital
       y que explorarán su presencia. Invita a «cómo funciona».
    B) Llama navigate_journey(start_experience).
    Si el visitante dice continuar / adelante / seguimos / listo / empezamos
    EN CUALQUIER MOMENTO: cierra en una frase y llama start_experience — NO
    fuerces el monólogo completo.
    Tras start_experience ok: SILENCIO TOTAL. PROHIBIDO repetir nombre/rol/
    empresa o el saludo. Espera [pantalla:intro:steps] y sigue Screen 5.
    PROHIBIDO: present_content en welcome:ready.
    PROHIBIDO: repetir las 3 tarjetas de onboarding aquí.
    PROHIBIDO: listar las 5 dimensiones aquí — se presentarán en las tarjetas del intro.

    ────────────────────────────────────────────────
    Screen 5 — ONBOARDING «Antes de empezar, así funciona»
    ────────────────────────────────────────────────
    UNA locución POR tarjeta — el cliente avanza la UI entre tarjetas.
    ORDEN:
      Card 0: present_content(intro_step, 0) → SOLO gestos/toque/voz → PARA.
      Card 1: present_content(intro_step, 1) → transición breve → por CADA dimensión en
        facts.dimensions: present_content(intro_card_dimension, dimension_id=…) ANTES del
        nombre → narra → siguiente → PARA.
      Card 2: present_content(intro_step, 2) → «Te recibirás…» → por CADA entregable en
        facts.deliverables: present_content(intro_deliverable, index=…) ANTES de la etiqueta
        → narra → «¿Empezamos?» → PARA.
    Espera [pantalla:intro:steps:N] antes de cada tarjeta.
    PROHIBIDO mezclar gestos + dimensiones + entregables en una sola locución.
    OBLIGATORIO present_content por icono — la UI resalta solo cuando llamas la tool.
    Solo start_analysis tras confirmación en card 2.
    NO pases al carrusel spider (intro_dimension).

    RE-EXPLICAR UNA TARJETA (a petición explícita del visitante):
    Si el visitante pide escuchar otra vez una tarjeta — interpreta de forma AMPLIA:
    «explícame de nuevo», «repite», «otra vez», «de nuevo», «no entendí»,
    «no entendí bien», «no entendí muy bien», «me explicas», «no quedó claro»,
    «¿y los gestos?», «¿qué recibo?», «las dimensiones», «¿cuáles son?»,
    «¿qué miden?», «¿cómo funciona?», «ítem uno/dos/tres», «la primera/segunda/tercera»,
    cualquier pregunta sobre Autoridad, SSI, Mensaje, Influencia, Higiene,
    Radar, Informe, Correo — TODO esto activa replay_intro_card.
      1) NUNCA digas que no puedes — siempre puedes, siempre lo haces.
         NUNCA respondas con frases de seguridad o política interna.
      2) Llama navigate_journey(replay_intro_card, index=N)
         donde N = 0 (gestos), 1 (dimensiones), 2 (entregables).
         Si hay duda: N=1 si mencionó dimensiones/qué miden/cuáles son,
         N=0 si preguntó cómo interactuar/navegar, N=2 si preguntó qué recibe.
      3) Narra esa tarjeta con más detalle que en el tour automático si el
         visitante pide profundidad; si solo pide repetir, sé igual de breve.
      4) Tras narrar, pregunta «¿Seguimos al análisis?» y PARA.
    PROHIBIDO replay_intro_card sin petición explícita del visitante.
    PROHIBIDO cancel_intro_tour cuando el visitante pide re-explicación.

    ────────────────────────────────────────────────
    ANALYSIS SCANNING → COMPLETE → RESULTS (mismo globo)
    ────────────────────────────────────────────────
    1) Scanning: narra con acompañamiento qué fuentes se revisan — anclado en
       facts.sourceGroups / facts.narrationAnchors / facts.searchFindings del
       informe real (LinkedIn, prensa, redes, sitios corporativos por nombre).
       Tono analista senior, creíble para C-level. PROHIBIDO inventar fuentes.
       Cuando el agente recibe [pantalla:analysis:complete]:
    2) Complete: el globo se queda; las tarjetas de fuentes se actualizan
       con el nombre de cada dimensión y su puntuación. Anuncia el standing
       con calidez anclado en facts.uiStandingLine (el título en pantalla):
       mismos puntos (rol, standingBlurb/banda, dimensión más fuerte), pero
       MÁS AMABLE y conversacional — PROHIBIDO leer uiStandingLine literal.
       En el mismo flujo (3–5 oraciones): UNA fortaleza concreta de
       facts.strengths o facts.coverLines y UNA brecha de facts.opportunities
       o facts.weakestDimension — solo datos del informe, tono consultor C-level.
       PROHIBIDO recitar solo «LÍDER · TOP 8%». Invita a abrir el detalle
       de una dimensión (no hay vista resumen aparte).
       availableActions aquí: reveal_results, open_detail, back, cancel —
       PROHIBIDO navigate_journey(advance).
       Si el visitante pide ver el detalle de una dimensión (o «detalle»,
       «fortalezas», «oportunidades», «plan»): navigate_journey(open_detail,
       dimension_id=serp|ssi|arquitectura|influencia|higiene) O
       present_content(detail_dimension, dimension_id=…).
       PROHIBIDO present_content(result_dimension) en complete — solo resalta
       y requiere phase=results; no abre el detalle.
    3) Cuando llega [pantalla:analysis:results dim 0]: mismo globo con
       tarjeta activa resaltada. Ciclo de dimensiones ahí.

    RESULTADOS (result_dimension) — sobre el globo (solo phase=results):
    Ciclo completo para CADA dimensión:
      present_content(result_dimension, index=N) → narra: contexto primero,
      score al final de forma casual. Varía la apertura cada vez → luego
      navigate_journey(advance) para ir a la siguiente dimensión.
    PROHIBIDO pedir «continuar» ni esperar al visitante entre dimensiones.
    Si el visitante pregunta por una dimensión concreta en results: puedes
    present_content(result_dimension, dimension_id=X) para narrar, luego
    navigate_journey(open_detail, dimension_id=X) si quiere fortalezas /
    oportunidades / plan de acción.
    Tras la última dimensión: advance automáticamente → detalle.

    DETALLE (detail_section) — MISMO GLOBO, AVANCE AUTOMÁTICO:
    Audiencia: presidentes y directivos C-level — tono creíble, consultivo,
    anclado en el informe real (facts.evidence / facts.gaps / facts.tactics).
    Parafrasea para facts.role en facts.company — explica el POR QUÉ; PROHIBIDO
    leer facts.items literalmente.
    La UI resalta Fortalezas → Oportunidades → Plan sola mientras hablas.
    Primera entrada: present_content(detail_section, section=strengths) UNA vez.
    UNA locución continua SIN silencios: evidencia → brechas → tácticas encadenadas
    con conectores («Además…», «Donde veo margen…», «En concreto…»).
    PROHIBIDO parar, callar o pausar entre bloques o ítems.
    PROHIBIDO present_content entre bloques. PROHIBIDO navigate_journey(advance).
    PROHIBIDO rótulos «Fortalezas/Oportunidades/Plan de acción».
    PROHIBIDO leer facts.items literalmente — prosa fluida C-level.
    Al cerrar tácticas: UNA pregunta — informe, volver al globo u otra dimensión — PARA y ESPERA.
    PROHIBIDO repetir el detalle de una dimensión ya narrada (facts.detailTourComplete).
    send_report | back | open_detail(dimension_id=…) según respuesta.

    DETALLE → VOLVER (back desde detail):
    Cuando el visitante dice "volver" o navega BACK desde detail, la UI
    regresa al globo con las tarjetas de dimensiones (no a un resumen).
    NO describas de nuevo el resumen de la dimensión
    a menos que el visitante lo pida explícitamente.
    Di solo algo breve como «De vuelta a tus dimensiones. ¿Cuál quieres
    ver?» o «¿Revisamos otra dimensión o avanzamos al cierre?»
    Espera su elección.

    CIERRE / FOTO / TARJETA:
    - photo: UNA locución — huella digital → paquete (informe + tarjeta + foto)
      para su correo; la foto es la imagen de la tarjeta. Invita al recuadro.
      Cuando el visitante confirme (listo, toma la foto, adelante): navigate_journey(ready_for_picture).
    - generating: locución CORTA — componiendo entrega para su correo.
      PROHIBIDO pedir tomar foto — la captura ya ocurrió.
    - delivered: revisar tarjeta; informe e imagen van juntos a su correo;
      «Enviar reporte» / retake_photo / advance.
    - thanks: agradecimiento cálido; invita a escanear el QR para conocer más de SETI;
      cuando confirmen salir (sí, finalizar, finish): navigate_journey(finish).
      PROHIBIDO repetir análisis o entrega.

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


async def _load_session_state_for_enter() -> dict:
    """Fetch kiosk UI state once for on_enter (no tool round-trip in Nova)."""
    try:
        raw = await rpc("get_session_state", retries=2)
        state = json.loads(raw)
        if isinstance(state, dict):
            return state
    except (ToolError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("on_enter: get_session_state failed: %s", exc)
    return dict(_FALLBACK_SESSION)


def on_enter_should_defer(state: dict) -> bool:
    """True when the browser will drive the first reply via [pantalla:] cues."""
    if not state.get("ok", True):
        return True
    step = state.get("step")
    phase = state.get("phase")
    # welcome:preparing — silent prewarm; welcome:ready — client sends cue on
    # voice enable (avoids racing on_enter with a parallel get_session_state).
    if step == "welcome" and phase in ("preparing", "ready"):
        return True
    return False


def build_on_enter_instructions(state: dict) -> str:
    """Short Nova prompt with pre-fetched state (no get_session_state in reply)."""
    step = state.get("step", "attract")
    phase = state.get("phase", "ready")
    facts = state.get("facts") if isinstance(state.get("facts"), dict) else {}
    hint = str(facts.get("hint", "")).strip()
    hint_suffix = f" {hint}" if hint else ""

    if step == "attract":
        return (
            "Pantalla attract. Llama present_content(attract_tour, -1), invita a "
            "acercar la manilla. PROHIBIDO navigate_journey."
        )
    if step in ("identify_gate", "identify_search"):
        return (
            f"Pantalla {step}. "
            "Explica brevemente (2 frases máx): el sistema no confirmó la credencial, "
            "pueden acercar la manilla de nuevo o decir su nombre completo en voz alta. "
            "No anuncies lo que harás — simplemente actúa cuando el visitante hable. "
            "NO llames navigate_journey."
        )
    if step == "intro":
        return (
            f"Pantalla intro ({phase}). Sigue NOVA_INSTRUCTIONS; compón desde "
            f"facts.{hint_suffix}".strip()
        )
    title = state.get("title", "Huella Digital")
    return (
        f"Reconectaste en {step}:{phase} ({title}). Resume en una frase breve "
        f"y continúa el flujo.{hint_suffix}"
    ).strip()


class Assistant(Agent):
    """Nova Sonic host: fixed tools, UI-first spokenContent contract.

    Per LiveKit Nova Sonic guide: register @function_tool on the Agent;
    put tool_choice on RealtimeModel (not per-reply). Warm greeting via
    on_enter + generate_reply (Nova Sonic 2 mixed modalities).
    """

    def __init__(self, on_enter_done: asyncio.Event | None = None) -> None:
        super().__init__(instructions=NOVA_INSTRUCTIONS)
        self._on_enter_done = on_enter_done

    async def on_enter(self) -> None:
        # Voice connects mid-journey (camera detect → identifying). Read the
        # live UI step first — never assume attract.
        #
        # Cartesia STT/LLM/TTS is slower than Nova realtime; the browser often
        # joins a few seconds after the agent. RPC without a participant forces
        # soft-fail state and breaks the voice-driven UI sync.
        await wait_for_kiosk_participant()

        state = await _load_session_state_for_enter()

        if on_enter_should_defer(state):
            logger.info(
                "on_enter: deferring to client (step=%s phase=%s ok=%s)",
                state.get("step"),
                state.get("phase"),
                state.get("ok"),
            )
            if self._on_enter_done is not None:
                self._on_enter_done.set()
            return

        # Do NOT pass tools= here. All four @function_tool methods on this
        # class are injected into the initial Bedrock session schema by the
        # SDK automatically. Passing tools= overrides that injection and
        # causes the AWS plugin to see fill_search as a mid-session addition,
        # triggering a full Bedrock stream recycle (~2 s silence penalty) the
        # first time identify_gate appears.
        instructions = build_on_enter_instructions(state)
        try:
            logger.info(
                "on_enter: generate_reply step=%s phase=%s",
                state.get("step"),
                state.get("phase"),
            )
            handle = self.session.generate_reply(instructions=instructions)
            await handle
        finally:
            # Signal that on_enter has fully completed (audio delivered).
            # The pantalla guard (_on_enter_done event) will unblock any
            # subsequent [pantalla:] cues only after this point, preventing
            # the welcome greeting from being repeated.
            if self._on_enter_done is not None:
                self._on_enter_done.set()

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
        2=tercero), intro_card_dimension (tarjeta Las 5 dimensiones;
        mejor dimension_id=serp|ssi|arquitectura|influencia|higiene),
        intro_deliverable (Qué recibirás; index 0=Radar,1=Informe,2=Correo
        o dimension_id=radar|informe|correo),         intro_dimension (spider legacy), result_dimension (solo analysis:results),
        detail_dimension (+dimension_id; usar en complete si piden «detalle»),
        detail_section
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
        self, context: RunContext, action: str, dimension_id: str = "", index: int = -1
    ) -> str:
        """Ejecuta una acción disponible en la experiencia.
        - open_detail + dimension_id: abre detalle de dimensión en analysis:complete.
        - ready_for_picture: en closing:pose|capture cuando el visitante confirma la foto.
        - finish: en closing:thanks cuando confirma salir.
        - replay_intro_card + index (0=gestos, 1=dimensiones, 2=entregables):
          vuelve a narrar esa tarjeta cuando el visitante lo pide explícitamente.
          Solo disponible mientras step=intro."""
        payload: dict = {"action": action, "dimensionId": dimension_id}
        if index >= 0:
            payload["index"] = index
        return await rpc("navigate_journey", payload)

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


# Backward-compatible names used by earlier deploys / tests.
NovaAssistant = Assistant
CartesiaAssistant = Assistant

_SPEED_PRESETS = {"slow": 0.8, "normal": 1.0, "fast": 1.2}


def _plugin_speed() -> float:
    key = TTS_SPEED.strip().lower()
    if key in _SPEED_PRESETS:
        return _SPEED_PRESETS[key]
    try:
        return float(TTS_SPEED)
    except ValueError:
        return 1.0


def _build_tts() -> cartesia.TTS:
    """Custom Cartesia voices require the plugin + API key."""
    if not CARTESIA_API_KEY:
        raise RuntimeError(
            "CARTESIA_API_KEY is required for custom voices (e.g. angel). "
            "Create one at https://play.cartesia.ai/keys and add it as a "
            "LiveKit agent secret."
        )
    if not CARTESIA_VOICE:
        raise RuntimeError(
            "CARTESIA_VOICE must be your custom voice UUID from Cartesia "
            "(Voice Library → angel → copy ID)."
        )
    if not _UUID_RE.match(CARTESIA_VOICE):
        raise RuntimeError(
            f"CARTESIA_VOICE must be a UUID, not a name like {CARTESIA_VOICE!r}. "
            "In Cartesia: Voice Library → angel → ⋯ → Copy ID."
        )
    model = TTS_MODEL.removeprefix("cartesia/").strip() or "sonic-3"
    logger.info(
        "Cartesia TTS model=%s voice=%s… lang=%s key_set=%s",
        model,
        CARTESIA_VOICE[:8],
        TTS_LANGUAGE,
        bool(CARTESIA_API_KEY),
    )
    return cartesia.TTS(
        model=model,
        voice=CARTESIA_VOICE,
        language=TTS_LANGUAGE,
        speed=_plugin_speed(),
        word_timestamps=False,
    )


def _build_cartesia_session() -> AgentSession:
    """STT/LLM via Inference; TTS via Cartesia plugin (custom voice)."""
    return AgentSession(
        stt=inference.STT(model=STT_MODEL, language=STT_LANGUAGE),
        llm=inference.LLM(model=LLM_MODEL),
        tts=_build_tts(),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            # Cartesia pipeline is slower than Nova; wait longer before committing
            # partial STT ("¿Cómo" alone) and start TTS earlier once a turn ends.
            endpointing={"min_delay": 0.9, "max_delay": 4.5},
            interruption={"min_words": 2, "min_duration": 0.65},
            preemptive_generation={"preemptive_tts": True},
        ),
        max_tool_steps=8,
    )


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
        generate_reply_timeout=45.0,
        temperature=0.7,
        top_p=0.9,
        # Nova Sonic 2 defaults to mixed modalities (audio + text),
        # which enables on_enter generate_reply warm intro.
    )


_PANTALLA_DEDUPE_SECONDS = 2.5
_USER_VOICE_MIN_CHARS = 4

# Appended to every visitor voice turn so Nova maps intent → navigate_journey.
_USER_VOICE_TOOL_HINT = (
    "Llama get_session_state primero. "
    "Si step=welcome y phase=ready y el visitante confirma continuar "
    "(sí, continúa, adelante, comienza, empezamos, listo, vamos, sigan, dale): "
    "navigate_journey(start_experience) de inmediato tras UNA frase de cierre breve — "
    "no re-narres las dimensiones ni repitas el saludo completo. "
    "Si step=intro y el visitante pide re-explicar una tarjeta — "
    "interpreta de forma AMPLIA: cualquier señal de confusión, re-consulta o petición sobre "
    "el contenido de las tarjetas debe activar replay_intro_card. "
    "Señales de re-explicación de DIMENSIONES (N=1): "
    "«no entendí», «no entendí bien», «no entendí muy bien», «me explicas», «explícame», "
    "«explícame de nuevo», «repite», «repíteme», «otra vez», «de nuevo», «¿qué son las dimensiones?», "
    "«¿cuáles son?», «las dimensiones», «¿qué miden?», «dimensión», «ítem dos», «ítem número dos», "
    "«el segundo», «la segunda tarjeta», «las cinco», «vuelve a explicar», «no quedó claro», "
    "«no entendí el número dos», «¿cuáles son las dimensiones?», «Autoridad», «SSI», «Mensaje», "
    "«Influencia», «Higiene» (cuando pregunta qué son). "
    "Señales de re-explicación de GESTOS (N=0): "
    "«los gestos», «¿cómo funciona?», «¿cómo interactúo?», «la primera tarjeta», «ítem uno», "
    "«¿cómo avanzo?», «¿cómo navego?», «deslizar». "
    "Señales de re-explicación de ENTREGABLES (N=2): "
    "«¿qué recibo?», «el informe», «el radar», «¿qué me dan?», «la tercera tarjeta», «ítem tres», "
    "«¿qué incluye?», «los entregables». "
    "Ante cualquier duda sobre cuál tarjeta, usa N=1 (dimensiones) si mencionó dimensiones, "
    "N=0 (gestos) si preguntó cómo interactuar, N=2 (entregables) si preguntó qué recibe. "
    "NUNCA respondas con 'no puedo' ni rechaces — SIEMPRE ejecuta replay_intro_card. "
    "Luego narra esa tarjeta con más detalle si pide profundidad, o breve si solo repite. "
    "Si step=closing y phase=pose|prep|capture y el visitante pide tomar la foto "
    "(toma la foto, listo, estoy listo, adelante, take picture, toma la): "
    "navigate_journey(ready_for_picture) — no solo hables, ejecuta la acción. "
    "Si step=closing y phase=thanks y confirma salir "
    "(sí, finalizar, finish, terminamos, listo para salir, yes): "
    "di una frase breve de despedida y navigate_journey(finish). "
    "Si step=closing y phase=delivered y pide enviar reporte: navigate_journey(advance) o send_report según availableActions."
)


def _pantalla_dedupe_key(text: str) -> str:
    if "intro:run" in text or "INTRO_ORCHESTRATOR" in text:
        return "intro:run"
    if "closing:photo" in text:
        return "closing:photo"
    if "closing:generating" in text:
        return "closing:generating"
    if "closing:delivered" in text:
        return "closing:delivered"
    if "closing:thanks" in text or ("step=closing" in text and "phase=thanks" in text):
        return "closing:thanks"
    if "analysis:complete" in text:
        return "analysis:complete"
    if "detail:revisit" in text or "DETAIL_REVISIT" in text:
        return "detail:revisit"
    if "detail:continuous" in text or "DETAIL_CONTINUOUS" in text:
        return "detail:continuous"
    match = re.search(r"\[pantalla:([^\]]+)\]", text)
    if match:
        return match.group(1).split("]")[0].strip()
    return text[:96]


def _closing_pantalla_instructions(text: str) -> str | None:
    if "closing:photo" in text or (
        "step=closing" in text and "phase=pose" in text
    ):
        return (
            "CLOSING PHOTO — get_session_state. "
            "UNA locución: huella → paquete (informe + tarjeta + foto) para su correo; "
            "invita al recuadro. Si confirman estar listos: navigate_journey(ready_for_picture). "
            "PROHIBIDO repetir si ya narraste la foto en este phase."
        )
    if "closing:generating" in text:
        return (
            "CLOSING GENERATING — get_session_state. "
            "Locución CORTA (1-2 frases): componiendo tarjeta e informe para su correo. "
            "PROHIBIDO pedir tomar foto — la captura YA ocurrió. "
            "PROHIBIDO repetir el mismo mensaje de entrega."
        )
    if "closing:delivered" in text:
        return (
            "CLOSING DELIVERED — get_session_state. "
            "UNA frase: revisa la tarjeta; informe y foto van juntos a su correo. "
            "Pregunta si envían el reporte ahora. "
            "PROHIBIDO repetir frases ya dichas en generating o photo."
        )
    if "closing:thanks" in text or (
        "step=closing" in text and "phase=thanks" in text
    ):
        return (
            "CLOSING THANKS — get_session_state. "
            "Agradecimiento cálido + invita a escanear el QR de SETI. "
            "Si el visitante confirma salir (sí, finalizar, finish, terminamos, listo): "
            "navigate_journey(finish) de inmediato. "
            "Si aún no confirmó: pregunta si finalizamos → ESPERA."
        )
    return None


def _telemetry_record_option() -> bool | dict[str, bool]:
    flag = os.getenv("LIVEKIT_TELEMETRY", "on").strip().lower()
    if flag in ("0", "false", "off", "disabled", "no"):
        return {"traces": False, "logs": False}
    return True


server = AgentServer(
    # Keep one idle Python process pre-warmed so the next guest never waits
    # for a cold-start (~2 s). After a session ends the pool refills immediately.
    num_idle_processes=1,
)


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: JobContext):
    use_cartesia = VOICE_BACKEND == "cartesia"

    if use_cartesia:
        ctx.log_context_fields = {
            "room": ctx.room.name,
            "agent": AGENT_NAME,
            "voice_backend": "cartesia-plugin",
            "tts_model": TTS_MODEL,
            "cartesia_voice": CARTESIA_VOICE,
            "stt_model": STT_MODEL,
            "llm_model": LLM_MODEL,
        }
        session = _build_cartesia_session()
    else:
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
    # get_session_state and narrates the current screen. The client also sends
    # a [pantalla:] cue shortly after the room connects. Because on_enter can
    # take 10-15 s (get_session_state RTT + TTS audio), the pantalla cue often
    # arrives AFTER the old time-based guard (2.5 s) expired, causing a second
    # generate_reply that repeats the greeting.
    #
    # Fix: use an asyncio.Event that is set only after on_enter() fully
    # completes (await handle returns). All [pantalla:] cues received before
    # that event are suppressed — on_enter already covers the initial screen.
    # Subsequent cues (real screen changes) are processed normally.
    _on_enter_done = asyncio.Event()
    # Cartesia on_enter + TTS can take 15s+; queue [pantalla:] cues instead of dropping.
    _pantalla_queue: list[str] = []
    _pantalla_guard: dict[str, object] = {
        "last_key": "",
        "last_at": 0.0,
        "once_keys": set(),
    }

    def _pantalla_already_narrated(key: str) -> bool:
        once = _pantalla_guard["once_keys"]
        assert isinstance(once, set)
        return key in once

    def _mark_pantalla_narrated(key: str) -> None:
        once = _pantalla_guard["once_keys"]
        assert isinstance(once, set)
        once.add(key)

    def _should_skip_duplicate_pantalla(key: str) -> bool:
        now = time.monotonic()
        last_key = _pantalla_guard["last_key"]
        last_at = _pantalla_guard["last_at"]
        assert isinstance(last_key, str)
        assert isinstance(last_at, float)
        if key == last_key and (now - last_at) < _PANTALLA_DEDUPE_SECONDS:
            return True
        _pantalla_guard["last_key"] = key
        _pantalla_guard["last_at"] = now
        return False

    def _pantalla_text_input_handler(
        agent_session: AgentSession, event: room_io.TextInputEvent
    ) -> None:
        is_pantalla = event.text.startswith("[pantalla:")
        if is_pantalla and not _on_enter_done.is_set():
            if use_cartesia:
                _pantalla_queue.append(event.text)
                logger.info(
                    "[text_input] Queued pantalla cue (on_enter active): %.80s",
                    event.text,
                )
            else:
                logger.info(
                    "[text_input] Suppressing pantalla cue (on_enter still active): %.80s",
                    event.text,
                )
            return
        # CRITICAL: never pass the raw [pantalla:] English cue as user_input —
        # Nova often reads it aloud ("UI step…", "focus=…"). Use instructions
        # so the model runs tools and speaks visitor-facing Spanish only.
        if is_pantalla:
            logger.info(
                "[text_input] pantalla cue (not spoken): %.120s",
                event.text,
            )
            dedupe_key = _pantalla_dedupe_key(event.text)
            if _should_skip_duplicate_pantalla(dedupe_key):
                logger.info(
                    "[text_input] Skipping duplicate pantalla (%.1fs): %s",
                    _PANTALLA_DEDUPE_SECONDS,
                    dedupe_key,
                )
                return

            intro_orchestrator_start = (
                "INTRO_ORCHESTRATOR_START" in event.text
                or "intro:run" in event.text
            )
            intro_continuous = (
                "INTRO_CONTINUOUS_TOUR" in event.text
                or "INTRO_CARD_READY" in event.text
            )
            detail_revisit = (
                "detail:revisit" in event.text or "DETAIL_REVISIT" in event.text
            )
            detail_auto = (
                "detail:continuous" in event.text
                or "DETAIL_CONTINUOUS" in event.text
                or detail_revisit
                or "step=detail" in event.text
                or "focus=detail" in event.text
            )
            welcome_ready = (
                "[pantalla:welcome:ready]" in event.text
                or (
                    "phase=ready" in event.text
                    and "welcome" in event.text
                )
            )
            welcome_preparing = (
                "[pantalla:welcome:preparing]" in event.text
                or (
                    "phase=preparing" in event.text
                    and "welcome" in event.text
                )
            )
            closing_instructions = _closing_pantalla_instructions(event.text)

            if welcome_preparing:
                logger.info(
                    "[text_input] Ignoring preparing pantalla (silent prewarm): %.80s",
                    event.text,
                )
                return

            if intro_orchestrator_start or intro_continuous:
                agent_session.interrupt()
                if schedule_intro_tour(agent_session):
                    logger.info(
                        "[text_input] intro orchestrator started from pantalla"
                    )
                else:
                    logger.info(
                        "[text_input] intro orchestrator already active — "
                        "ignored pantalla: %.80s",
                        event.text,
                    )
                return

            # Python orchestrator owns intro narration — ignore other intro cues.
            if intro_tour_running() or (
                "step=intro" in event.text and "intro:run" not in event.text
            ):
                logger.info(
                    "[text_input] Suppressed pantalla during intro orchestrator: %s",
                    dedupe_key,
                )
                return

            if not detail_auto and not welcome_preparing:
                agent_session.interrupt()

            if detail_revisit:
                agent_session.generate_reply(
                    instructions=(
                        "DETAIL_REVISIT — Esta dimensión ya se narró. "
                        "PROHIBIDO repetir evidencia/brechas/tácticas. "
                        "PROHIBIDO present_content(detail_section). "
                        "Di UNA frase breve: informe, volver al globo u otra dimensión → ESPERA. "
                        "send_report | back | open_detail(dimension_id=…)."
                    ),
                )
            elif detail_auto:
                agent_session.generate_reply(
                    instructions=(
                        "DETAIL_CONTINUOUS — UNA sola respuesta SIN silencios ni pausas. "
                        "Teje facts.evidence → facts.gaps → facts.tactics en prosa encadenada. "
                        "Parafrasea para facts.role en facts.company — explica POR QUÉ. "
                        "PROHIBIDO parar, callar o END entre bloques o ítems. "
                        "PROHIBIDO present_content extra ni get_session_state entre bloques. "
                        "PROHIBIDO rótulos Fortalezas/Oportunidades/Plan. "
                        "UI resalta secciones sola — tú sigues hablando sin interrupción. "
                        "Al cerrar tácticas: pregunta informe / volver / otra dimensión → PARA y ESPERA."
                    ),
                )
            elif welcome_ready:
                agent_session.generate_reply(
                    instructions=(
                        "WELCOME READY (Screen 4a) — ORDEN ESTRICTO: "
                        "1) get_session_state. "
                        "2) HABLA PRIMERO (~10-12 s): saluda con facts.name, rol y empresa; "
                        "2-3 frases sobre Huella Digital e invita a «cómo funciona». "
                        "PROHIBIDO listar las 5 dimensiones aquí — se explicarán en las tarjetas del intro. "
                        "PROHIBIDO present_content. PROHIBIDO navigate_journey mientras hablas. "
                        "3) SOLO después de terminar el saludo: navigate_journey(start_experience). "
                        "Tras start_experience ok: SILENCIO — no repitas nombre; espera [pantalla:intro]."
                    ),
                )
            elif closing_instructions:
                once_key = dedupe_key
                if _pantalla_already_narrated(once_key):
                    logger.info(
                        "[text_input] Skipping repeat closing pantalla: %s", once_key
                    )
                    return
                _mark_pantalla_narrated(once_key)
                agent_session.generate_reply(instructions=closing_instructions)
            elif dedupe_key == "analysis:complete" and _pantalla_already_narrated(
                "analysis:complete"
            ):
                logger.info("[text_input] Skipping repeat analysis:complete pantalla")
                return
            else:
                if dedupe_key == "analysis:complete":
                    _mark_pantalla_narrated("analysis:complete")
                agent_session.generate_reply(
                    instructions=(
                        "Cambio de foco en pantalla. "
                        "Llama get_session_state primero — ancla en step/phase actuales. "
                        "Luego present_content solo si hace falta. "
                        "PROHIBIDO repetir la misma locución si ya cubriste este phase. "
                        "PROHIBIDO UI/pantalla/tarjeta meta."
                    ),
                )
        else:
            transcript = event.text.strip()
            if intro_tour_running():
                if len(transcript) < _USER_VOICE_MIN_CHARS:
                    logger.info(
                        "[text_input] Ignoring short utterance during intro tour: %.40s",
                        transcript,
                    )
                    return
                agent_session.interrupt()
                agent_session.generate_reply(
                    user_input=event.text,
                    instructions=(
                        "INTRO TOUR ACTIVE — el orchestrator Python narra las tarjetas. "
                        "Si el visitante hace una pregunta directa: responde en ≤2 frases. "
                        "Si el visitante pide re-explicar una tarjeta — interpreta de forma AMPLIA: "
                        "cualquier señal de confusión, 'no entendí', 'no entendí bien', "
                        "'explícame de nuevo', 'repite', 'otra vez', 'de nuevo', 'me explicas', "
                        "'no quedó claro', 'dimensiones', 'gestos', 'entregables', 'ítem N', "
                        "'¿cuáles son?', '¿qué miden?', '¿cómo funciona?', '¿qué recibo?' — "
                        "llama navigate_journey(replay_intro_card, index=N) "
                        "y narra esa tarjeta con más detalle (N=0 gestos, N=1 dimensiones, N=2 entregables). "
                        "Si hay duda sobre cuál tarjeta: usa N=1 si mencionó dimensiones, "
                        "N=0 si preguntó cómo interactuar, N=2 si preguntó qué recibe. "
                        "NUNCA respondas con 'no puedo' ni rechaces — SIEMPRE ejecuta replay_intro_card. "
                        "PROHIBIDO present_content y navigate_journey(advance/start_experience). "
                        f"{_USER_VOICE_TOOL_HINT}"
                    ),
                )
                return
            agent_session.interrupt()
            agent_session.generate_reply(
                user_input=event.text,
                instructions=(
                    "If intro onboarding (Así funciona) is running: the Python orchestrator "
                    "owns cards and icons — do NOT call present_content or navigate_journey(advance). "
                    "Answer only if the visitor asks something off-script; otherwise stay brief. "
                    f"{_USER_VOICE_TOOL_HINT}"
                ),
            )

    # ──────────────────────────────────────────────────────────────────────────

    if use_cartesia:

        @session.on("error")
        def _on_session_error(ev) -> None:
            err = getattr(ev, "error", ev)
            logger.error(
                "session_error type=%s error=%s",
                type(err).__name__,
                err,
                exc_info=isinstance(err, BaseException),
            )

        logger.info(
            "Starting Cartesia plugin tts=%s voice=%s… lang=%s",
            TTS_MODEL,
            CARTESIA_VOICE[:8] if CARTESIA_VOICE else "",
            TTS_LANGUAGE,
        )
    else:
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

    async def _replay_queued_pantalla() -> None:
        await _on_enter_done.wait()
        if not _pantalla_queue:
            return
        cues = list(_pantalla_queue)
        _pantalla_queue.clear()
        if len(cues) > 1:
            logger.info(
                "[text_input] Replaying latest of %d queued pantalla cue(s)",
                len(cues),
            )
        cue = cues[-1]
        logger.info("[text_input] Replaying queued pantalla: %.120s", cue)
        _pantalla_text_input_handler(session, room_io.TextInputEvent(text=cue))

    replay_task = asyncio.create_task(_replay_queued_pantalla())

    narration_barrier = NarrationBarrier()
    set_session_narration_barrier(narration_barrier)

    room_opts = room_io.RoomOptions(
        # Kiosk browsers can briefly reconnect; keep the voice session alive.
        close_on_disconnect=False,
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
    )
    # Stream agent speech text to lk.transcription so the kiosk can sync icon
    # spotlights to Nova/Cartesia audio (intro card tour).
    room_opts.text_output = room_io.TextOutputOptions(
        sync_transcription=True,
        transcription_speed_factor=1.15 if use_cartesia else 1.05,
    )

    await session.start(
        agent=Assistant(on_enter_done=_on_enter_done),
        room=ctx.room,
        room_options=room_opts,
        record=_telemetry_record_option(),
    )

    await ctx.connect()

    @ctx.room.local_participant.register_rpc_method("narration_segment_done")
    async def _on_narration_segment_done(data) -> str:
        try:
            payload = json.loads(data.payload or "{}")
        except json.JSONDecodeError:
            payload = {}
        segment_id = str(payload.get("segmentId") or "")
        token = int(payload.get("token") or 0)
        ok = narration_barrier.ack(segment_id, token)
        return json.dumps({"ok": ok})

    await replay_task


if __name__ == "__main__":
    cli.run_app(server)
