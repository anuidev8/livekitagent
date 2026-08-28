"""Huella Digital voice guide — Amazon Nova Sonic 2 (LiveKit AWS realtime)."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable
import asyncio
import importlib
import json
import logging
import logging.handlers
import os
import re
import textwrap
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
from livekit.agents.voice.agent import Agent as VoiceAgent, ModelSettings
from livekit.plugins import ai_coustics, cartesia, elevenlabs

from nova_session_continuation import install_nova_session_continuation_fix
from rpc_client import rpc, wait_for_kiosk_participant

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

# Voice backend: nova | cartesia | elevenlabs
# cartesia / elevenlabs = Deepgram STT + Bedrock LLM + plugin TTS
VOICE_BACKEND = os.getenv("VOICE_BACKEND", "nova").strip().lower()
HALF_CASCADE_BACKENDS = frozenset({"cartesia", "elevenlabs"})
if VOICE_BACKEND not in {"nova", *HALF_CASCADE_BACKENDS}:
    raise ValueError(
        f"VOICE_BACKEND must be nova, cartesia, or elevenlabs — got {VOICE_BACKEND!r}"
    )

# Cartesia plugin — custom voice via CARTESIA_VOICE UUID (see huella-guide-cartesia)
STT_MODEL = os.getenv("STT_MODEL", "deepgram/nova-3")
# Fixed Spanish beats multi for kiosk STT (logs: late transcripts with multi).
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "es")
# Bedrock Claude by default — Gemma inference lacks reliable multi-tool chains here.
# Nova Pro: better tool chains than Lite; same Bedrock grant as Nova Sonic.
DEFAULT_CARTESIA_LLM = "amazon.nova-pro-v1:0"
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_CARTESIA_LLM)
CARTESIA_LLM_TEMPERATURE = float(os.getenv("CARTESIA_LLM_TEMPERATURE", "0.4"))
TTS_MODEL = os.getenv("TTS_MODEL", "sonic-3")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "").strip()
CARTESIA_VOICE = os.getenv("CARTESIA_VOICE", "").strip()
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "es")
TTS_SPEED = os.getenv("TTS_SPEED", "1.0")

# ElevenLabs plugin — custom voice via ELEVEN_VOICE_ID (only when VOICE_BACKEND=elevenlabs)
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "").strip()
ELEVEN_TTS_MODEL = os.getenv("ELEVEN_TTS_MODEL", "eleven_flash_v2_5").strip()

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
    PROHIBIDO etiquetas <thinking>, razonamiento interno o metatexto.
    Tu salida es SOLO lo que el visitante escuchará en voz alta.
    PROHIBIDO inglés en voz alta. PROHIBIDO narrar tu plan («I need to call»,
    «The user is on», «first step»). Las herramientas se ejecutan en silencio.

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
    4) En intro «Así funciona»: NUNCA llames navigate_journey(advance) — el
       cliente avanza tarjetas e iconos solo. Tras hablar: PARA.
       En detail: la UI avanza secciones sola — PROHIBIDO navigate_journey(advance)
       entre secciones; en [pantalla:detail] narra desde get_session_state sin
       present_content extra (salvo primera entrada o petición explícita del visitante).
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

    ────────────────────────────────────────────────
    Screen 5 — ONBOARDING «Antes de empezar, así funciona»
    ────────────────────────────────────────────────
    UNA sola locución continua para las 3 tarjetas (0→1→2) — sin parar entre ellas.
    El cliente cambia tarjeta e iconos mientras hablas; tú no llamas tools por icono.
    ORDEN:
      (A) present_content(intro_step, 0) UNA vez al inicio
      (B) UNA locución fluida:
        — Cómo interactuar: 2-3 frases (gestos, toque, voz)
        — sin pausa: «Estas son las cinco dimensiones…» → Autoridad+línea → SSI (decir «S S I» claro) → Mensaje
          → Influencia → Higiene (orden estricto 1→5, ~6–7s cada una, no lista)
        — sin pausa: «Te recibirás…» → Radar → Informe → Correo
        — «¿Empezamos el análisis?»
      (C) PARA.
    PROHIBIDO parar entre tarjetas. PROHIBIDO present_content extra por tarjeta/icono.
    PROHIBIDO navigate_journey(advance) en intro. Solo start_analysis tras confirmación.
    NO pases al carrusel spider (intro_dimension).

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
    La UI NO cambia de pantalla: avanza sola tras cada narración
    (evidencia → brechas → tácticas).
    Primera entrada a detalle: present_content(detail_section, section=strengths)
    UNA vez → narra desde facts (PROHIBIDO decir «Fortalezas» como rótulo).
    En cada [pantalla:detail] posterior: get_session_state ÚNICAMENTE —
    PROHIBIDO present_content (la UI ya enfocó la sección).
    PROHIBIDO navigate_journey(advance) entre secciones.
    PROHIBIDO enumerar ítems o leer facts.items literal.
    PROHIBIDO decir «Oportunidades» o «Plan de acción» como encabezados.
    2–3 oraciones pausadas por sección; cita hallazgos concretos del análisis.
    SÍ usa present_content(detail_section) si el visitante pide una sección
    concreta por nombre.
    En action_plan (facts.isLastSection=true): invita Volver, otra dimensión
    o el informe — espera elección.

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

# Cartesia pipeline is slow — UI may advance (e.g. welcome preparing→ready) before
# the first TTS chunk. Force a fresh get_session_state before every reply.
_RESTATE_BEFORE_SPEAK = (
    "ANTES de hablar: llama get_session_state de nuevo — la UI pudo avanzar "
    "mientras generabas. Narra SOLO el step/phase/facts actuales; PROHIBIDO "
    "guión de una pantalla anterior."
)

_ON_ENTER_CARTESIA_EXTRA = (
    "PIPELINE HALF-CASCADE — la UI avanza sola y más rápido que tú. "
    "Habla SOLO español al visitante; cero inglés; cero planes en voz alta. "
    "Herramientas en silencio. "
    "Si welcome phase=ready: saludo con facts.name, luego navigate_journey(start_experience). "
    "Si intro y dicen empecemos/sí/adelante: navigate_journey(start_analysis) al instante."
)

_HALF_CASCADE_ON_ENTER = (
    "Conectaste al kiosk. Consulta el estado UI (herramienta, en silencio). "
    "Luego habla SOLO en español al visitante según step/phase/facts — "
    "sin inglés, sin narrar tu plan, sin nombrar herramientas. "
    "welcome preparing: locución cálida de identificación. "
    "welcome ready: saluda con facts.name (~10 s), luego navigate_journey(start_experience). "
    "PROHIBIDO present_content en welcome ready. "
    "Tras start_experience: silencio hasta [pantalla:intro]. "
    "intro: present_content(intro_step,0) y tour continuo. "
    "Si confirman análisis: navigate_journey(start_analysis)."
)

_WELCOME_READY_PANTALLA = (
    "PANTALLA welcome:ready — perfil visible en pantalla. "
    f"{_RESTATE_BEFORE_SPEAK} "
    "Saludo CORTO con facts.name, rol y empresa (~10-12 s). "
    "PROHIBIDO texto de cargando/identificando/preparing. "
    "Tras hablar: navigate_journey(start_experience). "
    "PROHIBIDO present_content en welcome:ready."
)

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


def _pantalla_welcome_ready(cue: str) -> bool:
    return ":welcome:ready" in cue or (
        "step=welcome" in cue and "phase=ready" in cue
    )


def _pantalla_welcome_preparing(cue: str) -> bool:
    return ":welcome:preparing" in cue or (
        "step=welcome" in cue and "phase=preparing" in cue
    )


def _dispatch_pantalla_reply(agent_session: AgentSession, cue: str) -> None:
    """Route a [pantalla:] cue to generate_reply (never as spoken user_input)."""
    logger.info("[text_input] pantalla cue (not spoken): %.120s", cue)
    intro_continuous = "INTRO_CONTINUOUS_TOUR" in cue
    detail_auto = "step=detail" in cue or "focus=detail" in cue
    if not intro_continuous:
        agent_session.interrupt()
    if intro_continuous:
        agent_session.generate_reply(
            instructions=(
                f"{_RESTATE_BEFORE_SPEAK} "
                "INTRO_CONTINUOUS_TOUR — UNA locución para las 3 tarjetas. "
                "Si ya llamaste present_content(intro_step, 0): continúa sin pausa "
                "dimensiones → entregables → «¿Empezamos?». "
                "Si aún no empezaste: get_session_state → present_content(intro_step, 0) "
                "→ gestos/voz → dimensiones una a una → entregables. "
                "NO pares entre tarjetas. PROHIBIDO present_content extra."
            ),
        )
    elif _pantalla_welcome_ready(cue):
        agent_session.generate_reply(instructions=_WELCOME_READY_PANTALLA)
    elif detail_auto:
        agent_session.generate_reply(
            instructions=(
                f"{_RESTATE_BEFORE_SPEAK} "
                "DETAIL_SECTION_AUTO — la UI ya avanzó a la sección activa. "
                "Llama get_session_state ÚNICAMENTE (PROHIBIDO present_content salvo "
                "que sea la primera entrada a detalle o el visitante pidió otra sección). "
                "Compón 2–3 oraciones desde facts.hint para audiencia C-level. "
                "PROHIBIDO decir Fortalezas/Oportunidades/Plan de acción como rótulos. "
                "Ancla en facts.evidence, facts.gaps o facts.tactics del informe real. "
                "PROHIBIDO navigate_journey(advance). Ritmo pausado, sin enumerar."
            ),
        )
    else:
        agent_session.generate_reply(
            instructions=(
                f"{_RESTATE_BEFORE_SPEAK} "
                "Cambio de foco en pantalla. "
                "Llama get_session_state, luego present_content si hace falta. "
                "PROHIBIDO UI/pantalla/tarjeta meta."
            ),
        )


class _ThinkingStripper:
    """Streaming filter for <thinking>…</thinking> (Nova Pro/Lite on Bedrock)."""

    def __init__(self) -> None:
        self._buf = ""
        self._in_thinking = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        tag_open = "<thinking>"
        tag_close = "</thinking>"
        while self._buf:
            lower = self._buf.lower()
            if self._in_thinking:
                close_at = lower.find(tag_close)
                if close_at < 0:
                    self._buf = ""
                    break
                self._buf = self._buf[close_at + len(tag_close) :]
                self._in_thinking = False
                continue
            open_at = lower.find(tag_open)
            if open_at < 0:
                hold = self._buf.rfind("<")
                if hold >= 0 and len(self._buf) - hold < len(tag_open) + 2:
                    out.append(self._buf[:hold])
                    self._buf = self._buf[hold:]
                    break
                out.append(self._buf)
                self._buf = ""
                break
            out.append(self._buf[:open_at])
            self._buf = self._buf[open_at + len(tag_open) :]
            self._in_thinking = True
        return "".join(out)


def _strip_thinking_text(text: str) -> str:
    return _ThinkingStripper().feed(text)


_META_TOOL_RE = re.compile(
    r"\b(get_session_state|navigate_journey|present_content|fill_search)\b",
    re.I,
)
_META_PLAN_EN_RE = re.compile(
    r"\b(the user (?:is on|has)|i need to call|i will call|the first step|"
    r"then, i will|based on the|which means their)\b",
    re.I,
)
_SCREEN_ID_RE = re.compile(r"\bwelcome:(?:ready|preparing)\b", re.I)


def _looks_like_spanish_visitor(text: str) -> bool:
    lower = text.lower().strip()
    if re.search(r"[áéíóúñ¿¡]", lower):
        return True
    spanish_hints = (
        "bienvenido",
        "hola",
        "perfecto",
        "explor",
        "dimens",
        "análisis",
        "empecemos",
        "huella",
        "presencia",
        "¿",
        "visitante",
    )
    return any(h in lower for h in spanish_hints)


def _is_meta_planning_speech(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    has_meta = bool(
        _META_TOOL_RE.search(stripped)
        or _META_PLAN_EN_RE.search(stripped)
        or _SCREEN_ID_RE.search(stripped)
        or "availableactions" in lower
        or "ui step=" in lower
        or "[pantalla:" in lower
    )
    if not has_meta:
        return False
    return not _looks_like_spanish_visitor(stripped)


def _strip_speakable_text(text: str) -> str:
    """Drop thinking tags and English tool-planning before TTS."""
    cleaned = _strip_thinking_text(text)
    if not cleaned.strip():
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    kept = [p.strip() for p in parts if p.strip() and not _is_meta_planning_speech(p)]
    return " ".join(kept).strip()


async def _filter_speech_stream(
    text: AsyncIterable[str],
) -> AsyncGenerator[str, None]:
    """Strip meta-planning from LLM text before TTS or room transcription."""
    stripper = _ThinkingStripper()
    buf = ""

    async for chunk in text:
        raw = chunk if isinstance(chunk, str) else str(chunk)
        buf += stripper.feed(raw)
        while True:
            m = re.search(r"(?<=[.!?])\s+", buf)
            if not m:
                break
            sentence = buf[: m.start() + 1].strip()
            buf = buf[m.end() :]
            cleaned = _strip_speakable_text(sentence)
            if cleaned:
                yield cleaned + " "

    tail = buf.strip()
    if tail:
        cleaned = _strip_speakable_text(tail)
        if cleaned:
            yield cleaned


_ANALYSIS_CONFIRM_RE = re.compile(
    r"\b(empecemos|empezamos|empezar|sí|si|adelante|dale|listo|vamos|"
    r"iniciar|iniciemos|análisis|analisis|comenzar)\b",
    re.I,
)


async def _maybe_start_analysis_from_voice(transcript: str) -> None:
    """Half-cascade fallback: Nova Pro often narrates without calling tools."""
    if not _ANALYSIS_CONFIRM_RE.search(transcript):
        return
    try:
        raw = await rpc("get_session_state", retries=1)
        state = json.loads(raw)
    except Exception as exc:
        logger.debug("[voice-hint] get_session_state skipped: %s", exc)
        return
    if state.get("step") != "intro":
        return
    if int(state.get("introCardIndex", 0)) < 2:
        return
    actions = state.get("availableActions") or []
    if "start_analysis" not in actions:
        return
    try:
        result = await rpc(
            "navigate_journey", {"action": "start_analysis"}, retries=1
        )
        logger.info(
            "[voice-hint] start_analysis from voice (%r): %s",
            transcript,
            result[:120],
        )
    except Exception as exc:
        logger.warning("[voice-hint] start_analysis failed: %s", exc)


class Assistant(Agent):
    """Nova Sonic host: fixed tools, UI-first spokenContent contract.

    Per LiveKit Nova Sonic guide: register @function_tool on the Agent;
    put tool_choice on RealtimeModel (not per-reply). Warm greeting via
    on_enter + generate_reply (Nova Sonic 2 mixed modalities).
    """

    def __init__(
        self,
        on_enter_done: asyncio.Event | None = None,
        *,
        cartesia_pipeline: bool = False,
    ) -> None:
        super().__init__(
            instructions=(
                f"{NOVA_INSTRUCTIONS}\n{_ON_ENTER_CARTESIA_EXTRA}"
                if cartesia_pipeline
                else NOVA_INSTRUCTIONS
            )
        )
        self._on_enter_done = on_enter_done
        self._cartesia_pipeline = cartesia_pipeline

    def transcription_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ):
        if not self._cartesia_pipeline:
            return VoiceAgent.default.transcription_node(self, text, model_settings)
        return self._cartesia_transcription_node(text, model_settings)

    def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ):
        if not self._cartesia_pipeline:
            return VoiceAgent.default.tts_node(self, text, model_settings)
        return VoiceAgent.default.tts_node(
            self, _filter_speech_stream(text), model_settings
        )

    async def _cartesia_transcription_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[str, None]:
        async for out in VoiceAgent.default.transcription_node(
            self, _filter_speech_stream(text), model_settings
        ):
            if isinstance(out, str):
                cleaned = _strip_speakable_text(out)
                if cleaned:
                    yield cleaned
            else:
                yield out

    async def on_enter(self) -> None:
        # Voice connects mid-journey (camera detect → identifying). Read the
        # live UI step first — never assume attract.
        #
        # Cartesia STT/LLM/TTS is slower than Nova realtime; the browser often
        # joins a few seconds after the agent. RPC without a participant forces
        # soft-fail state and breaks the voice-driven UI sync.
        await wait_for_kiosk_participant()

        # Do NOT pass tools= here. All four @function_tool methods on this
        # class are injected into the initial Bedrock session schema by the
        # SDK automatically. Passing tools= overrides that injection and
        # causes the AWS plugin to see fill_search as a mid-session addition,
        # triggering a full Bedrock stream recycle (~2 s silence penalty) the
        # first time identify_gate appears.
        on_enter_instructions = (
            _HALF_CASCADE_ON_ENTER
            if self._cartesia_pipeline
            else (
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
                "facts.name, menciona rol + empresa, entrega el saludo corto "
                "(~10-12 s). SOLO DESPUÉS de terminar de hablar llama "
                "navigate_journey(start_experience). "
                "PROHIBIDO llamar navigate_journey ANTES de hablar. "
                "PROHIBIDO llamar present_content en welcome:ready. "
                "Tras start_experience ok: SILENCIO — no repitas nombre ni "
                "saludo; espera [pantalla:intro]. "
                "   • identify_gate / identify_search: explica que debe "
                "identificarse; manilla o buscar por nombre. En identify_search "
                "pide el nombre en voz alta, llama fill_search(query=<nombre>) "
                "en cuanto lo diga y di solo 'Buscando…'. La UI auto-selecciona "
                "el resultado y avanza sola — NO llames navigate_journey. "
                "   • intro / analysis / detail / closing: sigue el flujo "
                "normal de esa pantalla. "
                "3) Compón desde facts.hint — no leas spokenContent literal. "
                "PROHIBIDO abrir con «Hola» o diminutivos."
            )
        )
        if self._cartesia_pipeline:
            on_enter_instructions = (
                f"{on_enter_instructions} {_ON_ENTER_CARTESIA_EXTRA}"
            )
        try:
            handle = self.session.generate_reply(instructions=on_enter_instructions)
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
        self, context: RunContext, action: str, dimension_id: str = ""
    ) -> str:
        """Ejecuta una acción disponible en la experiencia. En analysis:complete
        usa open_detail (+ dimension_id) para abrir detalle de dimensión; advance
        no está disponible ahí."""
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


def _build_cartesia_tts() -> cartesia.TTS:
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
        word_timestamps=True,
    )


def _build_elevenlabs_tts() -> elevenlabs.TTS:
    """ElevenLabs custom / cloned voices via voice_id."""
    if not ELEVEN_API_KEY:
        raise RuntimeError(
            "ELEVEN_API_KEY is required when VOICE_BACKEND=elevenlabs. "
            "Create one at https://elevenlabs.io/app/settings/api-keys"
        )
    if not ELEVEN_VOICE_ID:
        raise RuntimeError(
            "ELEVEN_VOICE_ID is required when VOICE_BACKEND=elevenlabs. "
            "Copy the voice ID from ElevenLabs Voice Library or your PVC."
        )
    model = ELEVEN_TTS_MODEL.removeprefix("elevenlabs/").strip() or "eleven_flash_v2_5"
    logger.info(
        "ElevenLabs TTS model=%s voice=%s… lang=%s key_set=%s",
        model,
        ELEVEN_VOICE_ID[:8],
        TTS_LANGUAGE,
        bool(ELEVEN_API_KEY),
    )
    return elevenlabs.TTS(
        model=model,
        voice_id=ELEVEN_VOICE_ID,
        language=TTS_LANGUAGE,
        api_key=ELEVEN_API_KEY,
        voice_settings=elevenlabs.VoiceSettings(
            speed=_plugin_speed(),
            stability=0.5,
            similarity_boost=0.75,
        ),
    )


def _build_half_cascade_tts() -> cartesia.TTS | elevenlabs.TTS:
    if VOICE_BACKEND == "cartesia":
        return _build_cartesia_tts()
    if VOICE_BACKEND == "elevenlabs":
        return _build_elevenlabs_tts()
    raise RuntimeError(f"No half-cascade TTS for VOICE_BACKEND={VOICE_BACKEND!r}")


def _build_cartesia_llm() -> inference.LLM | aws.LLM:
    """Cartesia pipeline LLM — Bedrock Claude default; Inference if model has '/'."""
    if "/" in LLM_MODEL:
        logger.info("Cartesia LLM via LiveKit Inference model=%s", LLM_MODEL)
        return inference.LLM(model=LLM_MODEL)
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        logger.warning(
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY missing — "
            "Bedrock LLM calls will fail until secrets are set."
        )
    logger.info(
        "Cartesia LLM via Bedrock model=%s region=%s",
        LLM_MODEL,
        AWS_REGION,
    )
    return aws.LLM(
        model=LLM_MODEL,
        region=AWS_REGION,
        temperature=CARTESIA_LLM_TEMPERATURE,
        tool_choice="auto",
    )


def _build_half_cascade_session() -> AgentSession:
    """STT via Inference; LLM via Bedrock; TTS via Cartesia or ElevenLabs plugin."""
    return AgentSession(
        stt=inference.STT(model=STT_MODEL, language=STT_LANGUAGE),
        llm=_build_cartesia_llm(),
        tts=_build_half_cascade_tts(),
        use_tts_aligned_transcript=True,
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            # Half-cascade is slower than Nova; wait longer before committing
            # partial STT ("¿Cómo" alone) and start TTS earlier once a turn ends.
            endpointing={"min_delay": 1.2, "max_delay": 4.5},
            interruption={"min_words": 2, "min_duration": 0.65},
            preemptive_generation={"preemptive_tts": True},
        ),
        max_tool_steps=8,
    )


# Backward-compatible alias
_build_cartesia_session = _build_half_cascade_session


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


server = AgentServer(
    # Keep one idle Python process pre-warmed so the next guest never waits
    # for a cold-start (~2 s). After a session ends the pool refills immediately.
    num_idle_processes=1,
)


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: JobContext):
    use_half_cascade = VOICE_BACKEND in HALF_CASCADE_BACKENDS

    if use_half_cascade:
        llm_backend = "inference" if "/" in LLM_MODEL else "bedrock"
        ctx.log_context_fields = {
            "room": ctx.room.name,
            "agent": AGENT_NAME,
            "voice_backend": VOICE_BACKEND,
            "tts_model": (
                ELEVEN_TTS_MODEL if VOICE_BACKEND == "elevenlabs" else TTS_MODEL
            ),
            "cartesia_voice": CARTESIA_VOICE if VOICE_BACKEND == "cartesia" else "",
            "elevenlabs_voice": (
                ELEVEN_VOICE_ID if VOICE_BACKEND == "elevenlabs" else ""
            ),
            "stt_model": STT_MODEL,
            "stt_language": STT_LANGUAGE,
            "llm_model": LLM_MODEL,
            "llm_backend": llm_backend,
        }
        session = _build_half_cascade_session()
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
    # Nova: suppress duplicate pantalla during on_enter. Cartesia: interrupt stale
    # on_enter when UI advances (e.g. welcome preparing→ready) — never queue stale.
    _pantalla_queue: list[str] = []
    _pantalla_handled_during_enter: str | None = None

    def _pantalla_text_input_handler(
        agent_session: AgentSession, event: room_io.TextInputEvent
    ) -> None:
        is_pantalla = event.text.startswith("[pantalla:")
        if is_pantalla and not _on_enter_done.is_set():
            if use_half_cascade:
                if _pantalla_welcome_preparing(event.text):
                    logger.info(
                        "[text_input] Ignoring preparing pantalla (on_enter active): %.80s",
                        event.text,
                    )
                    return
                logger.info(
                    "[text_input] Interrupting on_enter for pantalla: %.80s",
                    event.text,
                )
                nonlocal _pantalla_handled_during_enter
                _pantalla_handled_during_enter = event.text
                _dispatch_pantalla_reply(agent_session, event.text)
            else:
                logger.info(
                    "[text_input] Suppressing pantalla cue (on_enter still active): %.80s",
                    event.text,
                )
            return
        if is_pantalla:
            _dispatch_pantalla_reply(agent_session, event.text)
        else:
            agent_session.interrupt()
            agent_session.generate_reply(
                user_input=event.text,
                instructions=(
                    f"{_RESTATE_BEFORE_SPEAK} "
                    "Si intro y el visitante confirma análisis (empecemos / sí / "
                    "adelante / empezamos): navigate_journey(start_analysis) de "
                    "inmediato — no describas pantallas que no existen. "
                    "Si intro onboarding (Así funciona) y dice continuar / "
                    "qué sigue / adelante: continúa la MISMA locución — "
                    "dimensiones → entregables → «¿Empezamos?». "
                    "PROHIBIDO present_content extra; el cliente avanza tarjetas solo."
                ),
            )

    # ──────────────────────────────────────────────────────────────────────────

    if use_half_cascade:

        @session.on("error")
        def _on_session_error(ev) -> None:
            err = getattr(ev, "error", ev)
            logger.error(
                "session_error type=%s error=%s",
                type(err).__name__,
                err,
                exc_info=isinstance(err, BaseException),
            )

        if VOICE_BACKEND == "elevenlabs":
            logger.info(
                "Starting ElevenLabs plugin model=%s voice=%s… lang=%s",
                ELEVEN_TTS_MODEL,
                ELEVEN_VOICE_ID[:8] if ELEVEN_VOICE_ID else "",
                TTS_LANGUAGE,
            )
        else:
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
        if use_half_cascade:
            asyncio.create_task(_maybe_start_analysis_from_voice(event.transcript))

    @session.on("conversation_item_added")
    def _on_item_added(event) -> None:
        item = event.item
        role = getattr(item, "role", "?")
        text = getattr(item, "text_content", None)
        if text:
            interrupted = getattr(item, "interrupted", False)
            suffix = " [interrupted]" if interrupted else ""
            spoken = _strip_speakable_text(text) or _strip_thinking_text(text)
            if spoken.strip():
                logger.info("[%s]%s %s", role.upper(), suffix, spoken)

    async def _replay_queued_pantalla() -> None:
        await _on_enter_done.wait()
        if use_half_cascade:
            if _pantalla_handled_during_enter:
                logger.info(
                    "[text_input] Skipping pantalla replay — handled during on_enter: %.80s",
                    _pantalla_handled_during_enter,
                )
            return
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
        _dispatch_pantalla_reply(session, cue)

    replay_task = asyncio.create_task(_replay_queued_pantalla())

    room_opts = room_io.RoomOptions(
        # Brief browser reconnects should not tear down a slow Cartesia session.
        close_on_disconnect=not use_half_cascade,
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
    if use_half_cascade:
        room_opts.text_output = room_io.TextOutputOptions(
            sync_transcription=True,
            transcription_speed_factor=1.15,
        )

    await session.start(
        agent=Assistant(
            on_enter_done=_on_enter_done,
            cartesia_pipeline=use_half_cascade,
        ),
        room=ctx.room,
        room_options=room_opts,
    )

    await ctx.connect()

    await replay_task


if __name__ == "__main__":
    cli.run_app(server)
