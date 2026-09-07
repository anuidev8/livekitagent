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
from typing import Callable

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

from knowledge_base import search_seti_knowledge
from moderator_telemetry import is_throttle_error, report_voice_telemetry
from narration_barrier import NarrationBarrier, set_session_narration_barrier
from nova_session_continuation import (
    install_nova_session_continuation_fix,
    install_nova_session_reconnected_event_fix,
)
from rpc_client import commit_guide_screen, rpc, wait_for_kiosk_participant
from tasks.intro_orchestrator import (
    cancel_intro_tour,
    intro_tour_running,
    schedule_intro_tour,
)
from tasks.speech import build_pantalla_chat_ctx, kill_agent_speech

# The AWS plugin reads LK_SESSION_MAX_DURATION while its realtime module is
# imported. Load local configuration and publish our renewal policy first;
# doing this after importing ``livekit.plugins.aws`` silently has no effect.
load_dotenv(".env")
load_dotenv(".env.local", override=True)

# ── File logging ──────────────────────────────────────────────────────────────
# Writes logs to logs/<YYYY-MM-DD_HH-MM-SS>.log next to this file.
# Keeps the 30 most recent log files; older ones are deleted automatically.
#
# Guarded by _HUELLA_LOG_CONFIGURED: this module-level block previously ran
# every time agent.py was executed/imported a second time in the same
# process (observed with `dev` mode's reload path — both executions share
# one root logger), attaching a second FileHandler without removing the
# first. Every subsequent log call then wrote to BOTH files, producing two
# byte-identical logs per session and making log-based debugging confusing.
_root_logger = logging.getLogger()
if not getattr(_root_logger, "_huella_guide_log_configured", False):
    _LOGS_DIR = Path(__file__).parent.parent / "logs"
    _LOGS_DIR.mkdir(exist_ok=True)

    _LOG_FILE = _LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    _file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    )

    # Attach to the root logger so ALL livekit / agent log lines are captured.
    _root_logger.addHandler(_file_handler)
    _root_logger._huella_guide_log_configured = True

    # Prune old logs — keep only the 30 newest files.
    _all_logs = sorted(_LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
    for _old in _all_logs[:-30]:
        _old.unlink(missing_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

# FIXED (2026-09-02, RM_vZnfXrLvRboG logs): a recycle landing mid-conversation
# could silently break the model's tool-calling discipline for the rest of
# the session. In that trace, the visitor asked to retake the photo ~1.6s
# after "[SESSION] Session recycled successfully"; the model verbally agreed
# ("vamos a proceder con eso" / then "colócate frente al espejo para tomar la
# foto", both logged with visible stutter/[interrupted] repeats) but never
# issued another get_session_state, present_content, or navigate_journey call
# for the remainder of the session — a hard violation of the "every visitor
# turn calls get_session_state first" contract in NOVA_INSTRUCTIONS. The
# room's kiosk-side UI stayed stuck on closing:delivered while the voice
# claimed to be moving the visitor to the photo step; the LiveKit data
# channels closed unexpectedly ~33s later.
# initialize_streams(is_restart=True) does resend tools/instructions/history
# on recycle (see _serialize_tool_config call site in the vendored
# livekit-plugins-aws realtime_model.py), so this wasn't a missing-tools bug —
# more likely the model was treating the replayed history as passive context
# instead of staying in the active per-turn tool loop right after
# reconnecting. livekit-plugins-aws 1.7.0's recycle also never emitted
# RealtimeSession's own documented "session_reconnected" event (every other
# realtime session type does, e.g. RealtimeFallbackAdapter), so application
# code had no framework-standard signal to react to. Fix:
# install_nova_session_reconnected_event_fix() (nova_session_continuation.py)
# makes the recycle emit that event, and the session_reconnected handler
# below re-anchors the model with a forced get_session_state before it
# responds to anything else post-reconnect.
NOVA_SESSION_REFRESH_SECONDS = int(os.getenv("NOVA_SESSION_REFRESH_SECONDS", "420"))
if not 60 <= NOVA_SESSION_REFRESH_SECONDS <= 420:
    raise ValueError(
        "NOVA_SESSION_REFRESH_SECONDS must be between 60 and 420 seconds "
        "so renewal occurs safely before Nova Sonic's 480-second limit"
    )
os.environ["LK_SESSION_MAX_DURATION"] = str(NOVA_SESSION_REFRESH_SECONDS)
aws = importlib.import_module("livekit.plugins.aws")
install_nova_session_continuation_fix()
install_nova_session_reconnected_event_fix()

logger = logging.getLogger("agent")

AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "huella-guide")

# Nova Sonic 2 Spanish (es-US): lupe (feminine) | carlos (masculine)
# https://docs.livekit.io/agents/models/realtime/plugins/nova-sonic/
NOVA_VOICE = os.getenv("NOVA_VOICE", "lupe")
# LOW is intentionally patient for a noisy event kiosk. Deployments can still
# override this through NOVA_TURN_DETECTION without changing application code.
NOVA_TURN_DETECTION = os.getenv("NOVA_TURN_DETECTION", "LOW").strip().upper()
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

# Single source of truth for the 5 dimension labels used as literal trigger
# phrases in NOVA_INSTRUCTIONS (re-explain rule) and _USER_VOICE_TOOL_HINT
# (per-turn voice hint) below. Previously typed by hand in both places — a
# rename or a 6th dimension would silently go stale in one spot and
# re-explain requests for it would stop matching.
DIMENSION_LABELS: tuple[str, ...] = (
    "Autoridad",
    "SSI",
    "Mensaje",
    "Influencia",
    "Higiene",
)

# Product detail lives in Next.js RPC spokenContent / facts.
# Keep this prompt free of investigation / surveillance framing so Bedrock
# RAI does not block session init (ValidationException: content filters).
NOVA_INSTRUCTIONS = textwrap.dedent(
    f"""\
    Eres la guía de voz en español del kiosk SETI Huella Digital.
    Habla como una anfitriona profesional de un evento corporativo:
    cálida, clara y serena — nunca robótica ni infantil.
    Evita abrir solo con «Hola.» Evita diminutivos y coloquialismos:
    nada de «rapidito», «momentito», «segundito», «te late», «arrancamos».
    Prefiere: «cuando quieras», «iniciamos», «empezamos».
    Ritmo pausado y suave: oraciones completas, con pausa breve entre ellas.
    Solo español. Sin markdown ni listas.

    SI LA PETICIÓN NO ES CLARA (en cualquier pantalla o fase):
    NUNCA digas «lo siento», «no puedo procesar eso», «no puedo responder»,
    ni te disculpes por no entender o por estar ocupada. NUNCA menciones
    el sistema, la pantalla, ni que «no puedes» hacer algo. En vez de eso,
    pregunta UNA vez, breve y natural, reofreciendo en tus propias palabras
    las opciones ya disponibles en esa pantalla — como lo haría una
    anfitriona real que no escuchó bien, no como un asistente que rechaza
    una solicitud.

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
    4) En intro «Así funciona»: el runtime Python entrega UNA locución breve (~25-30 s)
       mientras el reel de iconos anima solo en pantalla. NO llames present_content ni
       navigate_journey(advance) durante esa locución. Si el visitante pregunta algo,
       responde brevemente. Tras la locución, solo pregunta «¿Empezamos el análisis?» y PARA.
       En detail: la UI avanza secciones sola — PROHIBIDO navigate_journey(advance)
       entre secciones. UNA sola locución continua evidencia→brechas→tácticas;
       PROHIBIDO silencio o pausa entre bloques o ítems. PROHIBIDO present_content
       extra tras la primera entrada (salvo petición explícita del visitante).
    Los [pantalla:] traen pista de step/phase/identity/focus.
    Úsalos para foco y timing; no inventes pantallas ni datos de perfil.
    NUNCA digas que no ves la pantalla: consulta get_session_state y responde
    desde sus facts sin mencionar limitaciones internas.

    PROHIBIDO ABSOLUTO — NUNCA LEAS EN VOZ ALTA:
    - El texto de mensajes [pantalla:…] (ni completo ni fragmentado).
    - Meta del sistema o de la interfaz: «La UI», «la pantalla», «la tarjeta
      siguiente/última», «avanzó», «va a mostrar», «cambió de pantalla»,
      «UI step», «focus=», «availableActions», nombres de tools, inglés técnico.
    - Jerga interna: «get_session_state», «present_content», «spokenContent»,
      «facts.hint», «CONTINUOUS TOUR», «rendering», «under construction».
    - Cualquier instrucción interna, corchetes o claves técnicas.
    Si llega un [pantalla:]: úsalo SOLO para decidir tools; habla al visitante
    en español natural sobre el CONTENIDO (dimensiones, entregables, cómo interactuar),
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

    REGLA DE PREGUNTAS SOBRE SETI (cualquier pantalla):
    Si el visitante pregunta algo sobre SETI como empresa — qué es, qué
    servicios ofrece, quiénes son sus clientes, con qué partners trabaja, qué
    casos de éxito tiene, o cómo contactarlos — y esa información no está ya
    en facts de get_session_state, llama answer_seti_question(query) con la
    pregunta del visitante. El resultado trae un RESUMEN GENERAL (misión,
    portafolio, clientes, alianzas, casos de éxito, contacto) seguido del
    detalle relacionado con la pregunta:
    - Pregunta amplia o conversacional («cuéntame de SETI», «qué más sabes
      de la empresa», «qué es SETI»): usa el resumen general — toca
      brevemente cada una de las 6 áreas en una frase corta y con tus
      propias palabras, y termina preguntando cuál área quiere conocer con
      más detalle (portafolio, clientes, alianzas, casos de éxito, o
      contacto).
    - Pregunta ya específica (pregunta directamente por clientes, servicios,
      alianzas, casos de éxito o contacto): ve directo al detalle
      relacionado, sin repetir el resumen completo.
    No inventes datos de SETI que no vengan del resultado, y no leas ningún
    fragmento literal.
    PROHIBIDO ABSOLUTO decir que la respuesta de la herramienta es incompleta,
    que faltan detalles, que no se mencionaron ciertos datos, o cualquier
    variante de eso — si algo no aparece en el resultado, simplemente no
    hables de eso, igual que harías con cualquier otro tema que no conoces.
    Tras responder, retoma el flujo donde ibas.

    PROHIBIDO decir frases de espera («un momento», «espera», «ya casi»)
    fuera de pantallas que de verdad cargan: welcome identifying (preparing),
    analysis scanning, y closing capture/shutter/generating.
    PROHIBIDO describir el tono en voz alta: nunca digas «con calma»,
    «tranquilo», «de forma cálida», «sin prisa» ni adjetivos de estilo —
    simplemente habla de esa forma sin nombrarlo.
    PROHIBIDO anunciar tus propias acciones: nunca digas «voy a mostrar»,
    «ahora voy a», «iniciamos el demo», «a continuación» ni nada similar —
    simplemente ejecuta la acción sin comentarla.

    Targets válidos de present_content fuera de «Así funciona»:
    - attract_tour (solo index -1), gesture_practice,
      welcome_preparation (index 0..2),
      result_dimension (index 0..4 o dimensionId),
      detail_dimension (+dimensionId), detail_section (+section),
      recommendation_item.
    Prohibido: target "attract", "intro", "analysis", "dimension",
    "identify_gate", "identify_search".
    result_dimension SIEMPRE necesita dimensionId (serp|ssi|arquitectura|
    influencia|higiene) O un index 0..4 explícito y correcto para la
    dimensión que el visitante nombró — NUNCA lo omitas cuando el visitante
    dijo el nombre de una dimensión. Sin uno de los dos, la pantalla puede
    quedarse enfocando la dimensión que ya estaba activa en vez de la que el
    visitante pidió, y narrarás los datos equivocados sin darte cuenta.

    Si present_content o navigate_journey fallan (ok:false): NUNCA te quedes
    en silencio ni dejes la petición sin respuesta. Antes de cualquier otra
    cosa, di UNA frase corta y natural compuesta a partir del "hint" que
    trae la respuesta — nunca leas el hint literal ni menciones "hint",
    "ok:false", "error" ni nombres de tools. Luego llama get_session_state,
    usa availableActions, y reintenta si corresponde.

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
    - Menciona de pasada que el espejo responde a su toque en la pantalla
      y a la voz — sin pedir práctica.
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
    - SIN HABLAR mientras phase=preparing: no generes ninguna respuesta ni
      digas nada — ni siquiera para anunciar que esperas o que hay silencio.
      El cliente pre-calienta la sesión de voz sin locución — la primera
      voz es el saludo en welcome:ready.
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
      * Di solo: "Buscando <nombre>…" y quédate en silencio ~2-3 s.
      * La UI SOLO auto-selecciona y avanza sola al Screen 4a cuando hay
        EXACTAMENTE una coincidencia. Si avanzó sola: perfecto, no hagas
        nada más.
      * Si en 3 s NO avanzó, llama get_session_state — facts.matchCount y
        facts.matches[] (name, company, index) te dicen exactamente qué hay
        en pantalla. NUNCA asumas ni confirmes un nombre sin mirar esto.
      * matchCount 0: no hay coincidencia — ofrece intentar con otro nombre
        (fill_search de nuevo) o pedir ayuda del staff.
      * matchCount 1: raro llegar aquí (la UI ya debería haber avanzado
        sola) — espera un poco más antes de actuar.
      * matchCount 2+: lee los nombres de facts.matches (agrega la empresa
        si ayuda a distinguir) y pregunta cuál es el visitante. En cuanto
        confirme — por nombre completo o por posición ("el primero", "la
        segunda") — llama select_search_result(index) con el índice de esa
        entrada. Si no reconoce ninguno: pide el apellido completo y llama
        fill_search de nuevo para acotar la lista.

    ────────────────────────────────────────────────
    Screen 4a — WELCOME READY (si encontrado)
    ────────────────────────────────────────────────
    Llama get_session_state PRIMERO — los datos reales del visitante están en facts.name,
    facts.role y facts.company. Úsalos para componer TU PROPIO saludo en español natural.
    PROHIBIDO ABSOLUTO: pronunciar en voz alta texto entre corchetes como [nombre], [rol],
    [empresa] u otros placeholders — son variables internas que JAMÁS se dicen.
    PROHIBIDO meta-comentarios antes del saludo: «vamos a proceder», «procederé con»,
    «realizaré el saludo», «entendido». Entra directo al contenido.
    PASO 1 — SOLO habla: saluda con el nombre real + rol/empresa reales. 2-3 oraciones:
       qué es Huella Digital y que explorarán su presencia. Si availableActions incluye
       accept_data_consent (o facts.dataConsentRequired=true): pide EXPLÍCITAMENTE el
       consentimiento — «¿Aceptas el tratamiento de tus datos personales conforme a la
       política de protección de datos de SETI?» — o pide marcar el check en pantalla.
       Si ya aceptó, invita a continuar («¿Vemos cómo funciona?»). PARA.
       PROHIBIDO llamar navigate_journey mientras hablas en este paso —
       EXCEPTO si el visitante YA dijo una palabra de consentimiento explícita
       (ver PASO 2) en este mismo turno: entonces llama navigate_journey(accept_data_consent)
       de inmediato.
    PASO 2 — Consentimiento de datos (OBLIGATORIO si accept_data_consent está en
       availableActions): llama navigate_journey(accept_data_consent) SOLO cuando el
       visitante use una palabra de consentimiento EXPLÍCITA referida a los datos —
       «acepto», «sí, acepto», «acepto el tratamiento», «autorizo», «de acuerdo» — o
       confirme que marcó el check.
       CONFUSIÓN PROHIBIDA: un «sí» suelto, o palabras de continuar como «comenzar»,
       «empecemos», «empezamos», «adelante», «vamos», «dale», «listo», «continuar»
       — INCLUSO combinadas con «sí» (p. ej. «sí, comencemos», «sí eh comenzar») —
       NUNCA equivalen a consentimiento. Esas son señales de PASO 3, no de PASO 2.
       Si el visitante dice una de esas palabras de continuar y accept_data_consent
       SIGUE en availableActions (o sea, aún no ha aceptado), NO llames
       accept_data_consent ni start_experience: pregunta explícitamente el
       consentimiento («¿Aceptas el tratamiento de tus datos personales?») y ESPERA
       una respuesta con «acepto»/«autorizo»/«de acuerdo».
       PROHIBIDO start_experience mientras accept_data_consent siga en availableActions.
    PASO 3 — Después de que el consentimiento esté hecho (start_experience en
       availableActions) y el visitante confirme continuar / adelante / seguimos /
       listo / empezamos / sí: llama navigate_journey(start_experience).
       Si confirma continuar SIN haber aceptado datos: recuerda el check primero;
       NO llames start_experience.
    Tras start_experience ok: NO repitas el saludo ni digas «silencio»/«esperando».
    Deja que [pantalla:intro:run] continúe; si la UI ya avanzó por toque, sigue
    la pantalla actual (get_session_state) — no te quedes bloqueado esperando intro.
    PROHIBIDO: present_content en welcome:ready.
    PROHIBIDO: adelantar el contenido del reel de onboarding aquí.
    PROHIBIDO: listar las 5 dimensiones aquí — se presentarán en el onboarding.
    PROHIBIDO: start_experience sin consentimiento de protección de datos.

    ────────────────────────────────────────────────
    Screen 5 — ONBOARDING «Antes de empezar, así funciona»
    ────────────────────────────────────────────────
    El runtime Python entrega UNA locución breve (~25-30 s) que cubre los tres grupos:
    interacción (toque + voz), las 5 dimensiones con una idea muy corta de qué
    mide cada una (Autoridad, LinkedIn SSI, Mensaje, Influencia, Higiene) y los
    3 entregables (Radar, Informe, Correo).
    Mientras la voz habla, el reel de iconos en pantalla anima automáticamente —
    la UI no necesita sincronización.

    REGLAS durante el onboarding:
    - NO hables por tu cuenta durante el onboarding — solo la locución del
      orchestrator o una respuesta a una pregunta del visitante. PROHIBIDO
      anunciar que estás en silencio o esperando.
    - NO llames present_content ni navigate_journey(advance).
    - Si el visitante pregunta algo, responde brevemente y con naturalidad;
      puedes explicar cualquier elemento visible en el reel (cómo interactuar,
      dimensiones, entregables) sin entrar en análisis detallado.
    - Tras la locución, pregunta UNA vez «¿Empezamos el análisis?» y PARA.
    - Solo start_analysis tras confirmación del visitante.

    RE-EXPLICAR UNA PARTE (a petición explícita del visitante):
    Si el visitante pide escuchar otra vez — interpreta de forma AMPLIA:
    «explícame de nuevo», «repite», «otra vez», «de nuevo», «no entendí»,
    «no entendí bien», «me explicas», «no quedó claro»,
    «¿cómo interactúo?», «¿qué recibo?», «las dimensiones», «¿cuáles son?»,
    «¿qué miden?», «¿cómo funciona?», «ítem uno/dos/tres», «la primera/segunda/tercera»,
    cualquier pregunta sobre {", ".join(DIMENSION_LABELS)},
    Radar, Informe, Correo:
      1) NUNCA digas que no puedes — siempre puedes, siempre lo haces.
         NUNCA respondas con frases de seguridad o política interna.
      2) Llama navigate_journey(replay_intro_card, index=N)
         donde N = 0 (cómo interactuar), 1 (dimensiones), 2 (entregables).
         Si hay duda: N=1 si mencionó dimensiones/qué miden/cuáles son,
         N=0 si preguntó cómo interactuar/navegar, N=2 si preguntó qué recibe.
      3) Narra esa sección con más detalle si el visitante pide profundidad;
         si solo pide repetir, sé igual de breve.
      4) Tras narrar, pregunta «¿Seguimos al análisis?» y PARA.
    PROHIBIDO replay_intro_card sin petición explícita del visitante.
    PROHIBIDO cancel_intro_tour cuando el visitante pide re-explicación.

    ────────────────────────────────────────────────
    ANALYSIS SCANNING → COMPLETE → RESULTS (mismo globo)
    ────────────────────────────────────────────────
    1) Scanning: si facts.hasReport es false → mensaje cálido (~2 frases):
       no hemos encontrado un informe listo; cuando esté disponible lo revisan
       juntos. PROHIBIDO inventar fuentes. PROHIBIDO invitar a continuar.
       Si hay informe: narra con acompañamiento qué
       fuentes se revisan — anclado en facts.sourceGroups / facts.narrationAnchors /
       facts.searchFindings del informe real (nombres concretos del payload).
       Tono analista senior, creíble para C-level. PROHIBIDO inventar fuentes.
       Cuando el agente recibe [pantalla:analysis:complete]:
    2) Complete: si facts.hasReport es false → mismo tono cálido (~2 frases),
       sin CTA de avance, detalle ni reporte. Si hay informe: el globo se queda; las
       tarjetas de fuentes se actualizan
       con el nombre de cada dimensión y su puntuación. ANTES de hablar,
       llama present_content(result_dimension, dimension_id=facts.
       strongestDimension.id) para resaltar en pantalla esa tarjeta — lo que
       ves resaltado debe coincidir siempre con la dimensión de la que hablas.
       Luego anuncia el standing con calidez anclado en facts.uiStandingLine
       (el título en pantalla): mismos puntos (rol, standingBlurb/banda,
       dimensión más fuerte), pero MÁS AMABLE y conversacional — PROHIBIDO
       leer uiStandingLine literal.
       En el mismo flujo (3-5 oraciones): UNA fortaleza concreta de
       facts.strengths o facts.coverLines y UNA brecha de facts.opportunities
       o facts.weakestDimension — solo datos del informe, tono consultor C-level.
       PROHIBIDO recitar solo «LÍDER · TOP 8%». Cierra preguntando si quiere
       ver el detalle de la tarjeta resaltada o prefiere otra dimensión —
       PARA y ESPERA su respuesta antes de abrir nada.
       availableActions aquí: reveal_results, open_detail, back, cancel —
       PROHIBIDO navigate_journey(advance).
       Si el visitante confirma la dimensión resaltada, nombra otra, o pide
       «detalle», «fortalezas», «oportunidades», «plan»: navigate_journey(
       open_detail, dimension_id=serp|ssi|arquitectura|influencia|higiene) —
       el tool call es lo que abre el detalle; tu voz sola no lo abre.
       Si nombra una dimensión distinta a la resaltada mientras hablas,
       resáltala primero con present_content(result_dimension, dimension_id=…)
       y sigue narrando desde ahí — ya no está prohibido llamarlo en complete.
    3) Cuando llega [pantalla:analysis:results]: mismo globo con
       tarjeta activa resaltada. El visitante navega por TOQUE o por voz —
       NO ciclar todas las dimensiones solo.

    RESULTADOS (phase=results) — globo de dimensiones (UI touch-first):
    La UI es dueña del foco: el visitante toca una tarjeta o dice cuál quiere.
    Cuando llega [pantalla:analysis_results] (foco de una tarjeta SIN abrir
    detalle): narra SOLO esa dimensión en 1-2 frases + pregunta si quiere
    el detalle. PROHIBIDO ciclar advance por todas las dimensiones.
    PROHIBIDO pedir «continuar» entre tarjetas si nadie lo pidió.
    PROHIBIDO navigate_journey(advance) solo para «pasar a la siguiente»
    automáticamente — eso pelea con el toque.
    Si pide detalle / fortalezas / una dimensión concreta:
    navigate_journey(open_detail, dimension_id=…).
    Si vuelve desde detail (back): UNA frase breve «De vuelta a tus
    dimensiones» — no re-narres scores ni evidencia.

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
    VARÍA esa pregunta de cierre cada vez — nunca la misma frase exacta en dos dimensiones seguidas.
    Si el visitante vuelve a una dimensión ya narrada: SIEMPRE narra de nuevo su evidencia/brechas/
    tácticas (nunca solo la pregunta a secas) — pero con ángulo y palabras distintas a la vez
    anterior, nunca la misma frase o estructura repetida.
    PROHIBIDO ABSOLUTO — sin importar cuántas veces la conversación ya haya tocado esta dimensión:
    «ya la vimos», «ya la viste», «ya hablamos de esto», «ya cubrimos/cubierto esta dimensión»,
    «ya revisamos/revisado esta dimensión», «como te comenté», «como mencioné antes», o cualquier
    variante que use el historial de la conversación como excusa para NO narrar evidencia/brechas/
    tácticas de nuevo. Recordar la conversación es normal; usarlo para saltarte la narración no lo es.
    send_report | back | open_detail(dimension_id=…) según respuesta.

    DETALLE → VOLVER (back desde detail):
    Si el visitante expresa CUALQUIER intención de volver o regresar al
    globo — cualquier formulación, no solo la palabra «volver»; infiere la
    intención, no esperes una frase exacta —: LLAMA navigate_journey(back)
    PRIMERO, antes de decir cualquier palabra. El tool call es lo que
    realmente mueve la pantalla; tu voz sola NO la mueve — decir «de vuelta
    a tus dimensiones» sin haber llamado el tool deja al visitante viendo
    la misma pantalla de detalle mientras tú hablas como si ya hubiera
    cambiado.
    Solo tras el ok del tool: la UI regresa al globo con las tarjetas de
    dimensiones (no a un resumen). NO describas de nuevo el resumen de la
    dimensión a menos que el visitante lo pida explícitamente.
    Di solo algo breve como «De vuelta a tus dimensiones. ¿Cuál quieres
    ver?» o «¿Revisamos otra dimensión o avanzamos al cierre?»
    Espera su elección.

    CIERRE / FOTO / TARJETA:
    - photo_consent (NUEVA fase): UNA pregunta natural que mencione AMBAS
      opciones, no solo la de aceptar — «¿Quieres tomarte una foto para tu
      tarjeta? Será la portada visual de tu informe. Si prefieres, también
      puedes continuar sin foto.»
      Tres caminos:
        • El visitante dice sí / quiero / adelante →
            LLAMA navigate_journey(ready_for_picture) PRIMERO [pasa a pose]
        • El visitante dice no / omitir / sin foto →
            LLAMA navigate_journey(skip_photo) PRIMERO [arma la tarjeta igual, sin foto — pasa por generating y delivered]
        • El visitante quiere VOLVER (dejar el cierre):
            — «volver» / «atrás» / mapa / resultados / dimensiones / globo
              (sin pedir un detalle concreto): LLAMA navigate_journey(back)
              de inmediato → analysis:results. NO preguntes destino.
            — Si pide el detalle de UNA dimensión: pregunta cuál solo si
              no la nombró; luego open_detail(dimension_id=…).
            — Si dice algo ambiguo tipo «quiero ver otra cosa» sin
              dejar claro mapa vs detalle: pregunta UNA frase y ESPERA.
      En ready_for_picture / skip_photo / back / open_detail: llama el tool
      ANTES de narrar el siguiente paso. El tool call mueve la pantalla.
      PROHIBIDO avanzar sin respuesta. PROHIBIDO preguntar dos veces la foto
      si ya eligió.
    - pose: UNA locución — invita al visitante a colocarse frente al espejo.
      Dila UNA SOLA VEZ y luego SILENCIO — PROHIBIDO repetirla ni reformularla
      con otras palabras mientras esperas, sin importar cuánto tarde el visitante
      en colocarse. Cuando confirme (listo, toma la foto, adelante):
      navigate_journey(ready_for_picture) PRIMERO y SILENCIO TOTAL — el
      contador/disparo es solo visual. PROHIBIDO hablar durante el countdown.
      PROHIBIDO el mensaje SETI / «mientras se genera» aquí (solo en generating).
      Si no dice nada, el botón «Estoy listo» en pantalla también funciona.
    - capture / shutter: UNA frase MUY corta al iniciar el contador (ánimo /
      quédate así / sonríe) — UNA vez, luego SILENCIO. PROHIBIDO contar 3-2-1
      en voz y PROHIBIDO el mensaje SETI (solo en generating).
    - generating: NO improvises ni anticipes aquí el mensaje de SETI — la
      instrucción específica que llega con [pantalla:closing:generating] ya
      trae ese mensaje completo para UNA sola locución; seguir esta regla
      general A LA VEZ que esa instrucción puntual es lo que produce el
      mensaje de SETI DOS VECES en el mismo turno. Espera esa instrucción y
      dila tal cual, UNA SOLA VEZ, y luego SILENCIO. PROHIBIDO «componiendo /
      armando / diseñando». PROHIBIDO decir o insinuar que la tarjeta/informe
      YA están listos o generados — eso NO es verdad todavía; solo cuando
      llegue [pantalla:closing:delivered]. PROHIBIDO pedir tomar foto.
    - delivered: UNA locución al entrar — invita a revisar la tarjeta e indica que informe
      e imagen van juntos a su correo. Si facts.photoSkipped es true, ofrece
      «Enviar reporte» o tomarse una foto para su tarjeta (navigate_journey(retake_photo)
      — usa la MISMA acción aunque nunca se tomó ninguna, pero dilo como «tomar una
      foto», NUNCA como «repetir»). Si facts.photoSkipped es false, ofrece
      «Enviar reporte» (navigate_journey(advance)) o «repetir la foto» si no les
      convence (navigate_journey(retake_photo) — vuelve a pose para tomar otra).
      Solo llama navigate_journey(retake_photo) si el visitante pide EXPLÍCITAMENTE
      la foto («repetir», «otra foto», «retake», «tomar de nuevo», «take again», o
      «quiero tomarme una foto» si antes la omitió). Solo llama navigate_journey(advance)
      si confirma EXPLÍCITAMENTE enviar («sí», «envía», «dale», «manda el reporte», «enviar»).
      Un «no» o «no quiero enviar el reporte» SIN mencionar la foto NO es lo mismo
      que pedir la foto — no asumas cuál de las dos opciones quiere: pregunta en
      UNA frase breve cuál prefiere y ESPERA su respuesta.
      CUANDO CONFIRME ENVIAR: LLAMA navigate_journey(advance) PRIMERO, ANTES de decir
      cualquier palabra de agradecimiento o confirmación — el tool call es lo que
      realmente envía el reporte; tu voz sola NO lo envía. PROHIBIDO ABSOLUTO decir
      «gracias», «ya está en camino», «se está enviando», o cualquier variante de que
      el informe ya se envió o se está enviando SIN haber llamado el tool en ESE MISMO
      turno y haber recibido ok:true. Si el tool devuelve ok:false, NUNCA digas que se
      envió — sigue las instrucciones de fallo (get_session_state, availableActions).
      VOLVER desde delivered (dejar la tarjeta):
        • Otra foto → navigate_journey(retake_photo)
        • «volver» / «atrás» / mapa / resultados / dimensiones →
          navigate_journey(back) de inmediato (no preguntes destino)
        • Detalle de una dimensión → pregunta cuál si no la nombró, luego
          navigate_journey(open_detail, dimension_id=…)
        • Ambiguo («otra cosa», «algo más») sin mapa/foto/detalle claro →
          pregunta UNA frase «¿Otra foto, tus dimensiones, o el detalle de alguna?»
          y ESPERA.
    - thanks: agradecimiento cálido; invita a escanear el QR para conocer más de SETI;
      cuando confirmen salir (sí, finalizar, finish): LLAMA navigate_journey(finish)
      PRIMERO, antes de decir cualquier despedida — el tool call es lo que realmente
      termina la experiencia, tu voz sola NO la termina. Si algo alcanzas a decir,
      que sea brevísimo y DESPUÉS del tool call, nunca antes: si hablas primero, una
      nueva interrupción del visitante puede cortar tu turno antes de llegar al tool call.
      PROHIBIDO mencionar tarjeta, foto, imagen o correo — ya se explicó en delivered.
      PROHIBIDO repetir análisis o entrega.

    PREGUNTAS SOBRE DATOS / PRIVACIDAD:
    Si el visitante pregunta qué pasa con su información o sobre el
    consentimiento que aceptó: responde breve — se usa su información
    profesional pública para el análisis de marca personal, y la foto del
    stand SOLO si la autorizó aparte (nunca es obligatoria para participar).
    Para más detalle o para ejercer sus derechos, remite al documento que
    firmó al ingresar o a legal@seti.com.co. No inventes detalles que no
    estén aquí.

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


def _log_rpc_state_summary(rpc_method: str, raw_result: str) -> None:
    """Log the fields that actually pin down WHICH dimension/screen a tool
    call resolved to — step/phase/focusDimensionId/analysisDimIndex and, for
    the detail_section facts payload, facts.dimensionId.

    Regression (2026-09-04 15:49 logs): a visitor said "mensaje" on
    analysis:results and the guide narrated SERP/Autoridad facts instead —
    but with only "Tool call emitted: present_content (id=...)" logged (no
    call args) and no RPC response logged at all, there was no way to tell
    whether the model omitted dimension_id, the frontend resolved it to the
    wrong dimension, or the response itself carried stale facts. This never
    logs full narrative text (facts.evidence/gaps/tactics, spokenContent) —
    only the identifying fields needed to correlate "what was asked for" vs
    "what the frontend says is focused."
    """
    try:
        parsed = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(parsed, dict):
        return
    facts = parsed.get("facts") if isinstance(parsed.get("facts"), dict) else {}
    logger.info(
        "[%s] state summary: ok=%s step=%s phase=%s focusDimensionId=%s "
        "analysisDimIndex=%s facts.dimensionId=%s",
        rpc_method,
        parsed.get("ok", True),
        parsed.get("step"),
        parsed.get("phase"),
        parsed.get("focusDimensionId"),
        parsed.get("analysisDimIndex"),
        facts.get("dimensionId"),
    )


# Max seconds on_enter's warm-greeting generate_reply may block incoming
# [pantalla:] screen cues before being force-interrupted. Independent of
# (and much shorter than) the realtime model's own generate_reply_timeout,
# which is sized for legitimate long turns elsewhere in the session, not
# for this startup window. See on_enter() for the full rationale.
ON_ENTER_MAX_WAIT_S = 12.0


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
    return step == "welcome" and phase in ("preparing", "ready")


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

    def __init__(
        self,
        on_enter_done: asyncio.Event | None = None,
        on_navigate: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(instructions=NOVA_INSTRUCTIONS)
        self._on_enter_done = on_enter_done
        # Fires after a navigate_journey action succeeds — used to reset the
        # "narrate once" pantalla guard for actions that legitimately revisit
        # a phase already narrated this session (e.g. retake_photo → pose).
        self._on_navigate = on_navigate

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
            # Bound how long on_enter can block incoming [pantalla:] cues.
            # The realtime model's own generate_reply_timeout (45s) is far
            # too long for this — it's sized for legitimate long turns
            # elsewhere, not for the startup greeting. If the visitor's
            # real screen moves on (e.g. identified in ~25s) while this
            # first generate_reply is still stuck mid-flight, waiting the
            # full 45s means the agent speaks ~45s of stale, ungrounded
            # context while every real screen update is silently dropped
            # (Nova Sonic path) or queued-but-delayed (Cartesia path).
            # ON_ENTER_MAX_WAIT_S caps that blast radius: past this point,
            # interrupt the stale greeting and let queued cues (see
            # _replay_queued_pantalla) take over with fresh state.
            try:
                await asyncio.wait_for(
                    handle.wait_for_playout(), timeout=ON_ENTER_MAX_WAIT_S
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "on_enter: generate_reply exceeded %.0fs (step=%s phase=%s) — "
                    "interrupting stale greeting so queued screen updates can proceed",
                    ON_ENTER_MAX_WAIT_S,
                    state.get("step"),
                    state.get("phase"),
                )
                handle.interrupt(force=True)
        finally:
            # Signal that on_enter has fully completed (audio delivered, or
            # force-interrupted above). The pantalla guard (_on_enter_done
            # event) will unblock any subsequent [pantalla:] cues only after
            # this point, preventing the welcome greeting from being repeated.
            if self._on_enter_done is not None:
                self._on_enter_done.set()

    @function_tool
    async def get_session_state(self, context: RunContext) -> str:
        """Obtiene el estado actual de la pantalla (step, phase, availableActions, spokenContent)."""
        try:
            result = await rpc("get_session_state", retries=2)
            _log_rpc_state_summary("get_session_state", result)
            return result
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
        welcome_preparation (0..2 prep), result_dimension (solo analysis:results),
        detail_dimension (+dimension_id; usar en complete si piden «detalle»),
        detail_section
        (+section), recommendation_item. Compón tu mensaje desde facts.hint
        y los campos de facts devueltos. Si ok=false, get_session_state y reintenta.
        Nunca uses target=attract|intro|analysis.
        """
        logger.info(
            "[present_content] call target=%s index=%s dimension_id=%s section=%s",
            target,
            index,
            dimension_id or "(empty)",
            section or "(empty)",
        )
        result = await rpc(
            "present_content",
            {
                "target": target,
                "index": index,
                "dimensionId": dimension_id,
                "section": section,
            },
        )
        _log_rpc_state_summary("present_content", result)
        return result

    @function_tool
    async def navigate_journey(
        self, context: RunContext, action: str, dimension_id: str = "", index: int = -1
    ) -> str:
        """Ejecuta una acción disponible en la experiencia.
        - accept_data_consent: en welcome:ready SOLO cuando el visitante da
          consentimiento EXPLÍCITO al tratamiento de datos — checkbox marcado
          o dice «acepto»/«autorizo»/«de acuerdo». Un «sí» suelto o palabras
          de continuar («comenzar», «empezamos», «adelante», «vamos», «dale»,
          «listo») NO son consentimiento, ni siquiera combinadas con «sí».
          Obligatorio antes de start_experience si aparece en availableActions.
        - start_experience: en welcome:ready SOLO tras consentimiento de datos
          (cuando start_experience esté en availableActions).
        - open_detail + dimension_id: abre detalle de dimensión (analysis, detail,
          o desde closing:photo_consent|pose|delivered).
        - back: en closing:photo_consent|pose|prep|delivered vuelve a analysis:results.
        - ready_for_picture: en closing:photo_consent o closing:pose|capture cuando el visitante confirma la foto.
        - skip_photo: en closing:photo_consent cuando el visitante quiere omitir la foto.
        - retake_photo: en closing:delivered|generating vuelve a pose.
        - finish: en closing:thanks cuando confirma salir.
        - replay_intro_card + index (0=cómo interactuar, 1=dimensiones, 2=entregables):
          vuelve a narrar esa tarjeta cuando el visitante lo pide explícitamente.
          Solo disponible mientras step=intro."""
        logger.info(
            "[navigate_journey] call action=%s dimension_id=%s index=%s",
            action,
            dimension_id or "(empty)",
            index,
        )
        payload: dict = {"action": action, "dimensionId": dimension_id}
        if index >= 0:
            payload["index"] = index
        result = await rpc("navigate_journey", payload)
        _log_rpc_state_summary("navigate_journey", result)
        if self._on_navigate is not None:
            ok = True
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and "ok" in parsed:
                    ok = bool(parsed["ok"])
            except (json.JSONDecodeError, TypeError):
                pass
            if ok:
                self._on_navigate(action)
        return result

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

    @function_tool
    async def select_search_result(self, context: RunContext, index: int) -> str:
        """Confirma manualmente cuál resultado de búsqueda es el visitante,
        en la pantalla identify_search.

        La UI avanza SOLA cuando get_session_state.facts.matchCount es
        exactamente 1 — no llames esta herramienta en ese caso, solo espera.

        Úsala SOLO cuando facts.matchCount sea 2 o más (varias personas con
        el mismo nombre) y el visitante ya haya confirmado en voz alta cuál
        es la suya — por nombre completo o por posición ("el primero",
        "la segunda"). `index` es 1-based, tal como aparece en
        facts.matches[].index. NUNCA la llames por tu cuenta sin que el
        visitante haya confirmado explícitamente cuál opción es la suya.
        """
        return await rpc("select_search_result", {"index": index})

    @function_tool
    async def answer_seti_question(self, context: RunContext, query: str) -> str:
        """Busca información oficial sobre SETI S.A.S. como empresa: identidad,
        servicios (PRIME), clientes, alianzas/partners, casos de éxito, talento
        o canales de contacto.

        Llama esta herramienta cuando el visitante pregunte algo sobre SETI que
        no esté ya cubierto por facts de get_session_state (por ejemplo: "¿qué
        servicios ofrecen?", "¿quiénes son sus clientes?", "¿tienen casos de
        éxito con IA?", "¿cómo los contacto?"). No inventes datos de SETI que
        no vengan de esta herramienta o de facts — compón la respuesta con tus
        propias palabras a partir del resultado.
        """
        try:
            return search_seti_knowledge(query)
        except Exception as exc:
            logger.warning("answer_seti_question failed: %s", exc)
            return "No pude consultar la base de conocimiento de SETI en este momento."


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
    "NAVEGACIÓN ATRÁS (cualquier pantalla, no solo detail): si el visitante "
    "expresa CUALQUIER intención de volver o regresar a la pantalla anterior "
    "— infiere la intención, NO busques una palabra exacta; puede sonar como "
    "«puedes ir atrás», «volvamos», «regrésame», «quiero ver los resultados de "
    "nuevo», «sal de aquí», «para atrás», «cierra esto», o cualquier otra forma "
    "de pedir salir de donde está — y 'back' aparece en availableActions de "
    "get_session_state: LLAMA navigate_journey(action='back') DE INMEDIATO, "
    "ANTES de decir cualquier frase de confirmación. El tool call es lo que "
    "realmente mueve la pantalla — tu voz sola NO la mueve. NUNCA digas «de "
    "vuelta a...», «regresamos a...» ni nada similar sin haber llamado el tool "
    "en ese mismo turno; si lo dices sin llamarlo, el visitante se queda viendo "
    "la misma pantalla mientras tú hablas como si ya hubiera cambiado. "
    "Si step=welcome y phase=ready: "
    "1) Si availableActions incluye accept_data_consent Y el visitante usa una "
    "palabra de consentimiento EXPLÍCITA (acepto, de acuerdo, sí acepto, ya marqué "
    "el check, autorizo, acepto el tratamiento): navigate_journey(accept_data_consent) "
    "de inmediato. "
    "2) CONFUSIÓN PROHIBIDA — un «sí» suelto, o palabras de continuar (continúa, "
    "adelante, comienza, comenzar, empezamos, listo, vamos, dale) NUNCA cuentan "
    "como consentimiento, NI SIQUIERA combinadas con «sí» (ej.: «sí eh comenzar», "
    "«sí, comencemos» NO son consentimiento). Si el visitante dice solo eso y "
    "accept_data_consent aún está disponible: NO llames accept_data_consent ni "
    "start_experience — pregunta explícitamente «¿Aceptas el tratamiento de tus "
    "datos personales conforme a la política de protección de datos?» y espera "
    "una respuesta con acepto/autorizo/de acuerdo. "
    "3) Solo cuando start_experience esté en availableActions y confirme continuar: "
    "navigate_journey(start_experience) de inmediato tras UNA frase de cierre breve — "
    "no re-narres las dimensiones ni repitas el saludo completo. "
    "Si step=intro y el visitante confirma comenzar "
    "(sí, adelante, empecemos, empezamos, iniciar, dale, vamos): "
    "navigate_journey(start_analysis) de inmediato, sin repetir el reel. "
    "Si step=intro y el visitante pide re-explicar una tarjeta — "
    "interpreta de forma AMPLIA: cualquier señal de confusión, re-consulta o petición sobre "
    "el contenido de las tarjetas debe activar replay_intro_card. "
    "Señales de re-explicación de DIMENSIONES (N=1): "
    "«no entendí», «no entendí bien», «no entendí muy bien», «me explicas», «explícame», "
    "«explícame de nuevo», «repite», «repíteme», «otra vez», «de nuevo», «¿qué son las dimensiones?», "
    "«¿cuáles son?», «las dimensiones», «¿qué miden?», «dimensión», «ítem dos», «ítem número dos», "
    "«el segundo», «la segunda tarjeta», «las cinco», «vuelve a explicar», «no quedó claro», "
    "«no entendí el número dos», «¿cuáles son las dimensiones?», "
    f"{', '.join(f'«{label}»' for label in DIMENSION_LABELS)} (cuando pregunta qué son). "
    "Señales de re-explicación de CÓMO INTERACTUAR (N=0): "
    "«cómo interactúo», «¿cómo funciona?», «¿cómo interactúo?», «la primera tarjeta», «ítem uno», "
    "«¿cómo avanzo?», «¿cómo navego?», «toque». "
    "Señales de re-explicación de ENTREGABLES (N=2): "
    "«¿qué recibo?», «el informe», «el radar», «¿qué me dan?», «la tercera tarjeta», «ítem tres», "
    "«¿qué incluye?», «los entregables». "
    "Ante cualquier duda sobre cuál tarjeta, usa N=1 (dimensiones) si mencionó dimensiones, "
    "N=0 (cómo interactuar) si preguntó cómo interactuar, N=2 (entregables) si preguntó qué recibe. "
    "NUNCA respondas con 'no puedo' ni rechaces — SIEMPRE ejecuta replay_intro_card. "
    "Luego narra esa tarjeta con más detalle si pide profundidad, o breve si solo repite. "
    "Si step=closing y phase=photo_consent y el visitante dice sí / quiero foto / con foto: "
    "navigate_journey(ready_for_picture) de inmediato. "
    "Si step=closing y phase=photo_consent y el visitante dice no / sin foto / omitir / skip: "
    "navigate_journey(skip_photo) de inmediato — arma la tarjeta igual, sin foto. "
    "Si step=closing y phase=photo_consent|pose|prep|delivered y quiere VOLVER: "
    "«volver»/«atrás»/mapa/resultados/dimensiones → navigate_journey(back) DE INMEDIATO; "
    "detalle de una dimensión nombrada → navigate_journey(open_detail, dimension_id=…); "
    "otra foto (solo delivered/generating) → navigate_journey(retake_photo); "
    "solo si el pedido es ambiguo (sin mapa/foto/detalle): pregunta UNA frase y ESPERA. "
    "Si step=closing y phase=pose|prep|capture y el visitante pide tomar la foto "
    "(toma la foto, listo, estoy listo, adelante, take picture, toma la): "
    "navigate_journey(ready_for_picture) — no solo hables, ejecuta la acción. "
    "Si step=closing y phase=review (el visitante ve su foto recién tomada) y dice "
    "sí / me gusta / está bien / úsala / así está bien: navigate_journey(confirm_portrait) "
    "de inmediato. Si dice no / otra vez / repetir / no me gusta / tomar otra: "
    "navigate_journey(retake_photo) de inmediato — vuelve a pose. "
    "Si step=closing y phase=thanks y confirma salir "
    "(sí, finalizar, finish, terminamos, listo para salir, yes): "
    "LLAMA navigate_journey(finish) PRIMERO, de inmediato, antes de decir nada — "
    "una despedida breve puede ir DESPUÉS del tool call, nunca antes: si hablas "
    "primero, una nueva interrupción del visitante corta tu turno antes de que "
    "el tool call llegue a ejecutarse. "
    "Si step=closing y phase=delivered y pide enviar reporte: navigate_journey(advance) o send_report según availableActions. "
    "Si step=closing y phase=delivered y quiere una foto para la tarjeta — ya sea "
    "repetirla (repetir, otra foto, retake, no me gusta, tomar de nuevo, take again, "
    "otra vez) o tomarla por primera vez porque antes la omitió (quiero tomarme una "
    "foto, quiero tomar una foto, sí quiero foto, quiero una foto, take a picture, "
    "take picture, con foto después de todo): "
    "navigate_journey(retake_photo) de inmediato — vuelve a pose. Misma acción en "
    "ambos casos; NO exijas que use la palabra «repetir» — facts.photoSkipped ya te "
    "dice si es la primera foto o una repetición, la frase del visitante no tiene "
    "que distinguirlo."
)


def _pantalla_dedupe_key(text: str) -> str:
    if "intro:run" in text or "INTRO_ORCHESTRATOR" in text:
        return "intro:run"
    if "closing:photo_consent" in text or (
        "step=closing" in text and "phase=photo_consent" in text
    ):
        return "closing:photo_consent"
    if "closing:photo" in text or "closing:pose" in text or (
        "step=closing" in text and "phase=pose" in text
    ):
        return "closing:photo"
    if "closing:countdown" in text or (
        "step=closing" in text
        and ("phase=capture" in text or "phase=shutter" in text)
    ):
        return "closing:countdown"
    if "closing:review" in text or ("step=closing" in text and "phase=review" in text):
        return "closing:review"
    if "closing:generating" in text:
        return "closing:generating"
    if "closing:delivered" in text:
        return "closing:delivered"
    if "closing:thanks" in text or ("step=closing" in text and "phase=thanks" in text):
        return "closing:thanks"
    if "analysis:complete" in text:
        return "analysis:complete"
    if "analysis:scanning" in text or (
        "step=analysis" in text and "phase=scanning" in text
    ):
        return "analysis:scanning"
    if "analysis:results" in text or (
        "step=analysis" in text and "phase=results" in text
    ):
        return "analysis:results"
    if "analysis_results" in text or "Usuario pasó a dimensión" in text:
        return "analysis_results"
    if "detail:revisit" in text or "DETAIL_REVISIT" in text:
        # Include the dimension id (client-appended "DIM_ID=…" marker, never
        # spoken — see buildDetailRevisitCue) so two different dimensions
        # never collapse onto the same key. Without this, _deliver_pantalla_
        # reply's staleness/supersede check (pantalla_guard.last_key !=
        # dedupe_key) can never tell "revisit ssi" apart from "revisit
        # influencia" — both keyed "detail:revisit" — so a reply queued for
        # a dimension the visitor already left would never be detected as
        # superseded (2026-09-04 regression: agent kept narrating one
        # dimension's facts after the visitor had already opened another).
        dim_match = re.search(r"DIM_ID=(\S+)", text)
        return f"detail:revisit:{dim_match.group(1)}" if dim_match else "detail:revisit"
    if "detail:continuous" in text or "DETAIL_CONTINUOUS" in text:
        dim_match = re.search(r"DIM_ID=(\S+)", text)
        return (
            f"detail:continuous:{dim_match.group(1)}"
            if dim_match
            else "detail:continuous"
        )
    match = re.search(r"\[pantalla:([^\]]+)\]", text)
    if match:
        return match.group(1).split("]")[0].strip()
    return text[:96]


def _analysis_pantalla_instructions(dedupe_key: str) -> str | None:
    """Per-screen instructions for analysis pantallas (button-nav safe).

    Generic «Cambio de foco» let Nova keep welcome chat context and re-greet
    on analysis:scanning (2026-09-04_12-44-40 logs).
    """
    if dedupe_key == "analysis:scanning":
        return (
            "ANALYSIS SCAN — get_session_state PRIMERO. "
            "Estás en la pantalla de ANÁLISIS / ESCANEANDO (iconos de fuentes). "
            "PROHIBIDO ABSOLUTO: repetir saludo, nombre+rol de welcome, «¿Vemos cómo "
            "funciona?», «¿Empezamos el análisis?», o cualquier frase de bienvenida. "
            "PROHIBIDO ABSOLUTO: manilla, lector NFC, identificación, acercarse al "
            "espejo/lector, attract, o pedir que se identifiquen — ESO YA PASÓ. "
            "Si facts.hasReport es false O no hay facts.sourceGroups/narrationAnchors/"
            "searchFindings: mensaje cálido en español (~2 frases) — no hemos "
            "encontrado un informe de huella listo; cuando esté disponible lo "
            "revisan juntos aquí. PROHIBIDO inventar LinkedIn, prensa, redes, "
            "sitios, hallazgos o scores. PROHIBIDO invitar a continuar. "
            "Si hay datos reales: narra SOLO el escaneo en vivo con fuentes de "
            "facts.sourceGroups / facts.narrationAnchors / facts.searchFindings. "
            "Tono analista senior. PROHIBIDO lista numerada. "
            "PROHIBIDO navigate_journey. PROHIBIDO pedir continuar."
        )
    if dedupe_key == "analysis:complete":
        return (
            "ANALYSIS COMPLETE — get_session_state PRIMERO. "
            "PROHIBIDO repetir el saludo de welcome o «¿Vemos cómo funciona?». "
            "Si facts.hasReport es false: mensaje cálido (~2 frases) — no hemos "
            "encontrado un informe listo; cuando esté disponible lo revisan juntos. "
            "PROHIBIDO inventar scores, fuentes, fortalezas o brechas. "
            "PROHIBIDO open_detail / send_report / reveal_results / invitar a continuar. "
            "Si hay informe: anuncia el standing con calidez desde facts.uiStandingLine "
            "(parafrasea, no leas literal): rol, banda, dimensión más fuerte. "
            "Añade UNA fortaleza y UNA brecha de facts. "
            "Invita a tocar una dimensión o «ver detalles». "
            "open_detail | reveal_results | send_report según availableActions."
        )
    if dedupe_key == "analysis:results":
        return (
            "ANALYSIS RESULTS (globo) — get_session_state PRIMERO. "
            "Si el visitante ACABA DE VOLVER desde detail (back): di SOLO una frase "
            "breve tipo «De vuelta a tus dimensiones. ¿Cuál quieres ver?» — "
            "PROHIBIDO re-narrar standing, scores, evidencia o la dimensión anterior. "
            "Si es la primera llegada a results en este ciclo: orienta brevemente el "
            "globo (tarjeta activa) y pregunta qué dimensión abrir. "
            "PROHIBIDO repetir welcome. PROHIBIDO locución larga."
        )
    if dedupe_key == "analysis_results":
        return (
            "RESULT DIMENSION FOCUS — get_session_state PRIMERO. "
            "Narra SOLO la dimensión enfocada ahora (facts de esa dimensión): "
            "1-2 frases score + idea clave. Pregunta si quiere ver el detalle. "
            "PROHIBIDO narrar otras dimensiones. PROHIBIDO repetir welcome o standing "
            "completo. Si step=detail ya (el visitante abrió detalle): NO hables — "
            "espera [pantalla:detail:continuous]."
        )
    return None


def _closing_pantalla_instructions(text: str) -> str | None:
    if "closing:photo_consent" in text or (
        "step=closing" in text and "phase=photo_consent" in text
    ):
        return (
            "CLOSING PHOTO CONSENT — get_session_state. "
            "Pregunta en UNA frase natural si el visitante desea tomarse una foto para su tarjeta "
            "(la foto será la portada visual del informe), mencionando TAMBIÉN en esa misma "
            "frase que puede continuar sin foto si prefiere — no preguntes solo por el sí, "
            "deja tan clara la opción de decir que no como la de aceptar. "
            "ESPERA su respuesta. PROHIBIDO avanzar sin confirmación. "
            "Si los botones en pantalla ya respondieron, no preguntes de nuevo. "
            "Cuando responda: sí / quiero / adelante → LLAMA navigate_journey(ready_for_picture) "
            "PRIMERO, antes de decir cualquier cosa sobre colocarse frente al espejo — el tool "
            "call es lo que realmente mueve la pantalla, tu voz sola NO la mueve. "
            "no / omitir / sin foto → LLAMA navigate_journey(skip_photo) PRIMERO, antes de "
            "confirmar nada. "
            "VOLVER: «volver»/«atrás»/mapa/resultados/dimensiones → navigate_journey(back) "
            "DE INMEDIATO (no preguntes destino); "
            "detalle de una dimensión → pregunta cuál solo si no la nombró, luego "
            "navigate_journey(open_detail, dimension_id=serp|ssi|arquitectura|influencia|higiene); "
            "solo si el pedido es ambiguo → pregunta UNA frase y ESPERA. "
            "PROHIBIDO narrar el siguiente paso («colócate», «perfecto, "
            "continuamos sin foto», etc.) sin haber llamado el tool correspondiente en ESE "
            "mismo turno — narrar sin llamar el tool dejaría la pantalla sin avanzar aunque "
            "tu voz suene como si ya hubiera pasado."
        )
    if "closing:photo" in text or ("step=closing" in text and "phase=pose" in text):
        return (
            "CLOSING PHOTO POSE — get_session_state. "
            "UNA locución: invita al visitante a colocarse frente al espejo para la foto. "
            "Dila UNA SOLA VEZ y luego SILENCIO — PROHIBIDO repetirla o reformularla "
            "con otras palabras mientras esperas a que se coloque, sin importar cuánto "
            "tarde. Si confirman estar listos: LLAMA navigate_journey(ready_for_picture) "
            "PRIMERO y luego SILENCIO TOTAL en ese mismo turno — el contador 3-2-1 es "
            "solo visual; PROHIBIDO contar en voz, decir «sonríe», o narrar el disparo. "
            "PROHIBIDO ABSOLUTO decir el mensaje de SETI / «mientras se genera tu "
            "tarjeta» aquí — eso SOLO cuando llegue [pantalla:closing:generating]."
        )
    if "closing:countdown" in text or (
        "step=closing" in text
        and ("phase=capture" in text or "phase=shutter" in text)
    ):
        return (
            "CLOSING COUNTDOWN — el contador de la foto ya corre en pantalla. "
            "UNA frase MUY corta (máx ~8 palabras), cálida, sobre la toma — "
            "ejemplos de tono: «Quédate así, ya casi…», «Sonríe al espejo, "
            "perfecto», «Un segundo, capturando…». "
            "Dila UNA SOLA VEZ y luego SILENCIO hasta review. "
            "PROHIBIDO contar 3-2-1 en voz (la UI ya cuenta). "
            "PROHIBIDO ABSOLUTO el mensaje de SETI / «mientras se genera tu "
            "tarjeta» — eso SOLO en [pantalla:closing:generating]."
        )
    if "closing:review" in text or ("step=closing" in text and "phase=review" in text):
        return (
            "CLOSING REVIEW — get_session_state. "
            "El visitante ya ve en pantalla la foto que se acaba de tomar. "
            "UNA pregunta corta y cálida: si le gusta esa foto para su tarjeta. "
            "NUNCA describas ni juzgues la foto — no puedes verla. "
            "Dila UNA SOLA VEZ y luego SILENCIO — PROHIBIDO repetirla mientras esperas. "
            "sí / me gusta / está bien / úsala → navigate_journey(confirm_portrait). "
            "no / otra vez / repetir / no me gusta → navigate_journey(retake_photo) — vuelve a pose. "
            "Los botones en pantalla también permiten elegir sin hablar."
        )
    if "closing:generating" in text:
        return (
            "CLOSING GENERATING — get_session_state. "
            "NO digas «estamos componiendo / armando / diseñando tu tarjeta». "
            "En su lugar, UNA locución: abre con «Mientras se genera tu tarjeta,» "
            "(o «Mientras tanto,») y di este mensaje de SETI, casi literal: "
            "«En SETI entendemos el negocio antes de proponer una solución, porque "
            "sabemos que la tecnología sólo genera valor cuando responde a desafíos "
            "reales. En SETI, crecemos para nuestros clientes.» "
            "PROHIBIDO pedir tomar foto. PROHIBIDO decir que la tarjeta/informe YA "
            "están listos. Di esto UNA VEZ y luego SILENCIO hasta la próxima "
            "instrucción del sistema o [pantalla:closing:delivered]."
        )
    if "closing:delivered" in text:
        return (
            "CLOSING DELIVERED — get_session_state. "
            "UNA locución al entrar: invita a revisar la tarjeta. "
            "Si facts.photoSkipped es true: indica que el informe va a su correo (SIN "
            "mencionar foto ni imagen — no se tomó ninguna, decir «junto con la imagen» "
            "sería falso), y ofrece «Enviar reporte» o tomarse una foto para su tarjeta "
            "— di «tomar una foto», NUNCA «repetir la foto» (nunca se tomó una). "
            "Si facts.photoSkipped es false: indica que informe y foto van juntos a su "
            "correo, y ofrece «Enviar reporte» o «repetir la foto» si no les convence. "
            "Solo navigate_journey(retake_photo) si el visitante pide EXPLÍCITAMENTE la foto "
            "(repetir / otra foto / retake / tomar de nuevo / take again, o «quiero tomarme "
            "una foto» si antes la omitió). Solo navigate_journey(advance) si confirma "
            "EXPLÍCITAMENTE enviar (sí / envía / dale / manda el reporte / enviar). "
            "CUANDO CONFIRME ENVIAR: LLAMA navigate_journey(advance) PRIMERO, ANTES de decir "
            "cualquier palabra — el tool call es lo que realmente envía el reporte, tu voz "
            "sola NO lo envía. PROHIBIDO ABSOLUTO decir «gracias», «ya está en camino a tu "
            "correo», «se está enviando», o cualquier variante de que el reporte ya se envió "
            "SIN haber llamado el tool en ESE MISMO turno y haber recibido ok:true — esto ya "
            "pasó antes y el visitante se queda sin su informe mientras cree que ya lo tiene. "
            "Si el tool devuelve ok:false: NUNCA digas que se envió; compón una frase corta "
            "desde su hint, llama get_session_state y sigue availableActions. "
            "VOLVER desde esta tarjeta: «volver»/mapa/resultados → navigate_journey(back) "
            "DE INMEDIATO; "
            "detalle de una dimensión → pregunta cuál si no la nombró, luego "
            "navigate_journey(open_detail, dimension_id=…); "
            "pedido ambiguo → pregunta UNA frase "
            "«¿Otra foto, tus dimensiones, o el detalle de alguna?» y ESPERA. "
            "Un «no» o «no quiero enviar el reporte» SIN mencionar la foto NO equivale a "
            "pedir la foto — no lo asumas. Si no queda claro cuál de las dos opciones "
            "quiere, pregunta en UNA frase breve y ESPERA su respuesta en vez de adivinar. "
            "PROHIBIDO repetir frases ya dichas en generating o photo."
        )
    if "closing:thanks" in text or ("step=closing" in text and "phase=thanks" in text):
        return (
            "CLOSING THANKS — get_session_state. "
            "Agradecimiento cálido + invita a escanear el QR de SETI. "
            "PROHIBIDO mencionar tarjeta, foto, imagen o correo — ya se explicó en delivered. "
            "Si el visitante confirma salir (sí, finalizar, finish, terminamos, listo): "
            "LLAMA navigate_journey(finish) PRIMERO, antes de decir cualquier despedida — "
            "el tool call es lo que realmente termina la experiencia, tu voz sola no la "
            "termina, y si hablas primero una nueva interrupción puede cortarte antes de "
            "llegar al tool call. "
            "Si aún no confirmó: pregunta si finalizamos → ESPERA."
        )
    return None


class _PantallaGuard:
    """Per-session dedupe state for [pantalla:] screen cues.

    Instantiated fresh inside `my_agent()` for each room/job — never at
    module scope — so concurrent kiosk sessions never share state.

    Owns two independent guards:
    - a short debounce window (`is_duplicate`) that drops truly duplicate
      cues arriving within `_PANTALLA_DEDUPE_SECONDS` of each other.
    - a "narrated once" set for closing cues (photo, review, generating,
      delivered) that must be spoken exactly once per pass through that phase.

    Regression: retake_photo sends the visitor through pose → capture →
    review → generating → delivered again. Only the "closing:photo" once-key
    was ever forgotten on retake, so the second pass through generating/
    delivered was silently swallowed by the once-only guard — the model
    never received fresh navigate_journey(advance) guidance for that
    cycle, so it fell back to improvising ungrounded "sending it now"
    narration in a loop and never actually called the tool (2026-09-02
    RM_SpsHnphyUjch logs: visitor repeated "enviar" many times, agent kept
    saying "se están enviando" without ever navigating). `on_navigate_action`
    now forgets every closing once-key on retake, not just the pose one —
    "closing:review" must stay in that set too, or a second pass through
    review after a retake would silently skip asking about the new photo.
    """

    def __init__(self) -> None:
        self._last_key = ""
        self._last_at = 0.0
        self._once_keys: set[str] = set()

    @property
    def last_key(self) -> str:
        return self._last_key

    def already_narrated(self, key: str) -> bool:
        return key in self._once_keys

    def mark_narrated(self, key: str) -> None:
        self._once_keys.add(key)

    def forget_narrated(self, key: str) -> None:
        self._once_keys.discard(key)

    def is_duplicate(self, key: str) -> bool:
        now = time.monotonic()
        if key == self._last_key and (now - self._last_at) < _PANTALLA_DEDUPE_SECONDS:
            return True
        self._last_key = key
        self._last_at = now
        return False

    def on_navigate_action(self, action: str) -> None:
        # retake_photo restarts the closing pose → capture → review →
        # generating → delivered cycle. Forget every once-only key in that
        # cycle — not just "closing:photo" — or the second pass through
        # review/generating/delivered is silently skipped and the model
        # never gets fresh instructions telling it to ask about the new
        # photo or call navigate_journey(advance).
        if action == "retake_photo":
            for key in (
                "closing:photo",
                "closing:countdown",
                "closing:review",
                "closing:generating",
                "closing:delivered",
            ):
                self.forget_narrated(key)
            logger.info(
                "[navigate_journey] retake_photo — reset closing pantalla guards "
                "(photo, countdown, review, generating, delivered)"
            )


# Spoken while closing:generating — single SETI aside (no rotating fact list).
_GENERATING_SETI_FACTS: tuple[str, ...] = (
    "En SETI entendemos el negocio antes de proponer una solución, porque "
    "sabemos que la tecnología sólo genera valor cuando responde a desafíos "
    "reales. En SETI, crecemos para nuestros clientes.",
)

_GENERATING_KEEPALIVE_FIRST_DELAY_S = 14.0
_GENERATING_KEEPALIVE_REPEAT_S = 18.0
# Entry cue already speaks the single SETI purpose line. Keepalive would
# re-say the same phrase (~14s later) — seen in 2026-09-05_13-05-33 logs.
# Leave at 0 so we only schedule filler if we reintroduce distinct facts.
_GENERATING_KEEPALIVE_MAX_TICKS = 0


def _generating_keepalive_instructions(tick: int) -> str:
    """Instructions for the Nth (0-indexed) closing:generating filler line.

    Uses the fixed SETI purpose line — framed as a brief «mientras tanto»
    aside, never as if the card were ready.
    """
    fact = _GENERATING_SETI_FACTS[tick % len(_GENERATING_SETI_FACTS)]
    return (
        "CLOSING GENERATING — la tarjeta sigue en proceso, el visitante "
        "sigue esperando. Locución corta: abre con «Mientras tanto,» (o "
        "«Mientras se genera tu tarjeta,») y di este mensaje de SETI en "
        "español natural, casi literal — no inventes otros datos: "
        f"«{fact}» "
        "PROHIBIDO pedir foto. PROHIBIDO ABSOLUTO decir o insinuar que la "
        "tarjeta o el informe YA están listos, generados o pueden enviarse — "
        "eso solo es cierto cuando llegue [pantalla:closing:delivered]. "
        "Tras decirla, SILENCIO hasta la próxima instrucción o esa pantalla."
    )


# Regression (2026-09-02, RM_vZnfXrLvRboG logs): see the NOVA_SESSION_REFRESH_
# SECONDS comment above. A mid-call Nova recycle left the model speaking as if
# it were acting without ever calling a tool again. This instruction re-anchors
# it in the real UI state the instant the recycle-driven session_reconnected
# event fires, before it reacts to anything the visitor says next.
_SESSION_RECONNECTED_INSTRUCTIONS = (
    "RECONEXIÓN TÉCNICA — la sesión de voz se acaba de reconectar por dentro "
    "(invisible para el visitante, nunca lo menciones). Antes de decir o "
    "hacer cualquier otra cosa: llama get_session_state PRIMERO para "
    "recuperar el paso y la fase reales — PROHIBIDO asumir, recordar o "
    "improvisar en qué pantalla está a partir de lo que dijiste antes de "
    "reconectar. Si ya habías anunciado una acción (tomar foto, enviar "
    "reporte, avanzar, etc.) y no llamaste al tool correspondiente, "
    "retómala ahora con el tool real — nunca la des por hecha solo porque "
    "la mencionaste en voz. A partir de aquí sigue el contrato normal: "
    "get_session_state en cada turno, present_content si hace falta, "
    "navigate_journey solo cuando el visitante confirme una acción de "
    "availableActions."
)


# Grace period to let an in-flight utterance finish naturally before
# forcing new screen content through. Purely dynamic — checks whether the
# agent is actually speaking right now (session.current_speech), not a
# per-phase/per-cue allowlist. Previously the handler decided whether to
# hard-interrupt by checking cue *type* (detail_auto / closing / etc.),
# which meant every new screen category needed to be added to that list by
# hand or it would chop off mid-sentence narration (e.g. analysis:complete
# arriving while the scanning findings were still being read out). This
# applies the same "let it finish, or bounded-timeout-interrupt" policy to
# every cue uniformly, no matter which screen it's for.
#
# Regression (2026-09-02, RM_3HK2n8CFPegT logs): a closing:generating SETI
# fact filler (~9-11s to speak at Nova's pace) was still mid-sentence when
# closing:delivered landed a few seconds in. The old 8s grace period cut it
# off mid-word ("...bajo el propósito «Crecemos para" — never finished
# "nuestros clientes»"). 8s was sized for the old one-line "componiendo tu
# tarjeta" filler, not the longer grounded facts added afterward. Bumped
# with headroom for the longest fact + a short lead-in at natural pace.
#
# This long grace is ONLY for the two "reading a filler while something
# loads" call sites (_generating_keepalive, closing_instructions) — they
# pass it explicitly. Everything else (touch navigation: detail cards,
# welcome_ready, the generic fallback) uses _PANTALLA_INTERRUPT_GRACE_NAV_S
# instead, since a 12s wait there reads as an unresponsive agent, not a
# graceful finish (2026-09-04, RM_9vbBWpPmZjYn-follow-up logs: user tapped
# a dimension card, agent kept narrating the previous dimension for the
# full 12s before switching). This is a two-way split, not a per-screen
# allowlist — new screens default to the responsive grace automatically.
# Touch / button navigation must cut old speech immediately (LiveKit:
# session.interrupt() then generate_reply). A 2s wait left welcome audio
# playing on intro/analysis (2026-09-04_12-59-37 logs). Generating filler
# still uses the long grace explicitly.
# Long enough for the full SETI purpose line + short «mientras tanto» lead-in
# (~29 words ≈ 14s at 2.5 w/s) without cutting mid-sentence on delivered.
_PANTALLA_INTERRUPT_GRACE_S = 16.0
_PANTALLA_INTERRUPT_GRACE_NAV_S = 0.0

# Keep strong refs to fire-and-forget wait-then-speak tasks so they can't be
# garbage-collected mid-flight; each discards itself once done.
_pending_pantalla_replies: set[asyncio.Task[None]] = set()


def _cancel_pending_pantalla_replies() -> int:
    """Cancel in-flight wait-then-speak tasks from earlier pantalla cues.

    Rapid taps used to leave multiple independent tasks alive; each eventually
    called generate_reply(), so old-screen welcome/detail speech and tools
    finished after the UI had already moved.
    """
    cancelled = 0
    for task in list(_pending_pantalla_replies):
        if not task.done():
            task.cancel()
            cancelled += 1
        _pending_pantalla_replies.discard(task)
    return cancelled


async def _commit_guide_screen(dedupe_key: str | None) -> None:
    """Reveal the held UI only after kill — lockstep with new speech."""
    await commit_guide_screen(dedupe_key)


def _deliver_pantalla_reply(
    agent_session: AgentSession,
    instructions: str,
    grace_s: float = _PANTALLA_INTERRUPT_GRACE_NAV_S,
    pantalla_guard: _PantallaGuard | None = None,
    dedupe_key: str | None = None,
    speak: bool = True,
) -> None:
    """Speak `instructions` only after a real kill of prior speech/tools.

    Defaults to the short, touch-responsive grace period (interrupt right
    away). Callers reading a filler while something loads (generating
    keep-alive, closing) pass grace_s=_PANTALLA_INTERRUPT_GRACE_S explicitly
    to let it finish naturally first.

    Hard kill path: cancel superseded pending reply tasks, then
    ``kill_agent_speech`` = await ``interrupt(force=True)`` +
    ``output.audio.clear_buffer()`` + ``wait_for_idle`` + Nova VAD settle.
    Soft interrupt alone left residual audio and prior-turn tools running on
    the new screen; frontend mute is not the kill — this is.

    After kill, ``commit_guide_screen`` reveals the pending UI, then
    ``generate_reply`` starts — view + speech together (not UI-first).
    Pass speak=False for silence-only screens (photo countdown) — kill +
    commit, no new locution.

    Freshness: pass pantalla_guard + dedupe_key so this checks
    _PantallaGuard.last_key right before speaking and drops itself if a newer
    cue has since superseded it.
    """
    superseded = _cancel_pending_pantalla_replies()
    if superseded:
        logger.info(
            "[text_input] Cancelled %d superseded pantalla reply task(s) before %s",
            superseded,
            dedupe_key or "?",
        )

    def _stale() -> bool:
        return (
            pantalla_guard is not None
            and dedupe_key is not None
            and pantalla_guard.last_key != dedupe_key
        )

    async def _run() -> None:
        if grace_s > 0:
            current = agent_session.current_speech
            if current is not None and not current.done():
                # Poll in short slices so a captured `current` reference that
                # goes stale mid-wait doesn't eat the whole grace as dead air
                # (Nova autonomous turns can leave wait_for_playout hanging).
                poll_s = min(0.5, grace_s)
                remaining = grace_s
                finished_naturally = False
                while remaining > 0:
                    step = min(poll_s, remaining)
                    try:
                        await asyncio.wait_for(current.wait_for_playout(), timeout=step)
                        finished_naturally = True
                        break
                    except asyncio.TimeoutError:
                        remaining -= step
                        if agent_session.current_speech is not current:
                            break

                if finished_naturally and _stale():
                    logger.info(
                        "[text_input] Dropping superseded pantalla reply "
                        "(finished naturally, newer cue arrived): %s",
                        dedupe_key,
                    )
                    return
                if not finished_naturally:
                    logger.info(
                        "[text_input] Grace period elapsed (%.1fs) — "
                        "killing speech for new screen content: %s",
                        grace_s,
                        dedupe_key,
                    )

        await kill_agent_speech(agent_session)

        if _stale():
            logger.info(
                "[text_input] Dropping superseded pantalla reply (after kill): %s",
                dedupe_key,
            )
            return

        await _commit_guide_screen(dedupe_key)
        if not speak:
            logger.info(
                "[text_input] silence-only pantalla (no generate_reply): %s",
                dedupe_key,
            )
            return
        reply_kwargs: dict = {"instructions": instructions}
        chat_ctx = build_pantalla_chat_ctx(agent_session)
        if chat_ctx is not None:
            reply_kwargs["chat_ctx"] = chat_ctx
        agent_session.generate_reply(**reply_kwargs)

    task = asyncio.create_task(_run())
    _pending_pantalla_replies.add(task)
    task.add_done_callback(_pending_pantalla_replies.discard)


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
    _pantalla_guard = _PantallaGuard()

    def _pantalla_already_narrated(key: str) -> bool:
        return _pantalla_guard.already_narrated(key)

    def _mark_pantalla_narrated(key: str) -> None:
        _pantalla_guard.mark_narrated(key)

    def _forget_pantalla_narrated(key: str) -> None:
        _pantalla_guard.forget_narrated(key)

    def _on_navigate_action(action: str) -> None:
        _pantalla_guard.on_navigate_action(action)

    def _should_skip_duplicate_pantalla(key: str) -> bool:
        return _pantalla_guard.is_duplicate(key)

    # "generating" typically runs ~20-40s in practice — long enough that
    # total silence after the one allowed status line can feel dead, and
    # long enough to outlast a single filler line. But the earlier fix (say
    # one status line, then go silent) exists specifically because letting
    # the model re-narrate on its own led to it hallucinating a premature
    # "your card is ready" a few seconds in — or, seen later, repeating its
    # own prior line verbatim when it had nothing new to say. So this stays
    # scheduled by CODE on a timer, never left to the model's own judgment
    # about when to speak again: each tick calls
    # `_generating_keepalive_instructions(tick)` for content that is always
    # distinct from the last (a new SETI fact per tick) and constrained to
    # never claim completion, and stops once a newer cue (closing:delivered)
    # has landed or `_GENERATING_KEEPALIVE_MAX_TICKS` is reached.
    async def _generating_keepalive(
        agent_session: AgentSession, token_key: str
    ) -> None:
        delay = _GENERATING_KEEPALIVE_FIRST_DELAY_S
        for tick in range(_GENERATING_KEEPALIVE_MAX_TICKS):
            await asyncio.sleep(delay)
            # If a newer distinct cue (e.g. closing:delivered) has already
            # landed, the wait is over — nothing to fill anymore.
            if _pantalla_guard.last_key != token_key:
                return
            logger.info(
                "[text_input] generating keep-alive firing (tick=%d, still on %s)",
                tick,
                token_key,
            )
            _deliver_pantalla_reply(
                agent_session,
                _generating_keepalive_instructions(tick),
                grace_s=_PANTALLA_INTERRUPT_GRACE_S,
            )
            delay = _GENERATING_KEEPALIVE_REPEAT_S

    def _pantalla_text_input_handler(
        agent_session: AgentSession, event: room_io.TextInputEvent
    ) -> None:
        is_pantalla = event.text.startswith("[pantalla:")
        if is_pantalla and not _on_enter_done.is_set():
            # Queue rather than drop, for both backends. Previously Nova
            # Sonic sessions dropped cues that arrived here outright — if
            # on_enter's generate_reply ran long (it's now bounded by
            # ON_ENTER_MAX_WAIT_S, but was previously bounded only by the
            # realtime model's 45s generate_reply_timeout), any real screen
            # change during that window was lost for good, leaving the
            # agent to finish speaking stale, ungrounded context. Queueing
            # means _replay_queued_pantalla always catches the visitor up
            # once on_enter finishes or is force-interrupted.
            _pantalla_queue.append(event.text)
            logger.info(
                "[text_input] Queued pantalla cue (on_enter active): %.80s",
                event.text,
            )
            return
        # CRITICAL: never pass the raw [pantalla:] English cue as user_input —
        # Nova often reads it aloud ("UI step…", "focus=…"). Use instructions
        # so the model runs tools and speaks visitor-facing Spanish only.
        if is_pantalla:
            dedupe_key = _pantalla_dedupe_key(event.text)
            # dedupe_key (not the truncated cue text) is what actually
            # carries the dimension id for detail:continuous/detail:revisit
            # cues (see _pantalla_dedupe_key's DIM_ID= parsing) — the cue
            # body itself is instruction prose that runs well past 120
            # chars before ever reaching that marker, so logging only
            # event.text left every past debugging session unable to
            # confirm which dimension a given cue actually targeted.
            logger.info(
                "[text_input] pantalla cue (not spoken) key=%s: %.120s",
                dedupe_key,
                event.text,
            )
            if _should_skip_duplicate_pantalla(dedupe_key):
                logger.info(
                    "[text_input] Skipping duplicate pantalla (%.1fs): %s",
                    _PANTALLA_DEDUPE_SECONDS,
                    dedupe_key,
                )
                return

            intro_orchestrator_start = (
                "INTRO_ORCHESTRATOR_START" in event.text or "intro:run" in event.text
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
            welcome_ready = "[pantalla:welcome:ready]" in event.text or (
                "phase=ready" in event.text and "welcome" in event.text
            )
            welcome_preparing = "[pantalla:welcome:preparing]" in event.text or (
                "phase=preparing" in event.text and "welcome" in event.text
            )
            closing_instructions = _closing_pantalla_instructions(event.text)

            if welcome_preparing:
                logger.info(
                    "[text_input] Ignoring preparing pantalla (silent prewarm): %.80s",
                    event.text,
                )
                return

            if intro_orchestrator_start:
                if schedule_intro_tour(agent_session):
                    logger.info("[text_input] intro orchestrator started from pantalla")
                else:
                    logger.info(
                        "[text_input] intro orchestrator already active — "
                        "ignored pantalla: %.80s",
                        event.text,
                    )
                return

            # Other intro-step cues (not intro:run) are owned by the Python
            # orchestrator — never let them race a second generate_reply.
            if "step=intro" in event.text and "intro:run" not in event.text:
                logger.info(
                    "[text_input] Suppressed non-run intro pantalla: %s",
                    dedupe_key,
                )
                return

            # Touch/UI left intro while the tour was still speaking. The frontend
            # correctly pushes [pantalla:analysis|detail|…] but we used to
            # suppress every cue until the intro finished — so voice kept
            # reading the onboarding script while the visitor was already on
            # results (see logs 2026-09-04_12-08-50: analysis:scanning /
            # detail:continuous all "Suppressed pantalla during intro").
            # LiveKit pattern: interrupt() + generate_reply for the new screen.
            if intro_tour_running():
                logger.info(
                    "[text_input] Cancelling intro orchestrator — UI navigated to: %s",
                    dedupe_key,
                )
                cancel_intro_tour()
                try:
                    agent_session.interrupt(force=True)
                except Exception:
                    logger.warning(
                        "[text_input] interrupt(force=True) after intro cancel raised",
                        exc_info=True,
                    )
                # Fall through to normal per-screen pantalla delivery below.

            # Whether to speak immediately or let an in-flight utterance finish
            # first is decided dynamically by _deliver_pantalla_reply (checks
            # agent_session.current_speech) — not by cue type. No hardcoded
            # per-phase exemption list to maintain as new screens are added.
            #
            # Revisiting an already-toured dimension used to skip straight to
            # "quieres el informe, volver, u otra dimensión?" with zero
            # narration (a past fix to avoid tedious verbatim repeats). User
            # feedback (2026-09-04): that reads as broken, not concise — every
            # detail entry, first visit or not, must narrate the dimension.
            # detail_revisit now shares detail_auto's full-narration
            # instructions (below), with one addition: vary the retelling
            # instead of repeating the earlier phrasing verbatim.
            if detail_auto:
                revisit_note = (
                    " Esta dimensión ya se narró antes en la conversación — cuéntala "
                    "con ángulo y palabras distintas a tu narración anterior (mismo "
                    "contenido de facts, otra forma de decirlo); PROHIBIDO repetir la "
                    "misma frase o estructura que usaste la primera vez. "
                    "PROHIBIDO ABSOLUTO decir «ya la vimos», «ya hablamos de esto», «ya "
                    "cubrimos/cubierto esta dimensión», «ya revisamos/revisado esta "
                    "dimensión», o cualquier variante que use el historial de la "
                    "conversación como excusa para NO narrar evidencia/brechas/tácticas "
                    "de nuevo — esto ya pasó antes y el visitante se quedó sin narración "
                    "en su segunda visita a una dimensión."
                    if detail_revisit
                    else ""
                )
                _deliver_pantalla_reply(
                    agent_session,
                    "DETAIL_CONTINUOUS — La dimensión enfocada pudo haber cambiado desde tu "
                    "última respuesta (el visitante pudo haber navegado por voz o tocando la "
                    "pantalla). ANTES de decir una sola palabra: llama get_session_state y usa "
                    "SOLO facts.evidence / facts.gaps / facts.tactics de ESA respuesta fresca. "
                    "PROHIBIDO ABSOLUTO componer esta síntesis desde datos o el nombre de una "
                    "dimensión que hayas narrado antes en la conversación — cada entrada a esta "
                    "pantalla es una dimensión nueva hasta que get_session_state confirme lo "
                    "contrario. "
                    "Síntesis BREVE, no exhaustiva: 3-4 frases en total, "
                    "no una recitación bloque por bloque. Elige SOLO la evidencia más "
                    "relevante de facts.evidence, la brecha más importante de facts.gaps, "
                    "y UNA táctica concreta de facts.tactics — parafraseado en prosa natural "
                    "ligada a facts.role en facts.company, explicando brevemente el POR QUÉ."
                    + revisit_note
                    + " "
                    "UNA sola respuesta SIN silencios ni pausas para esas 3-4 frases. "
                    "PROHIBIDO present_content extra ni get_session_state OTRA VEZ a mitad de "
                    "la síntesis (una sola llamada a get_session_state al inicio es obligatoria; "
                    "una segunda llamada a mitad de la síntesis no lo es). "
                    "PROHIBIDO rótulos Fortalezas/Oportunidades/Plan. "
                    "UI resalta secciones sola — tú sigues hablando sin interrupción. "
                    "OBLIGATORIO — la ÚLTIMA oración de esta locución DEBE ser una pregunta "
                    "de elección (no un punto final tras la táctica): "
                    "«¿Quieres el informe, volver al globo, u otra dimensión?» "
                    "o equivalente — VARÍA la redacción de esta pregunta cada vez, nunca "
                    "la misma frase exacta que en la dimensión anterior. "
                    "Sin esa pregunta la locución está incompleta. "
                    "Luego PARA y ESPERA. Si el visitante pide más detalle, ahí sí profundiza.",
                    pantalla_guard=_pantalla_guard,
                    dedupe_key=dedupe_key,
                )
            elif welcome_ready:
                _deliver_pantalla_reply(
                    agent_session,
                    "WELCOME READY — PASO 1: "
                    "Llama get_session_state AHORA MISMO antes de hablar. "
                    "Usa facts.name, facts.role y facts.company del resultado para componer "
                    "UN saludo propio en español natural (~10-12 s). "
                    "PROHIBIDO ABSOLUTO: leer en voz alta texto entre corchetes como [nombre], "
                    "[rol], [empresa] o cualquier otro placeholder — son variables internas, NUNCA se dicen. "
                    "PROHIBIDO meta-comentarios: 'vamos a proceder', 'procederé', 'realizaré el saludo'. "
                    "Entra directo al saludo. 2-3 frases: quién es el visitante + qué es Huella Digital. "
                    "Si availableActions incluye accept_data_consent (o facts.dataConsentRequired): "
                    "pide EXPLÍCITAMENTE el consentimiento — «¿Aceptas el tratamiento de tus "
                    "datos personales conforme a la política de protección de datos de SETI?» "
                    "— o pide marcar el check de protección de datos. "
                    "PROHIBIDO nombrar o listar las cinco dimensiones en esta bienvenida. "
                    "PROHIBIDO present_content. PROHIBIDO navigate_journey en este paso "
                    "(salvo accept_data_consent si ya dijo acepto/autorizo/de acuerdo EXPLÍCITAMENTE "
                    "en este turno — un «sí» suelto o «comenzar/empezamos/adelante/vamos/dale/listo», "
                    "incluso junto a «sí», NO cuenta como consentimiento). "
                    "PARA y espera. "
                    "PASO 2 — consentimiento: si dice acepto / de acuerdo / autorizo (no un «sí» "
                    "genérico ni una palabra de continuar), llama navigate_journey(accept_data_consent). "
                    "Si dice solo una palabra de continuar sin esas palabras de consentimiento: "
                    "NO llames accept_data_consent — pregunta el consentimiento explícito y espera. "
                    "PROHIBIDO start_experience mientras accept_data_consent esté en availableActions. "
                    "PASO 3 — solo cuando start_experience esté disponible y confirme "
                    "(sí / continuar / adelante / vamos / dale): "
                    "llama navigate_journey(start_experience) — nunca junto al saludo inicial, "
                    "nunca sin consentimiento. "
                    "Tras ok: NO digas nada más en este turno — ni el saludo, ni "
                    "palabras como «silencio» o «esperando», ni ningún comentario "
                    "de cierre. Deja que [pantalla:intro] continúe sola.",
                    pantalla_guard=_pantalla_guard,
                    dedupe_key=dedupe_key,
                )
            elif closing_instructions:
                once_key = dedupe_key
                # Touch retake / photo-accept re-enters pose without
                # navigate_journey(retake_photo), so once-keys must clear here.
                # BUG (2026-09-05): used bare `text` → NameError; pantalla never
                # ran → no interrupt, no pose invite (consent speech kept playing).
                pose_reentry = once_key in ("closing:photo", "closing:pose") and (
                    "CLOSING_RETAKE" in event.text
                    or "CLOSING_PHOTO_ACCEPT" in event.text
                )
                if pose_reentry:
                    _pantalla_guard.on_navigate_action("retake_photo")
                    logger.info(
                        "[text_input] closing pose re-entry — cleared once-keys "
                        "(%s)",
                        once_key,
                    )
                # Normalize pose aliases onto the once-key used for first visit.
                if once_key == "closing:pose":
                    once_key = "closing:photo"
                if _pantalla_already_narrated(once_key):
                    logger.info(
                        "[text_input] Skipping repeat closing pantalla: %s", once_key
                    )
                    return
                _mark_pantalla_narrated(once_key)
                # photo_consent / delivered / thanks are real screen changes —
                # interrupt immediately (nav grace). Only generating filler
                # keeps the long grace so SETI facts aren't cut mid-sentence.
                # Countdown: short photo aside once; generating keeps long grace.
                closing_grace = (
                    _PANTALLA_INTERRUPT_GRACE_S
                    if once_key == "closing:generating"
                    else _PANTALLA_INTERRUPT_GRACE_NAV_S
                )
                _deliver_pantalla_reply(
                    agent_session,
                    closing_instructions,
                    grace_s=closing_grace,
                    pantalla_guard=_pantalla_guard,
                    dedupe_key=dedupe_key,
                )
                if once_key == "closing:generating" and _GENERATING_KEEPALIVE_MAX_TICKS > 0:
                    task = asyncio.create_task(
                        _generating_keepalive(agent_session, once_key)
                    )
                    _pending_pantalla_replies.add(task)
                    task.add_done_callback(_pending_pantalla_replies.discard)
            elif dedupe_key == "analysis:complete" and _pantalla_already_narrated(
                "analysis:complete"
            ):
                logger.info("[text_input] Skipping repeat analysis:complete pantalla")
                return
            else:
                analysis_instructions = _analysis_pantalla_instructions(dedupe_key)
                if dedupe_key == "analysis:complete":
                    _mark_pantalla_narrated("analysis:complete")
                # Let scanning narration finish before complete (loading UI
                # should track voice; nav grace 0 was cutting mid-LinkedIn).
                analysis_grace = (
                    _PANTALLA_INTERRUPT_GRACE_S
                    if dedupe_key == "analysis:complete"
                    else _PANTALLA_INTERRUPT_GRACE_NAV_S
                )
                _deliver_pantalla_reply(
                    agent_session,
                    analysis_instructions
                    or (
                        "Cambio de foco en pantalla. "
                        "Llama get_session_state primero — ancla en step/phase actuales. "
                        "Luego present_content solo si hace falta. "
                        "PROHIBIDO repetir la misma locución si ya cubriste este phase. "
                        "PROHIBIDO continuar hablando de la dimensión o pantalla anterior — "
                        "el visitante ya cambió de vista. "
                        "PROHIBIDO repetir el saludo de welcome. "
                        "PROHIBIDO UI/pantalla/tarjeta meta."
                    ),
                    grace_s=analysis_grace,
                    pantalla_guard=_pantalla_guard,
                    dedupe_key=dedupe_key,
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
                agent_session.interrupt(force=True)
                agent_session.generate_reply(
                    user_input=event.text,
                    instructions=(
                        "INTRO TOUR ACTIVE — el orchestrator Python narra el reel. "
                        "Si el visitante hace una pregunta directa: responde en ≤2 frases. "
                        "Si el visitante pide re-explicar una tarjeta — interpreta de forma AMPLIA: "
                        "cualquier señal de confusión, 'no entendí', 'no entendí bien', "
                        "'explícame de nuevo', 'repite', 'otra vez', 'de nuevo', 'me explicas', "
                        "'no quedó claro', 'dimensiones', 'cómo interactúo', 'entregables', 'ítem N', "
                        "'¿cuáles son?', '¿qué miden?', '¿cómo funciona?', '¿qué recibo?' — "
                        "llama navigate_journey(replay_intro_card, index=N) "
                        "y narra esa tarjeta con más detalle (N=0 cómo interactuar, N=1 dimensiones, N=2 entregables). "
                        "Si hay duda sobre cuál tarjeta: usa N=1 si mencionó dimensiones, "
                        "N=0 si preguntó cómo interactuar, N=2 si preguntó qué recibe. "
                        "NUNCA respondas con 'no puedo' ni rechaces — SIEMPRE ejecuta replay_intro_card. "
                        "PROHIBIDO present_content y navigate_journey(advance/start_experience). "
                        f"{_USER_VOICE_TOOL_HINT}"
                    ),
                )
                return
            agent_session.interrupt(force=True)
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

    _telemetry_tasks: set[asyncio.Task[None]] = set()

    def _spawn_telemetry(coro) -> None:
        task = asyncio.create_task(coro)
        _telemetry_tasks.add(task)
        task.add_done_callback(_telemetry_tasks.discard)

    @session.on("error")
    def _on_session_error(ev) -> None:
        err = getattr(ev, "error", ev)
        logger.error(
            "session_error type=%s error=%s",
            type(err).__name__,
            err,
            exc_info=isinstance(err, BaseException),
        )
        event = "throttle" if is_throttle_error(err) else "error"
        _spawn_telemetry(
            report_voice_telemetry(
                event,
                room=ctx.room.name,
                detail=f"{type(err).__name__}: {err}",
            )
        )

    if use_cartesia:
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

    _last_usage_tokens = 0

    @session.on("session_usage_updated")
    def _on_session_usage(ev) -> None:
        nonlocal _last_usage_tokens
        usage = getattr(ev, "usage", None)
        model_usage = getattr(usage, "model_usage", None) or []
        total = 0
        for item in model_usage:
            total += int(getattr(item, "input_tokens", 0) or 0)
            total += int(getattr(item, "output_tokens", 0) or 0)
            total += int(getattr(item, "total_tokens", 0) or 0)
        delta = max(0, total - _last_usage_tokens)
        _last_usage_tokens = total
        if delta > 0:
            _spawn_telemetry(
                report_voice_telemetry(
                    "usage",
                    tokens=delta,
                    room=ctx.room.name,
                )
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

    @session.on("session_reconnected")
    def _on_session_reconnected(event) -> None:
        # See install_nova_session_reconnected_event_fix() and the
        # RM_vZnfXrLvRboG regression note above NOVA_SESSION_REFRESH_SECONDS.
        logger.info(
            "[SESSION] session_reconnected — re-anchoring via get_session_state"
        )
        _deliver_pantalla_reply(session, _SESSION_RECONNECTED_INSTRUCTIONS)

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
        agent=Assistant(on_enter_done=_on_enter_done, on_navigate=_on_navigate_action),
        room=ctx.room,
        room_options=room_opts,
        record=_telemetry_record_option(),
    )

    await report_voice_telemetry("session_start", room=ctx.room.name)

    async def _report_session_end() -> None:
        await report_voice_telemetry("session_end", room=ctx.room.name)

    ctx.add_shutdown_callback(_report_session_end)

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
