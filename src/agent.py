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
from narration_barrier import NarrationBarrier, set_session_narration_barrier
from nova_session_continuation import (
    install_nova_session_continuation_fix,
    install_nova_session_reconnected_event_fix,
)
from rpc_client import rpc, wait_for_kiosk_participant
from tasks.intro_orchestrator import intro_tour_running, schedule_intro_tour

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
       qué es Huella Digital y que explorarán su presencia. Termina con
       una invitación («¿Vemos cómo funciona?» o similar) y PARA.
       PROHIBIDO llamar navigate_journey mientras hablas en este paso.
    PASO 2 — Después de que el visitante responda (continuar / adelante /
       seguimos / listo / empezamos / sí): llama navigate_journey(start_experience) en ese
       momento — NO en el mismo turno que el saludo.
    Tras start_experience ok: NO digas nada más en este turno — ni el
    saludo, ni nombre/rol/empresa, ni palabras como «silencio» o
    «esperando» ni ningún meta-comentario sobre pausas. Deja que
    [pantalla:intro:run] continúe sola; sigue Screen 5 cuando llegue.
    PROHIBIDO: present_content en welcome:ready.
    PROHIBIDO: adelantar el contenido del reel de onboarding aquí.
    PROHIBIDO: listar las 5 dimensiones aquí — se presentarán en el onboarding.

    ────────────────────────────────────────────────
    Screen 5 — ONBOARDING «Antes de empezar, así funciona»
    ────────────────────────────────────────────────
    El runtime Python entrega UNA locución breve (~25-30 s) que cubre los tres grupos:
    interacción (gestos + voz), las 5 dimensiones con una idea muy corta de qué
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
      puedes explicar cualquier elemento visible en el reel (gestos, dimensiones,
      entregables) sin entrar en análisis detallado.
    - Tras la locución, pregunta UNA vez «¿Empezamos el análisis?» y PARA.
    - Solo start_analysis tras confirmación del visitante.

    RE-EXPLICAR UNA PARTE (a petición explícita del visitante):
    Si el visitante pide escuchar otra vez — interpreta de forma AMPLIA:
    «explícame de nuevo», «repite», «otra vez», «de nuevo», «no entendí»,
    «no entendí bien», «me explicas», «no quedó claro»,
    «¿y los gestos?», «¿qué recibo?», «las dimensiones», «¿cuáles son?»,
    «¿qué miden?», «¿cómo funciona?», «ítem uno/dos/tres», «la primera/segunda/tercera»,
    cualquier pregunta sobre Autoridad, SSI, Mensaje, Influencia, Higiene,
    Radar, Informe, Correo:
      1) NUNCA digas que no puedes — siempre puedes, siempre lo haces.
         NUNCA respondas con frases de seguridad o política interna.
      2) Llama navigate_journey(replay_intro_card, index=N)
         donde N = 0 (gestos), 1 (dimensiones), 2 (entregables).
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
       En el mismo flujo (3-5 oraciones): UNA fortaleza concreta de
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
    - photo_consent (NUEVA fase): UNA pregunta natural que mencione AMBAS
      opciones, no solo la de aceptar — «¿Quieres tomarte una foto para tu
      tarjeta? Será la portada visual de tu informe. Si prefieres, también
      puedes continuar sin foto.»
      Dos caminos:
        • El visitante dice sí / quiero / adelante →
            LLAMA navigate_journey(ready_for_picture) PRIMERO [genera la tarjeta con foto]
        • El visitante dice no / omitir / sin foto →
            LLAMA navigate_journey(skip_photo) PRIMERO [arma la tarjeta igual, sin foto — pasa por generating y delivered]
      En ambos casos: llama el tool ANTES de decir cualquier frase de
      confirmación o del siguiente paso («colócate», «perfecto, continuamos
      sin foto»...). El tool call es lo que mueve la pantalla — narrar el
      resultado sin haberlo llamado deja la pantalla sin avanzar aunque tu
      voz suene como si ya hubiera pasado.
      PROHIBIDO avanzar sin respuesta. PROHIBIDO preguntar dos veces.
      Si el visitante no responde o la voz no funciona, los botones
      en pantalla están disponibles («Sí, con foto» / «Continuar sin foto»).
    - pose: UNA locución — invita al visitante a colocarse frente al espejo.
      Dila UNA SOLA VEZ y luego SILENCIO — PROHIBIDO repetirla ni reformularla
      con otras palabras mientras esperas, sin importar cuánto tarde el visitante
      en colocarse. Cuando confirme (listo, toma la foto, adelante):
      navigate_journey(ready_for_picture). Si no dice nada, el botón
      «Estoy listo» en pantalla también funciona.
    - generating: locución CORTA — componiendo entrega para su correo. Dila
      UNA SOLA VEZ y luego SILENCIO — PROHIBIDO repetirla o reformularla con
      otras palabras, PROHIBIDO volver a llamar get_session_state por tu
      cuenta para dar otra actualización, sin importar cuánto tarde.
      PROHIBIDO decir o insinuar que la tarjeta/informe YA están listos o
      generados — eso NO es verdad todavía en esta fase; solo lo dirás
      cuando de verdad llegue [pantalla:closing:delivered].
      PROHIBIDO pedir tomar foto (ya se tomó, o el visitante prefirió omitirla).
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
      si confirma EXPLÍCITAMENTE enviar («sí», «envía», «dale», «manda el reporte»).
      Un «no» o «no quiero enviar el reporte» SIN mencionar la foto NO es lo mismo
      que pedir la foto — no asumas cuál de las dos opciones quiere: pregunta en
      UNA frase breve cuál prefiere y ESPERA su respuesta.
    - thanks: agradecimiento cálido; invita a escanear el QR para conocer más de SETI;
      cuando confirmen salir (sí, finalizar, finish): LLAMA navigate_journey(finish)
      PRIMERO, antes de decir cualquier despedida — el tool call es lo que realmente
      termina la experiencia, tu voz sola NO la termina. Si algo alcanzas a decir,
      que sea brevísimo y DESPUÉS del tool call, nunca antes: si hablas primero, una
      nueva interrupción del visitante puede cortar tu turno antes de llegar al tool call.
      PROHIBIDO mencionar tarjeta, foto, imagen o correo — ya se explicó en delivered.
      PROHIBIDO repetir análisis o entrega.

    GESTOS (si preguntan o llegan por swipe):
    — Onboarding (reel único dentro de intro): el reel anima solo durante una
      locución continua. Ignora swipes ahí (el cliente no los usa).
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
                handle.interrupt()
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
        welcome_preparation (0..2 prep), result_dimension (solo analysis:results),
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
        - ready_for_picture: en closing:photo_consent o closing:pose|capture cuando el visitante confirma la foto.
        - skip_photo: en closing:photo_consent cuando el visitante quiere omitir la foto.
        - finish: en closing:thanks cuando confirma salir.
        - replay_intro_card + index (0=gestos, 1=dimensiones, 2=entregables):
          vuelve a narrar esa tarjeta cuando el visitante lo pide explícitamente.
          Solo disponible mientras step=intro."""
        payload: dict = {"action": action, "dimensionId": dimension_id}
        if index >= 0:
            payload["index"] = index
        result = await rpc("navigate_journey", payload)
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
    "Si step=welcome y phase=ready y el visitante confirma continuar "
    "(sí, continúa, adelante, comienza, empezamos, listo, vamos, sigan, dale): "
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
    "Si step=closing y phase=photo_consent y el visitante dice sí / quiero foto / con foto: "
    "navigate_journey(ready_for_picture) de inmediato. "
    "Si step=closing y phase=photo_consent y el visitante dice no / sin foto / omitir / skip: "
    "navigate_journey(skip_photo) de inmediato — arma la tarjeta igual, sin foto. "
    "Si step=closing y phase=pose|prep|capture y el visitante pide tomar la foto "
    "(toma la foto, listo, estoy listo, adelante, take picture, toma la): "
    "navigate_journey(ready_for_picture) — no solo hables, ejecuta la acción. "
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
            "confirmar nada. PROHIBIDO narrar el siguiente paso («colócate», «perfecto, "
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
            "tarde. Si confirman estar listos: navigate_journey(ready_for_picture)."
        )
    if "closing:generating" in text:
        return (
            "CLOSING GENERATING — get_session_state. "
            "Locución CORTA (1-2 frases): componiendo tarjeta e informe para su correo. "
            "PROHIBIDO pedir tomar foto (ya se tomó, o el visitante prefirió omitirla). "
            "Di esta locución UNA SOLA VEZ y luego SILENCIO — no la repitas, no la "
            "reformules con otras palabras («diseñando», «armando», «casi listo», etc.), "
            "y PROHIBIDO decir o insinuar que la tarjeta/informe YA están listos, "
            "generados o pueden enviarse — eso no es cierto todavía en esta fase. "
            "No vuelvas a llamar get_session_state por tu cuenta ni des ninguna "
            "actualización más POR INICIATIVA PROPIA, sin importar cuánto tarde — "
            "quédate en silencio hasta la próxima instrucción. Si el sistema te "
            "envía una nueva instrucción de espera con un dato de SETI, ese es un "
            "aviso legítimo del sistema, no algo que decidiste tú: síguela con "
            "una locución breve y distinta, sin comentar que estás siguiendo una "
            "instrucción."
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
            "EXPLÍCITAMENTE enviar (sí / envía / dale / manda el reporte). "
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
    - a "narrated once" set for closing cues (photo, generating, delivered)
      that must be spoken exactly once per pass through that phase.

    Regression: retake_photo sends the visitor through pose → capture →
    generating → delivered again. Only the "closing:photo" once-key was
    ever forgotten on retake, so the second pass through generating/
    delivered was silently swallowed by the once-only guard — the model
    never received fresh navigate_journey(advance) guidance for that
    cycle, so it fell back to improvising ungrounded "sending it now"
    narration in a loop and never actually called the tool (2026-09-02
    RM_SpsHnphyUjch logs: visitor repeated "enviar" many times, agent kept
    saying "se están enviando" without ever navigating). `on_navigate_action`
    now forgets every closing once-key on retake, not just the pose one.
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
        # retake_photo restarts the closing pose → capture → generating →
        # delivered cycle. Forget every once-only key in that cycle — not
        # just "closing:photo" — or the second pass through generating/
        # delivered is silently skipped and the model never gets fresh
        # instructions telling it to call navigate_journey(advance).
        if action == "retake_photo":
            for key in ("closing:photo", "closing:generating", "closing:delivered"):
                self.forget_narrated(key)
            logger.info(
                "[navigate_journey] retake_photo — reset closing pantalla guards "
                "(photo, generating, delivered)"
            )


# Grounded facts for the closing:generating wait — same canonical numbers as
# knowledge_base._OVERVIEW (identity, PRIME model, clients, partners, success
# cases), rewritten as short standalone spoken lines. Used to fill dead air
# while the card/report is being composed, instead of the model either going
# fully silent for 20-40s or (worse, seen in real sessions) improvising
# repeated/ungrounded filler. Never invent new SETI facts elsewhere in this
# file — this tuple and knowledge_base.py are the only sources of truth.
_GENERATING_SETI_FACTS: tuple[str, ...] = (
    "SETI S.A.S. lleva 29 años ayudando a que sus clientes crezcan con "
    'tecnología, bajo el propósito "Crecemos para nuestros clientes".',
    "SETI hace parte del holding KATIO Sistemas Globales Informáticos, con "
    "sede en Madrid, y atiende a más de 160 clientes corporativos.",
    "El modelo PRIME de SETI combina desarrollo de software, ingeniería de "
    "datos e inteligencia artificial, nube, y operación de infraestructura "
    "24/7.",
    "SETI es partner certificado de AWS, Microsoft, Google Cloud, Oracle, "
    "MongoDB e IBM.",
    "SETI ha liderado migraciones críticas como la de BTG Pactual, con "
    "ahorros de costos comprobados para sus clientes.",
    "Detrás de SETI hay más de mil colaboradores especializados en tecnología.",
)

_GENERATING_KEEPALIVE_FIRST_DELAY_S = 14.0
_GENERATING_KEEPALIVE_REPEAT_S = 16.0
# Bounded so a stuck/never-arriving closing:delivered can't turn this into
# indefinite chatter — after this many ticks it goes back to full silence,
# same as the old one-shot behavior.
_GENERATING_KEEPALIVE_MAX_TICKS = 6


def _generating_keepalive_instructions(tick: int) -> str:
    """Instructions for the Nth (0-indexed) closing:generating filler line.

    Regression: visitors reported the wait feeling dead, and in real
    sessions the model — with nothing new to say and no further guidance —
    fell back to literally repeating its own prior line. Nova sees the full
    turn history, so a verbatim repeat reads as the agent being stuck. Each
    tick must therefore sound different from every earlier one in this
    phase; ticks after the first pivot to one grounded SETI fact apiece
    (cycling through `_GENERATING_SETI_FACTS`), framed as a brief
    "while we wait" aside — never as if the card were ready.

    Regression (2026-09-02, RM_MeiLzPnKgbwA logs): with zero lead-in
    guidance, the model recited the SETI fact almost verbatim and cold —
    it landed like a company ad interrupting the wait rather than a "keeping
    you company" aside. An earlier fix had banned a *long* fixed preamble
    ("mientras se termina de armar tu tarjeta...") for eating into the
    speaking budget and getting the fact cut off mid-word — the fix here is
    a much shorter (2-4 word) transition, not the absence of one.
    """
    fact = _GENERATING_SETI_FACTS[tick % len(_GENERATING_SETI_FACTS)]
    return (
        "CLOSING GENERATING — la tarjeta sigue en proceso, el visitante "
        "sigue esperando. Locución MUY corta (1 frase, sin preámbulos "
        "largos — ve directo casi al dato, sin frases de relleno antes), en "
        "tus propias palabras, DISTINTA a cualquier frase ya dicha en esta "
        "fase (incluida la primera locución al entrar a esta pantalla) — "
        "PROHIBIDO repetir o reformular algo ya dicho. Abre con una "
        "transición ultra breve de 2 a 4 palabras («mientras tanto,», "
        "«aprovechando la espera,», «de paso,» o similar) para que no "
        "suene como una interrupción en frío — nunca sueltes el dato "
        "directamente sin esa transición. Usa este dato real de SETI como "
        "contenido, parafraseado, breve, tono cálido, NUNCA leído literal: "
        f"«{fact}» PROHIBIDO ABSOLUTO decir o insinuar que la tarjeta o el "
        "informe YA están listos, generados o pueden enviarse — eso solo es "
        "cierto cuando llegue de verdad [pantalla:closing:delivered]. Tras "
        "decirla, SILENCIO otra vez hasta la próxima actualización o hasta "
        "que llegue esa pantalla."
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
_PANTALLA_INTERRUPT_GRACE_S = 12.0

# Keep strong refs to fire-and-forget wait-then-speak tasks so they can't be
# garbage-collected mid-flight; each discards itself once done.
_pending_pantalla_replies: set[asyncio.Task[None]] = set()


def _deliver_pantalla_reply(agent_session: AgentSession, instructions: str) -> None:
    """Speak `instructions` now if the agent is idle, or once the in-flight
    utterance finishes — bounded wait, then force-interrupt if it runs long."""
    current = agent_session.current_speech
    if current is None or current.done():
        agent_session.generate_reply(instructions=instructions)
        return

    async def _wait_then_speak() -> None:
        try:
            await asyncio.wait_for(
                current.wait_for_playout(), timeout=_PANTALLA_INTERRUPT_GRACE_S
            )
        except asyncio.TimeoutError:
            logger.info(
                "[text_input] In-flight speech exceeded %.0fs — interrupting "
                "for new screen content",
                _PANTALLA_INTERRUPT_GRACE_S,
            )
            current.interrupt()
        agent_session.generate_reply(instructions=instructions)

    task = asyncio.create_task(_wait_then_speak())
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
                agent_session, _generating_keepalive_instructions(tick)
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

            # Python orchestrator owns intro narration — ignore other intro cues.
            if intro_tour_running() or (
                "step=intro" in event.text and "intro:run" not in event.text
            ):
                logger.info(
                    "[text_input] Suppressed pantalla during intro orchestrator: %s",
                    dedupe_key,
                )
                return

            # Whether to speak immediately or let an in-flight utterance finish
            # first is decided dynamically by _deliver_pantalla_reply (checks
            # agent_session.current_speech) — not by cue type. No hardcoded
            # per-phase exemption list to maintain as new screens are added.
            if detail_revisit:
                _deliver_pantalla_reply(
                    agent_session,
                    "DETAIL_REVISIT — Esta dimensión ya se narró. "
                    "PROHIBIDO repetir evidencia/brechas/tácticas. "
                    "PROHIBIDO present_content(detail_section). "
                    "Di UNA frase breve: informe, volver al globo u otra dimensión → ESPERA. "
                    "send_report | back | open_detail(dimension_id=…).",
                )
            elif detail_auto:
                _deliver_pantalla_reply(
                    agent_session,
                    "DETAIL_CONTINUOUS — UNA sola respuesta SIN silencios ni pausas. "
                    "Teje facts.evidence → facts.gaps → facts.tactics en prosa encadenada. "
                    "Parafrasea para facts.role en facts.company — explica POR QUÉ. "
                    "PROHIBIDO parar, callar o END entre bloques o ítems. "
                    "PROHIBIDO present_content extra ni get_session_state entre bloques. "
                    "PROHIBIDO rótulos Fortalezas/Oportunidades/Plan. "
                    "UI resalta secciones sola — tú sigues hablando sin interrupción. "
                    "Al cerrar tácticas: pregunta informe / volver / otra dimensión → PARA y ESPERA.",
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
                    "PROHIBIDO nombrar o listar las cinco dimensiones en esta bienvenida. "
                    "PROHIBIDO present_content. PROHIBIDO navigate_journey en este paso. "
                    "PARA y espera confirmación del visitante. "
                    "PASO 2 — solo cuando confirme (sí / continuar / adelante / vamos / dale): "
                    "llama navigate_journey(start_experience) en ese turno — nunca junto al saludo. "
                    "Tras ok: NO digas nada más en este turno — ni el saludo, ni "
                    "palabras como «silencio» o «esperando», ni ningún comentario "
                    "de cierre. Deja que [pantalla:intro] continúe sola.",
                )
            elif closing_instructions:
                once_key = dedupe_key
                if _pantalla_already_narrated(once_key):
                    logger.info(
                        "[text_input] Skipping repeat closing pantalla: %s", once_key
                    )
                    return
                _mark_pantalla_narrated(once_key)
                _deliver_pantalla_reply(agent_session, closing_instructions)
                if once_key == "closing:generating":
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
                if dedupe_key == "analysis:complete":
                    _mark_pantalla_narrated("analysis:complete")
                _deliver_pantalla_reply(
                    agent_session,
                    "Cambio de foco en pantalla. "
                    "Llama get_session_state primero — ancla en step/phase actuales. "
                    "Luego present_content solo si hace falta. "
                    "PROHIBIDO repetir la misma locución si ya cubriste este phase. "
                    "PROHIBIDO UI/pantalla/tarjeta meta.",
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
                        "INTRO TOUR ACTIVE — el orchestrator Python narra el reel. "
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
