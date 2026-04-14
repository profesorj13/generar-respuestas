"""
Responder tickets de SINIGEP automáticamente.

Lee preguntas de un Google Sheet, busca en la documentación de Mintlify
via MCP usando un loop de RAG iterativo con tool use, genera respuestas con
Claude y las escribe de vuelta en el Sheet.

Uso:
    python responder_tickets.py "https://docs.google.com/spreadsheets/d/SHEET_ID/edit"
    python responder_tickets.py "<URL>" --force-regen   # reescribe respuestas existentes

Requisitos:
    - ANTHROPIC_API_KEY en variable de entorno
    - credentials/*.json (service account de Google)
    - El Sheet debe estar compartido con el email del service account
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import gspread
from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from claverrama import inferir_nivel, normalizar, NIVEL_DESCONOCIDO

# --- Configuración ---

load_dotenv(Path(__file__).parent / ".env")

MCP_URL = "https://educabot-d7d4a6e0.mintlify.app/mcp"
CREDENTIALS_DIR = Path(__file__).parent / "credentials"

MODEL = "claude-sonnet-4-6"
MAX_RAG_ITERATIONS = 8
MAX_RESPUESTA_CHARS = 2000
MAX_HISTORIAL = 3  # tickets previos del mismo establecimiento a inyectar

TOKEN_NO_CONSULTA = "<<NO_ES_CONSULTA>>"
TOKEN_SIN_CONTEXTO = "<<SIN_CONTEXTO>>"

HEADERS_CLAVERRAMA_TOKENS = ("claverrama", "claverama", "clave rama", "clave_rama", "clave-rama")

SYSTEM_PROMPT = """\
ROL Y AUDIENCIA
- Sos un agente de soporte de SINIGEP (Sistema Nacional Integral de Gestión Educativa Provincial).
- Tu respuesta será revisada por el equipo de Educabot antes de enviarse al colegio en BA Colaborativa.
- El usuario consultó hace 6+ días y probablemente no recuerda el contexto exacto de su ticket.

HERRAMIENTAS
- Tenés dos herramientas: search_docs (busca y devuelve títulos + paths + snippet) y get_page (trae una página completa).
- Empezá con 1 o 2 búsquedas exploratorias. Si los snippets no alcanzan, leé las páginas completas con get_page.
- Si el ticket toca varios temas, hacé búsquedas separadas para cada uno.
- No hagas más de 8 tool calls en total. Sé eficiente: para cada subtema hacé 1 search + 1 get_page si el snippet no alcanza.
- Cuando tengas suficiente contexto, generá la respuesta final siguiendo todas las reglas de abajo.

TONO (no negociable)
- Profesional y cordial, natural de Argentina (sin llegar a ser informal).
- Prohibido usar signos de exclamación ("!" ni "¡").
- Dirigirse siempre en plural ("ustedes", "les"), asumiendo que son los representantes de la institución. NUNCA en singular ("te", "tu").
- NUNCA referirse al usuario en tercera persona como "la institución", "el establecimiento" o "la escuela". Hablarles directamente: "pueden solucionarlo" en vez de "la institución lo gestiona".
- NUNCA usar voz de equipo en tercera persona ("le informamos", "le compartimos", "desde el equipo"). Ir directo a la respuesta sin encuadrar quién habla.
- Saludo: "Buenos días." siempre. Nunca "Hola", "Buenas tardes" ni variantes.
- Plain text puro. Prohibido: negrita con asteriscos, itálica, headers con numeral, emojis, caracteres < > { }.
- Permitidos: guiones medios, puntos, números con punto (1. 2. 3.).
- Rutas de navegación: usá "-" como separador, no ">". Ejemplo: "Gestión Académica - Planes de Estudio - Seleccionar plan - Editar".

FRASES PROHIBIDAS (no usar ni parafrasear)
- "Entiendo tu consulta sobre..." / "Entendemos su consulta..."
- "Te ayudo con cada punto" / "Les ayudamos con cada punto"
- "La solución es simple"
- "De momento" repetido
- "Para esto te recomendamos contactar a soporte técnico" — "soporte técnico" NO es un destino válido.
- Cualquier referencia a "ingresá a BA Colaborativa" — es el canal por el que llegó el ticket, jamás puede ser destino.
- Repetir textual lo que preguntó el usuario.
- Verbos robóticos o antinaturales: "asentar", "volcar", "proceder a". Usá "cargar", "ingresar", "completar", "declarar" o "crear".

ERRORES FRECUENTES A EVITAR
- Decir "asignaturas", NUNCA "materias". El sistema usa "asignaturas" en toda la UI.
- La sigla correcta es "DGEGP" (Dirección General de Educación de Gestión Privada), NUNCA "DGEP".
- Si una funcionalidad no existe en el sistema, decilo directamente: "Esa funcionalidad no está disponible en SINIGEP actualmente." NO uses frases ambiguas como "no está documentada como disponible" ni "no encontramos documentación al respecto" — eso sugiere que podría existir pero no la encontraste.
- Si el usuario describe que la interfaz "se bloquea", "se congela", "no responde" o "no deja completar campos", NO asumas que es comportamiento esperado del sistema. Derivá al portal de solicitudes para que el equipo investigue el caso específico.

ESTRUCTURA OBLIGATORIA
1. Empezar siempre con "Buenos días." a secas. Sin apertura temática — ir directo a la respuesta.
2. Respuesta concreta basada en la documentación que obtuviste con las tools.
3. Si el ticket contiene más de un problema, separalos en párrafos numerados (1., 2., 3.) y respondé cada uno por separado. Un ticket típico mezcla 2 a 4 temas.
4. Si la pregunta es ambigua, dá respuesta condicional: "Si el caso es A, hagan X. Si es B, hagan Y." NO pidas que reformulen.
5. NO incluyas un cierre ni despedida. El sistema appendea el cierre estándar automáticamente.
6. Empezá siempre por la respuesta más simple y directa. Si hay una acción que resuelve el problema en un paso (ej. "no le asignen curso y no se declara"), poné eso primero antes de explicar alternativas más complejas.
7. Si el ticket tiene 4+ problemas distintos, priorizá los más críticos (bloqueos > configuración > consultas informativas) y sé breve en cada uno. Mejor una respuesta completa de 3 temas que una truncada de 5.

CONTEXTO INYECTADO POR EL SISTEMA
- Con cada consulta podés recibir los siguientes bloques antes del texto del usuario:
  - "Clave rama" y "Nivel inferido" del establecimiento (Inicial / Primario / Medio / Superior / Técnico / Desconocido). Usalos para interpretar qué flujos aplican — Inicial y Técnico tienen diferencias con Primario/Medio.
  - "Historial reciente del establecimiento" — hasta 3 interacciones previas con la misma clave rama. Si el usuario referencia un ticket anterior o una respuesta previa, usá este historial para dar continuidad.

REGLAS DE CONTENIDO
- Cuando el ticket mencione que un cargo o autoridad NO APARECE en el sistema, seguí este orden estricto:
  1. PRIMERO asumí que es un tema de timing: los nombramientos de autoridades tardan como mínimo 10 días hábiles en impactar en SINIGEP. Respondé con los dos escenarios de timing (menos de 10 días → esperar y consultar al supervisor pedagógico; más de 10 días → derivar al portal con los datos).
  2. Solo si el nombre del cargo suena inusual o el usuario pregunta explícitamente si el cargo existe, buscá en el catálogo para verificarlo internamente.
  3. REGLA CRÍTICA: el catálogo de cargos es una herramienta INTERNA tuya. NUNCA le digas al usuario si un cargo "es válido para nivel X" ni "figura en el catálogo oficial". Esa información la usás solo para decidir tu respuesta. Al usuario dale directamente la acción: esperar el plazo, consultar al supervisor, o derivar al portal.
- Cuando el ticket mencione roles, cargos o denominaciones docentes en OTRO contexto (asignación de horas, carga horaria, etc.), ahí sí buscá directamente en el catálogo de cargos por nivel.
- SELF-SERVICE PRIMERO: si el usuario pide que "el equipo haga algo" o "carguen datos" que el propio colegio puede hacer desde el sistema (editar plan de estudios, crear cursos, asignar docentes, dar de baja cargos, dar de baja docentes), siempre enseñales cómo hacerlo ellos mismos con la ruta exacta en el sistema. Solo derivá al portal de solicitudes cuando la operación genuinamente requiere intervención del equipo de datos (corrección de DNI, nombres, autoridades faltantes por migración, etc.).
- SELF-SERVICE incluye ALTA DE ALUMNOS: si el usuario envía datos de alumnos nuevos (nombre, DNI, fecha nacimiento), primero indícales cómo cargarlos desde Menú - Personas - Estudiantes - Agregar estudiante. Solo derivá al portal si se trata de CORRECCIONES a datos de alumnos ya cargados.
- SELF-SERVICE incluye BAJA DE DOCENTES: si el usuario dice "el docente ya no trabaja acá" o "quiero sacar a Juan de la nómina", indicales el flujo de baja desde Personas - Docentes - menú del docente - Dar de baja. Aclarar que NO se pueden dar de baja autoridades (director, vice, secretario), rol legal (Representante Legal, Apoderado Legal) ni cargos no frente a curso; en esos casos hay que desasignar el cargo bloqueante primero o derivar al portal si no se puede.
- Cuando el usuario diga que "no puede editar", "no le permite modificar" o "no encuentra cómo hacer cambios", cubrí TRES causas posibles: (1) el estado de la DJ puede estar bloqueando la edición (firmada/presentada), (2) la operación puede hacerse desde otro lugar del sistema según el tipo (ej. docentes a cargo desde el detalle del curso, asignaturas desde el curso, cargos desde Cargos y Horas), y (3) la edición de datos personales de alumnos/docentes depende de una feature flag por colegio — si el colegio no ve los campos editables, derivar al portal. Para datos de tipo y número de documento la edición siempre va por el portal.
- Si el usuario reporta que NO hay datos migrados (ni docentes, ni estudiantes, ni conducción), sugerí verificar que el trámite de DJ se haya iniciado desde la claverrama correcta del establecimiento. Es una causa frecuente de listados vacíos.
- GRUPOS (Educación Física / Idioma) solo existen en nivel Medio. Si el colegio no es de nivel Medio y pregunta por la pestaña Grupos, respondé directo: "la pestaña Grupos es exclusiva de nivel Medio; en su nivel esa funcionalidad no existe." En grupos de Ed. Física e Idioma tampoco se declara el cargo del docente — el cargo se registra en la asignatura o en Cargos y Horas.
- PLANES DE ESTUDIO ya no requieren declarar cantidad de años. Si el usuario pregunta "cómo defino la cantidad de años" o "no me deja elegir si son 5 o 6 años", respondé que ese campo fue quitado: ahora se eligen directamente los grados a usar dentro del plan.
- HORARIOS DE ASIGNATURAS antes de las 07:00 ya están habilitados. Si reportan que no los dejaba cargar horarios tempranos (06:55, 06:30, etc.), respondé que ya se puede cargar cualquier hora de inicio.
- VALIDACIÓN BLOQUEANTE en Presentación: si el usuario reporta que no puede presentar la DJ y el error dice "No hay horarios de funcionamiento configurados", derivá a Datos Institucionales para cargar el horario de funcionamiento de categoría General. Al guardar, el bloqueo desaparece.
- MOVIMIENTOS +/- en Presentación: si preguntan qué significa +X o -X en las tablas de cursos/cargos de la pestaña Presentación, explicá que es la diferencia entre altas y bajas de aporte (ej. +1 significa que hay un cargo más con aporte respecto del ciclo anterior, considerando altas menos bajas).
- PDF del trámite — reglas: si preguntan por qué un cargo frente a curso no aparece en el PDF, aclará que el PDF solo lista los cargos de "Cargos y Horas" (no-frente-a-curso) a propósito. Si preguntan por las autoridades legales del PDF, aclará que salen los Representantes Legales y Apoderados Legales declarados por sistema, NO los usuarios con rol Legal (los roles son permisos de acceso, no autoridades).
- BAJA DE DOCENTES: si un colegio quiere sacar a alguien de la nómina, el flujo es Personas - Docentes - menú del docente - Dar de baja. El docente queda archivado en la pestaña "Dados de baja". Si el docente tiene cargo de autoridad, rol legal o cargo no-frente-a-curso, la baja está bloqueada — primero hay que desasignar el cargo o, si no se puede desde UI, derivar al portal.
- WARNINGS DE HORAS (72hs, horas permitidas por cargo, horas excedidas) ya no se muestran desde el release 2026-04-14. Si un colegio pregunta por esos warnings o el modal de "horas permitidas", aclará que fueron removidos porque generaban falsos positivos. Cargar las horas reales del cargo sin preocuparse por esa advertencia.
- Cuando la respuesta mencione el importador CSV de estudiantes, incluí siempre el link al instructivo: https://docs.google.com/document/d/1AUVa99FuZzzwvpRPLLJusRyQInsxNaOQuvMAflJnU-o/edit?tab=t.0
- Cuando derives al portal de solicitudes, indicá la sección/opción exacta a seleccionar. Ejemplo: "seleccionen la opción 'Docentes - Dato incorrecto en docente'" o "'Estudiantes - Dato incorrecto en alumno'".
- Respondé SOLO con lo que está en la documentación que obtuviste con las tools + el contexto inyectado. NO inventes pasos, nombres de pantalla ni botones que no estén citados.
- Si el ticket es sobre alta o corrección de datos (autoridades, cargos faltantes, docentes duplicados, DNI o nombre invertidos, alumnos no migrados), derivá al portal: https://solicitudes-sinigep.up.railway.app/solicitud
- Si la documentación no cubre el caso, devolvé exactamente el token <<SIN_CONTEXTO>> y nada más. El sistema lo reemplaza por un mensaje de derivación a tutoría. NUNCA derives a "soporte técnico" como destino vago.
- Si el ticket NO es una consulta (es un agradecimiento sin pregunta, un "ok gracias", o un mensaje sin contenido procesable), devolvé exactamente el token <<NO_ES_CONSULTA>> y nada más. Los seguimientos y envíos de datos NO son <<NO_ES_CONSULTA>> — ver sección SEGUIMIENTOS.

ADJUNTOS
- Si el ticket menciona un archivo adjunto (Word, PDF, imagen, captura de pantalla, planilla Excel, etc.) como fuente principal de información, y NO hay una consulta concreta en el texto, respondé EXACTAMENTE esto y nada más:
  "[Adjunto no procesable] Estamos trabajando para procesar archivos adjuntos, pronto disponible. Requiere revisión manual del equipo."
- NO busques en la documentación, NO hagas tool calls. Solo esa respuesta exacta.
- Si además del adjunto hay una consulta concreta en el texto, respondé la consulta normalmente e ignorá el adjunto.

SEGUIMIENTOS
- Detectá seguimientos por estos patrones: "en relación al ticket N°", "en respuesta al ticket", "con referencia al ticket", "envío lo solicitado", "adjunto lo que me pidieron", "datos que me solicitaron", "como me pidieron", "le envío", "remito la documentación", "acá van los datos".
- Cuando detectes un seguimiento, NO uses <<NO_ES_CONSULTA>>. Los seguimientos merecen respuesta.
- REGLA CRÍTICA: la respuesta a un seguimiento debe ser de MÁXIMO 2-3 oraciones. NO busques en la documentación. NO hagas tool calls. NO expliques pasos que ya se dieron. Solo acusá recibo y, si corresponde, derivá al portal.
- Ejemplo completo de respuesta a seguimiento (usá este tono y largo):
  "Buenos días. Recibimos los datos enviados. Para que queden procesados, les pedimos ingresarlos en https://solicitudes-sinigep.up.railway.app/solicitud seleccionando la opción correspondiente."
- NUNCA escribas tu razonamiento interno en la respuesta. No empieces con "Este mensaje es un seguimiento..." ni "Voy a revisar...". Hablales directo.

LENGUAJE — PROHIBIDO USAR TÉRMINOS TÉCNICOS EN INGLÉS
- El usuario final es personal administrativo de escuelas, NO es técnico. Jamás uses terminología de desarrollo o diseño de interfaces en inglés.
- Traducciones obligatorias (si necesitás referir estos conceptos):
  - path → "ruta" o "recorrido" ("Seguí esta ruta en el sistema…")
  - toggle → "interruptor" o "activar/desactivar"
  - autocomplete → "autocompletar" o "se completa automáticamente"
  - slider → "barra deslizante" o "control deslizante"
  - bug → "error" o "falla"
  - dropdown → "lista desplegable" o "menú desplegable"
  - checkbox → "casilla de verificación" o "casilla"
  - input → "campo" o "campo de texto"
  - modal → "ventana" o "cuadro"
  - tab → "pestaña" o "solapa"
  - tooltip → "mensaje emergente"
  - link → "enlace" o "vínculo"
  - scroll → "desplazar" o "bajar/subir en la página"
  - click → "hacer clic" (no "clickear")
  - header → "encabezado"
  - sidebar → "panel lateral" o "menú lateral"
  - upload → "subir" o "cargar"
  - download → "descargar" o "bajar"
  - backend → "el sistema" o "la plataforma"
  - frontend → "la pantalla" o "la interfaz"
- Si la documentación de las tools usa un término técnico, traducilo antes de incluirlo en la respuesta.

NOTAS INTERNAS
- Cualquier texto entre //...// en la consulta es contexto del equipo. Usalo para entender el caso pero NO lo cites en la respuesta final.

LÍMITE Y COMPACTACIÓN
- La respuesta final (sin contar el cierre estándar que agrega el sistema) debe ser de MÁXIMO 1700 caracteres. El cierre añade ~110 chars para cumplir el tope de 2000 de BA Colaborativa.
- Técnicas obligatorias para mantenerte dentro del budget:
  - Una sola ruta o acción por subtema. Si hay dos formas de hacer algo, elegí la más directa y omití la alternativa.
  - Prohibido usar frases puente o de encuadre antes de la respuesta: "Te compartimos lo siguiente", "En relación a tu consulta", "Respondemos sus consultas", "Procedemos a indicar". Ir directo al primer paso accionable después de "Buenos días.".
  - Para timing de nombramientos, una sola oración con los dos escenarios (ej: "Si pasaron menos de 10 días hábiles, esperen y consulten al supervisor pedagógico; si pasaron más, deriven al portal de solicitudes con los datos del cargo.").
  - Si el ticket toca muchos temas, priorizá los bloqueos sobre las consultas informativas (regla ya enunciada en ESTRUCTURA OBLIGATORIA punto 7) y sé breve en cada bloque.
"""

CIERRE_LARGO = (
    "\n\n"
    "Si necesitan hacer seguimiento, referencien el número de este ticket "
    "en cualquier nuevo reclamo."
)

CIERRE_CORTO = (
    "\n\nPara seguimiento, referencien este ticket en nuevos reclamos."
)

SIN_CONTEXTO_BODY = (
    "Hola. Este caso no está cubierto por la documentación actual del equipo. "
    "Te pedimos que agendes una tutoría en vivo para revisarlo en conjunto."
)

# Tools expuestas al modelo (schema Anthropic)
TOOLS_SCHEMA = [
    {
        "name": "search_docs",
        "description": (
            "Busca en la documentación de SINIGEP y devuelve una lista de páginas "
            "relevantes con título, path y un snippet corto. Usá queries cortas y "
            "específicas. Hacé búsquedas separadas para temas distintos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Consulta de búsqueda en español, corta y específica.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_page",
        "description": (
            "Trae el contenido completo de una página de la documentación por su path. "
            "Usá esta tool cuando los snippets de search_docs no te alcancen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path de la página tal como lo devolvió search_docs.",
                }
            },
            "required": ["path"],
        },
    },
]

# --- Google Sheets ---

WORKSHEET_NAME = "Tickets - General"

COL_OBSERVACIONES = 21  # U — flag manual de producto (ej: "F" para forzar respuesta)
COL_PREGUNTA = 22       # V
COL_RESPUESTA = 25      # Y
COL_CONTROL = 26        # Z — respuesta final de producto (control humano)
MAX_PENDIENTES = 100


def encontrar_credenciales() -> Path:
    jsons = list(CREDENTIALS_DIR.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .json en {CREDENTIALS_DIR}. "
            "Descargá el service account key de Google Cloud Console."
        )
    return jsons[0]


def conectar_sheet(spreadsheet_url: str) -> gspread.Worksheet:
    creds_path = encontrar_credenciales()
    gc = gspread.service_account(filename=str(creds_path))
    sh = gc.open_by_url(spreadsheet_url)
    return sh.worksheet(WORKSHEET_NAME)


# --- Contexto extra: claverrama, catálogo de cargos, historial ---


def detectar_col_claverrama(header: list[str]) -> int | None:
    """
    Devuelve el índice 1-based de la columna claverrama, o None si no está.

    Hace matching por substring normalizado (case-insensitive) para tolerar
    prefijos como 'Educación - Claverama' y variantes de escritura
    (claverrama / claverama / clave rama / clave_rama / clave-rama).
    """
    for i, h in enumerate(header, start=1):
        norm = (h or "").strip().lower()
        if any(tok in norm for tok in HEADERS_CLAVERRAMA_TOKENS):
            return i
    return None


def obtener_historial_establecimiento(
    filas: list[list[str]],
    col_claverrama: int | None,
    clave_actual: str,
    fila_actual: int,
    max_items: int = MAX_HISTORIAL,
) -> list[dict]:
    """
    Tickets previos del mismo establecimiento (misma clave rama) que ya tengan
    respuesta. Prioriza la respuesta de producto (col Z) sobre la IA (col Y).
    """
    if not clave_actual or col_claverrama is None:
        return []

    clave_norm = normalizar(clave_actual)
    items: list[dict] = []

    for i, fila in enumerate(filas[1:], start=2):
        if i == fila_actual:
            continue
        clave_fila = fila[col_claverrama - 1] if len(fila) >= col_claverrama else ""
        if normalizar(clave_fila) != clave_norm:
            continue

        pregunta = fila[COL_PREGUNTA - 1] if len(fila) >= COL_PREGUNTA else ""
        respuesta = ""
        if len(fila) >= COL_CONTROL and fila[COL_CONTROL - 1]:
            respuesta = fila[COL_CONTROL - 1]
        elif len(fila) >= COL_RESPUESTA and fila[COL_RESPUESTA - 1]:
            respuesta = fila[COL_RESPUESTA - 1]

        if pregunta and respuesta:
            items.append({
                "pregunta": pregunta.strip()[:300],
                "respuesta": respuesta.strip()[:500],
            })
        if len(items) >= max_items:
            break

    return items


def construir_mensaje_inicial(
    pregunta: str,
    clave_rama: str,
    nivel: str,
    historial: list[dict],
) -> str:
    """Arma el primer mensaje user con todos los bloques de contexto inyectado."""
    bloques: list[str] = []

    if clave_rama:
        bloques.append(
            f"Clave rama del establecimiento: {clave_rama}\n"
            f"Nivel inferido: {nivel}"
        )

    if historial:
        lineas = []
        for idx, it in enumerate(historial, start=1):
            lineas.append(
                f"{idx}. Consulta previa: {it['pregunta']}\n"
                f"   Respuesta dada: {it['respuesta']}"
            )
        bloques.append(
            "Historial reciente del establecimiento:\n" + "\n\n".join(lineas)
        )

    bloques.append(f"Consulta del usuario a responder:\n\n{pregunta}")
    return "\n\n---\n\n".join(bloques)


# --- MCP tool wrappers ---

MCP_SEARCH_TOOL = "search_sinigep_documentación"
MCP_FILESYSTEM_TOOL = "query_docs_filesystem_sinigep_documentación"


async def mcp_search_docs(session: ClientSession, query: str) -> str:
    """Devuelve una lista compacta de resultados (title, path, snippet)."""
    resultado = await session.call_tool(
        name=MCP_SEARCH_TOOL,
        arguments={"query": query},
    )

    if not resultado.content:
        return "No hay resultados para esa búsqueda."

    vistos: set[str] = set()
    items: list[dict] = []
    for raw in resultado.content:
        if not (hasattr(raw, "text") and raw.text):
            continue
        texto = re.sub(r"<[^>]+>", "", raw.text).strip()
        path_match = re.search(r"Page: (.+)", texto)
        path = path_match.group(1).strip() if path_match else ""
        if path and path in vistos:
            continue
        if path:
            vistos.add(path)
        title_match = re.search(r"Title: (.+)", texto)
        title = title_match.group(1).strip() if title_match else path or "(sin título)"
        snippet = texto[:400].replace("\n", " ")
        items.append({"title": title, "path": path, "snippet": snippet})

    if not items:
        return "No hay resultados estructurados. Probá con otra query."

    return json.dumps(items[:6], ensure_ascii=False, indent=2)


def _path_a_mdx(path: str) -> str:
    """
    Normaliza un path devuelto por search_docs al formato que espera el
    filesystem virtual del MCP: prefijo '/' y sufijo '.mdx'.
    """
    p = (path or "").strip()
    if not p:
        return ""
    if not p.startswith("/"):
        p = "/" + p
    if not p.endswith(".mdx"):
        p = p + ".mdx"
    return p


async def mcp_get_page(session: ClientSession, path: str) -> str:
    """Trae el contenido completo de una página usando `cat` sobre el filesystem virtual."""
    mdx_path = _path_a_mdx(path)
    if not mdx_path:
        return "Path vacío."

    try:
        pagina = await session.call_tool(
            name=MCP_FILESYSTEM_TOOL,
            arguments={"command": f"cat {mdx_path}"},
        )
    except Exception as e:
        return f"Error obteniendo la página {path}: {e}"

    if not pagina.content:
        return f"La página {path} está vacía."

    bloques = [b.text for b in pagina.content if hasattr(b, "text") and b.text]
    return "\n".join(bloques) if bloques else f"No se pudo leer {path}."


async def ejecutar_tool(session: ClientSession, nombre: str, inputs: dict) -> str:
    if nombre == "search_docs":
        return await mcp_search_docs(session, inputs.get("query", ""))
    if nombre == "get_page":
        return await mcp_get_page(session, inputs.get("path", ""))
    return f"Tool desconocida: {nombre}"


# --- RAG iterativo con tool use ---


async def responder_con_rag_iterativo(
    client: AsyncAnthropic,
    session: ClientSession,
    pregunta: str,
    clave_rama: str = "",
    nivel: str = NIVEL_DESCONOCIDO,
    historial_establecimiento: list[dict] | None = None,
    model_override: str | None = None,
) -> tuple[str, int]:
    """
    Loop de agente con tool use. Devuelve (respuesta_cruda, tool_calls_usadas).

    historial se instancia fresco cada llamada para garantizar aislamiento
    entre tickets — ningún contexto de tickets previos cruza acá.

    model_override: si se pasa, usa ese modelo en lugar del MODEL por defecto.
    """
    model = model_override or MODEL

    mensaje_inicial = construir_mensaje_inicial(
        pregunta=pregunta,
        clave_rama=clave_rama,
        nivel=nivel,
        historial=historial_establecimiento or [],
    )

    historial: list[dict] = [{"role": "user", "content": mensaje_inicial}]
    tool_calls = 0

    for _ in range(MAX_RAG_ITERATIONS):
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS_SCHEMA,
            messages=historial,
        )

        if response.stop_reason == "end_turn":
            textos = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return "\n".join(textos).strip(), tool_calls

        if response.stop_reason != "tool_use":
            textos = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return "\n".join(textos).strip(), tool_calls

        # Appendear el turno del asistente (con los tool_use blocks) al historial
        historial.append({"role": "assistant", "content": response.content})

        tool_results_content = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_calls += 1
            try:
                resultado = await ejecutar_tool(session, block.name, dict(block.input))
            except Exception as e:
                resultado = f"Error ejecutando {block.name}: {e}"
            tool_results_content.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": resultado[:8000],
            })

        historial.append({"role": "user", "content": tool_results_content})

    historial.append({
        "role": "user",
        "content": (
            "Llegaste al límite de búsquedas. Respondé ahora con lo que tengas, "
            "respetando todas las reglas del prompt. Si no tenés contexto suficiente "
            "devolvé exactamente <<SIN_CONTEXTO>>."
        ),
    })
    final = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=historial,
    )
    textos = [b.text for b in final.content if getattr(b, "type", None) == "text"]
    return "\n".join(textos).strip(), tool_calls


# --- Post-procesado ---

RE_MARKERS_INTERNOS = re.compile(
    r"\[SIN INFO EN DOCS\]|\[NO ES CONSULTA[^\]]*\]|<<[^>]*>>",
    flags=re.IGNORECASE,
)
RE_MARKDOWN = re.compile(r"\*\*|__|##+|\*(?=\S)|(?<=\S)\*")
RE_ANGULOS = re.compile(r"[<>]")
RE_EXCLAMACION = re.compile(r"[¡!]")
RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
RE_BLANKLINES = re.compile(r"\n{3,}")
RE_THINKING_LEAK = re.compile(r"^.*?(?=Buenos d[ií]as[\.\,\s])", flags=re.DOTALL | re.IGNORECASE)


COMPRESSION_SYSTEM = (
    "Sos un editor que reescribe respuestas de soporte de SINIGEP para que "
    "cumplan el límite de 1700 caracteres. Tu única tarea es compactar el texto "
    "que recibís preservando 100% del contenido accionable.\n\n"
    "REGLAS:\n"
    "- Preservá TODAS las acciones, rutas (Menú - Personas - …), botones, "
    "pasos numerados (1., 2., 3.) y links exactos (no acortes URLs).\n"
    "- Eliminá solo frases transicionales, repeticiones y explicaciones "
    "redundantes.\n"
    "- Mantené el saludo 'Buenos días.' al inicio.\n"
    "- NO agregues cierre ni despedida (ni 'Quedamos a disposición', ni "
    "'Muchas gracias', ni 'Saludos').\n"
    "- Plain text. Prohibido: exclamaciones (! ¡), markdown (**, __, ##), "
    "emojis, caracteres < > { }.\n"
    "- Hablá siempre en plural ('ustedes', 'les'), nunca en singular ni en "
    "tercera persona ('la institución').\n"
    "- Sin términos técnicos en inglés (path → ruta, dropdown → menú "
    "desplegable, click → hacer clic, etc.).\n"
    "- Devolvé SOLO el texto reescrito, sin explicar qué hiciste."
)


async def comprimir_respuesta(
    cuerpo: str,
    client: AsyncAnthropic,
    model: str,
) -> str:
    """
    Reescribe `cuerpo` para que entre en 1700 chars preservando acciones y links.
    Devuelve el texto comprimido (puede aún exceder si el modelo no logró acortar).
    """
    user_msg = (
        f"Reescribí esta respuesta en máximo 1700 caracteres siguiendo todas las "
        f"reglas del sistema. La respuesta original tiene {len(cuerpo)} chars:\n\n"
        f"---\n{cuerpo}\n---"
    )
    resp = await client.messages.create(
        model=model,
        max_tokens=1600,
        system=COMPRESSION_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    textos = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(textos).strip()


def _limpiar_formato(texto: str) -> str:
    """Aplica las regex de limpieza al texto en bruto."""
    limpio = RE_MARKERS_INTERNOS.sub("", texto)
    limpio = RE_MARKDOWN.sub("", limpio)
    limpio = RE_ANGULOS.sub("", limpio)
    limpio = RE_EXCLAMACION.sub("", limpio)
    limpio = RE_EMOJI.sub("", limpio)
    return RE_BLANKLINES.sub("\n\n", limpio).strip()


def _con_cierre_dentro_de_limite(limpio: str) -> str | None:
    """Intenta appendear cierre largo o corto. Devuelve None si ninguno entra."""
    final_largo = limpio + CIERRE_LARGO
    if len(final_largo) <= MAX_RESPUESTA_CHARS:
        return final_largo
    final_corto = limpio + CIERRE_CORTO
    if len(final_corto) <= MAX_RESPUESTA_CHARS:
        return final_corto
    return None


async def validar_y_postprocesar(
    respuesta_cruda: str,
    client: AsyncAnthropic,
    model: str,
) -> tuple[str, str]:
    """
    Aplica limpieza, validaciones de formato, append del cierre y check de longitud.
    Si el cuerpo + cierre supera MAX_RESPUESTA_CHARS, hace un retry de compresión
    con el LLM y reintenta una vez. Solo marca EXCEDE_LIMITE si tras el retry
    sigue sin entrar.

    Devuelve (respuesta_final, estado) donde estado ∈
    {OK, COMPRIMIDA, NO_CONSULTA, SIN_CONTEXTO, EXCEDE_LIMITE, VACIA}.
    """
    texto = (respuesta_cruda or "").strip()

    if not texto:
        return ("[RESPUESTA VACIA — revisar manualmente]", "VACIA")

    if TOKEN_NO_CONSULTA in texto:
        return ("[NO ES CONSULTA — revisar manualmente]", "NO_CONSULTA")

    if TOKEN_SIN_CONTEXTO in texto or texto == "SIN_CONTEXTO":
        cuerpo = (
            "[SIN CONTEXTO EN DOCS — revisar manualmente]\n\n"
            + SIN_CONTEXTO_BODY
            + CIERRE_LARGO
        )
        return (cuerpo, "SIN_CONTEXTO")

    # Cortar "thinking" que el modelo a veces escribe antes de "Buenos días."
    m = re.search(r"Buenos d[ií]as[\.\,\s]", texto, flags=re.IGNORECASE)
    if m and m.start() > 0:
        texto = texto[m.start():].strip()

    limpio = _limpiar_formato(texto)

    # Primer intento: cierre adaptativo con el cuerpo original.
    final = _con_cierre_dentro_de_limite(limpio)
    if final is not None:
        return (final, "OK")

    # Overflow → retry de compresión con el LLM.
    try:
        comprimido_crudo = await comprimir_respuesta(limpio, client, model)
    except Exception as e:
        print(f"  WARN: comprimir_respuesta falló ({e}), marcando EXCEDE_LIMITE.")
        marcado = f"[EXCEDE LIMITE {len(limpio + CIERRE_CORTO)} chars]\n\n{limpio + CIERRE_CORTO}"
        return (marcado, "EXCEDE_LIMITE")

    comprimido = _limpiar_formato(comprimido_crudo)
    final = _con_cierre_dentro_de_limite(comprimido)
    if final is not None:
        return (final, "COMPRIMIDA")

    # Tras compresión sigue sin entrar — marcamos para revisión manual.
    marcado = f"[EXCEDE LIMITE {len(comprimido + CIERRE_CORTO)} chars tras compresión]\n\n{comprimido + CIERRE_CORTO}"
    return (marcado, "EXCEDE_LIMITE")


# --- Procesamiento ---


async def procesar_sheet(
    session: ClientSession,
    anthropic_client: AsyncAnthropic,
    worksheet: gspread.Worksheet,
    force_regen: bool,
    from_row: int = 2,
    regen_pattern: str | None = None,
    obs_flag: str | None = None,
):
    filas = worksheet.get_all_values()
    total = len(filas) - 1
    procesadas = 0
    contadores: dict[str, int] = {}

    col_claverrama = detectar_col_claverrama(filas[0] if filas else [])
    if col_claverrama is None:
        print("WARN: no se encontró columna clave rama en el header. Siguiendo sin clave rama ni historial.")
    else:
        print(f"Clave rama detectada en columna {col_claverrama}.")

    print(f"\nTotal de filas (sin header): {total}")
    print(
        f"Modo force-regen: {force_regen}  from_row={from_row}  "
        f"regen_pattern={regen_pattern!r}  obs_flag={obs_flag!r}"
    )

    for i, fila in enumerate(filas[1:], start=2):
        if procesadas >= MAX_PENDIENTES:
            print(f"\nLímite de {MAX_PENDIENTES} alcanzado, frenando.")
            break

        if i < from_row:
            continue

        pregunta = fila[COL_PREGUNTA - 1] if len(fila) >= COL_PREGUNTA else ""
        control = fila[COL_CONTROL - 1] if len(fila) >= COL_CONTROL else ""
        respuesta_existente = fila[COL_RESPUESTA - 1] if len(fila) >= COL_RESPUESTA else ""
        observaciones = fila[COL_OBSERVACIONES - 1] if len(fila) >= COL_OBSERVACIONES else ""

        if not pregunta or control:
            continue

        if obs_flag is not None:
            # Solo procesar filas donde col U (Observaciones) sea exactamente el flag.
            if observaciones.strip().upper() != obs_flag.strip().upper():
                continue
            # Bajo --obs-flag respetamos "solo si no tiene respuesta" salvo --force-regen.
            if respuesta_existente and not force_regen:
                continue
        elif regen_pattern is not None:
            # Modo selectivo: solo regenerar filas cuya respuesta actual contenga el patrón.
            if regen_pattern not in respuesta_existente:
                continue
        elif respuesta_existente and not force_regen:
            continue

        clave_rama = ""
        if col_claverrama is not None and len(fila) >= col_claverrama:
            clave_rama = fila[col_claverrama - 1].strip()
        nivel = inferir_nivel(clave_rama) if clave_rama else NIVEL_DESCONOCIDO

        historial = obtener_historial_establecimiento(
            filas=filas,
            col_claverrama=col_claverrama,
            clave_actual=clave_rama,
            fila_actual=i,
        )

        print(
            f"[{procesadas + 1}/{MAX_PENDIENTES}] Fila {i}: "
            f"clave={clave_rama or '—'} nivel={nivel} historial={len(historial)}"
        )
        print(f"  Consulta: {pregunta[:80]}...")

        last_error = None
        respuesta_cruda = None
        tool_calls = 0
        for intento in range(4):
            try:
                respuesta_cruda, tool_calls = await responder_con_rag_iterativo(
                    client=anthropic_client,
                    session=session,
                    pregunta=pregunta,
                    clave_rama=clave_rama,
                    nivel=nivel,
                    historial_establecimiento=historial,
                )
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                if ("429" in err_str or "rate_limit" in err_str) and intento < 3:
                    wait = 65 * (intento + 1)
                    print(f"  Rate limit (intento {intento + 1}/3), esperando {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"  ERROR generando respuesta: {e}")
                contadores["ERROR"] = contadores.get("ERROR", 0) + 1
                procesadas += 1
                break
        else:
            print(f"  ERROR tras 3 retries: {last_error}")
            contadores["ERROR"] = contadores.get("ERROR", 0) + 1
            procesadas += 1
            continue

        if respuesta_cruda is None:
            continue

        respuesta_final, estado = await validar_y_postprocesar(
            respuesta_cruda, anthropic_client, MODEL,
        )
        contadores[estado] = contadores.get(estado, 0) + 1
        print(f"  → estado={estado}  tool_calls={tool_calls}  chars={len(respuesta_final)}")

        worksheet.update_cell(i, COL_RESPUESTA, respuesta_final)

        procesadas += 1
        await asyncio.sleep(1)  # rate-limit Google Sheets

    print(f"\nResultado: {procesadas} procesadas. Estados: {contadores}")


# --- Main ---


async def main():
    parser = argparse.ArgumentParser(description="Responder tickets de SINIGEP con RAG iterativo.")
    parser.add_argument("sheet_url", help="URL del Google Sheet")
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Reescribir respuestas existentes en col Y (por defecto se saltan).",
    )
    parser.add_argument(
        "--from-row",
        type=int,
        default=2,
        help="Empezar desde esta fila (1-based, header=1). Default: 2.",
    )
    parser.add_argument(
        "--regen-pattern",
        type=str,
        default=None,
        help=(
            "Solo regenerar filas cuya respuesta actual contenga este substring. "
            "Ej: --regen-pattern '[EXCEDE LIMITE'. Tiene precedencia sobre --force-regen."
        ),
    )
    parser.add_argument(
        "--obs-flag",
        type=str,
        default=None,
        help=(
            "Procesar SOLO filas cuya col U (Observaciones) coincida exactamente con este "
            "valor (case-insensitive). Por defecto respeta 'no pisar respuesta existente' "
            "salvo que se combine con --force-regen. Ej: --obs-flag F"
        ),
    )
    args = parser.parse_args()

    print("Conectando a Google Sheets...")
    worksheet = conectar_sheet(args.sheet_url)
    filas = worksheet.get_all_values()
    print(f"Sheet: {worksheet.title} — {len(filas) - 1} filas")

    anthropic_client = AsyncAnthropic()
    print("Claude API OK.")

    print(f"Conectando al MCP: {MCP_URL}...")
    async with streamablehttp_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"MCP conectado. Tools disponibles: {tool_names}")

            if MCP_SEARCH_TOOL not in tool_names:
                print(f"ERROR: el MCP no expone {MCP_SEARCH_TOOL}. Abortando.")
                sys.exit(1)
            if MCP_FILESYSTEM_TOOL not in tool_names:
                print(f"ERROR: el MCP no expone {MCP_FILESYSTEM_TOOL}. Abortando.")
                sys.exit(1)

            await procesar_sheet(
                session,
                anthropic_client,
                worksheet,
                force_regen=args.force_regen,
                from_row=args.from_row,
                regen_pattern=args.regen_pattern,
                obs_flag=args.obs_flag,
            )

    print("Listo.")


if __name__ == "__main__":
    asyncio.run(main())
