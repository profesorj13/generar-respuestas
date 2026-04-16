"""
Núcleo RAG reutilizable: prompt + tools + wrappers MCP + loop de tool use.

Usado por:
- responder_tickets.py (flujo Google Sheets → BA Colaborativa)
- slack-bot/ (flujo Slack app_mention)

No contiene I/O específico de ningún canal. Recibe todo por parámetro.
"""

import json
import re
from pathlib import Path

from anthropic import AsyncAnthropic
from mcp import ClientSession


# --- Configuración MCP / modelo ---

MCP_URL = "https://educabot-d7d4a6e0.mintlify.app/mcp"
MCP_SEARCH_TOOL = "search_sinigep_documentación"
MCP_FILESYSTEM_TOOL = "query_docs_filesystem_sinigep_documentación"

MODEL = "claude-sonnet-4-6"
MAX_RAG_ITERATIONS = 8

TOKEN_NO_CONSULTA = "<<NO_ES_CONSULTA>>"
TOKEN_SIN_CONTEXTO = "<<SIN_CONTEXTO>>"


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
- Saludo: "Hola," siempre (con coma, minúscula después). Nunca "Buenos días", "Buenas tardes" ni variantes.
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
1. Empezar siempre con "Hola," a secas. Sin apertura temática — ir directo a la respuesta.
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
- DATOS PERSONALES DE DOCENTE O ALUMNO (nombre, apellido, fecha de nacimiento, país, género), seguí este orden estricto:
  1. PRIMERO asumí que desde el release 2026-04-14 la edición está habilitada desde la plataforma en Personas - Docentes (o Estudiantes) - Ver detalles - botón Editar. Antes de decidir si derivás al portal, hacé search_docs("editar docente") o search_docs("editar estudiante") y confirmá el flujo contra la doc. Pasale al colegio esa ruta como primer paso.
  2. SOLO derivá al portal si se cumple alguna de estas condiciones: (a) el campo a corregir es tipo o número de documento (esos siempre van por portal), o (b) el colegio dice explícitamente que NO ve los campos editables (el feature flag de edición está apagado para ese establecimiento). En (b), igual dales la ruta UI primero y el portal como fallback condicional.
  3. REGLA CRÍTICA: nunca derives por defecto ante un pedido de "corrección de nombre", "corrección de apellido", "apellido vacío", "docente sin apellido" o similar. El default es enseñar la ruta de edición desde la plataforma. Derivar al portal sin ofrecer primero la ruta UI es una respuesta incorrecta.
- SELF-SERVICE PRIMERO: si el usuario pide que "el equipo haga algo" o "carguen datos" que el propio colegio puede hacer desde el sistema (editar plan de estudios, crear cursos, asignar docentes, dar de baja cargos, dar de baja docentes, editar datos personales de docentes o alumnos), siempre enseñales cómo hacerlo ellos mismos con la ruta exacta en el sistema. Solo derivá al portal de solicitudes cuando la operación genuinamente requiere intervención del equipo de datos (autoridades faltantes por migración, docentes duplicados, alumnos no migrados, tipo/número de documento).
- SELF-SERVICE incluye ALTA DE ALUMNOS: si el usuario envía datos de alumnos nuevos (nombre, DNI, fecha nacimiento), primero indícales cómo cargarlos desde Menú - Personas - Estudiantes - Agregar estudiante. Si se trata de CORRECCIONES a datos de alumnos ya cargados, aplicá la regla maestra de DATOS PERSONALES (edición desde UI como default, portal solo si aplica la excepción).
- SELF-SERVICE incluye BAJA DE DOCENTES: si el usuario dice "el docente ya no trabaja acá" o "quiero sacar a Juan de la nómina", indicales el flujo de baja desde Personas - Docentes - menú del docente - Dar de baja. Aclarar que NO se pueden dar de baja autoridades (director, vice, secretario), rol legal (Representante Legal, Apoderado Legal) ni cargos no frente a curso; en esos casos hay que desasignar el cargo bloqueante primero o derivar al portal si no se puede.
- Cuando el usuario diga que "no puede editar", "no le permite modificar" o "no encuentra cómo hacer cambios" para datos DE CONFIGURACIÓN del trámite (cursos, asignaturas, cargos, horarios, planes), cubrí DOS causas posibles: (1) el estado de la DJ puede estar bloqueando la edición (firmada/presentada), (2) la operación puede hacerse desde otro lugar del sistema según el tipo (ej. docentes a cargo desde el detalle del curso, asignaturas desde el curso, cargos desde Cargos y Horas). Si el pedido de edición es sobre datos personales de docente o alumno, NO uses este bullet — aplicá la regla maestra de DATOS PERSONALES.
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
- Si el ticket es sobre alta o corrección de datos ESTRUCTURALES (autoridades faltantes por migración, cargos faltantes, docentes duplicados, alumnos no migrados, tipo o número de documento), derivá al portal: https://solicitudes-sinigep.up.railway.app/solicitud. Los datos personales editables desde plataforma (nombre, apellido, fecha de nacimiento, país, género de docente o alumno) NO caen en este bullet — esos siguen la regla maestra de DATOS PERSONALES.
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
  "Hola, recibimos los datos enviados. Para que queden procesados, les pedimos ingresarlos en https://solicitudes-sinigep.up.railway.app/solicitud seleccionando la opción correspondiente."
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
- La respuesta final (sin contar el cierre estándar que agrega el sistema) debe ser de MÁXIMO 1600 caracteres. El cierre añade ~220 chars (incluye oferta de tutoría con link de calendario) para cumplir el tope de 2000 de BA Colaborativa.
- Técnicas obligatorias para mantenerte dentro del budget:
  - Una sola ruta o acción por subtema. Si hay dos formas de hacer algo, elegí la más directa y omití la alternativa.
  - Prohibido usar frases puente o de encuadre antes de la respuesta: "Te compartimos lo siguiente", "En relación a tu consulta", "Respondemos sus consultas", "Procedemos a indicar". Ir directo al primer paso accionable después de "Hola,".
  - Para timing de nombramientos, una sola oración con los dos escenarios (ej: "Si pasaron menos de 10 días hábiles, esperen y consulten al supervisor pedagógico; si pasaron más, deriven al portal de solicitudes con los datos del cargo.").
  - Si el ticket toca muchos temas, priorizá los bloqueos sobre las consultas informativas (regla ya enunciada en ESTRUCTURA OBLIGATORIA punto 7) y sé breve en cada bloque.
"""


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


# --- Contexto inyectado ---


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


# --- Adapter para doc source (MCP remoto o filesystem local) ---


class DocsAdapter:
    """Interfaz para consultar la doc. Implementaciones: MCP remoto o filesystem local."""

    async def search(self, query: str) -> str:
        raise NotImplementedError

    async def get_page(self, path: str) -> str:
        raise NotImplementedError


class MCPAdapter(DocsAdapter):
    """Usa las funciones mcp_search_docs / mcp_get_page existentes."""

    def __init__(self, session: ClientSession):
        self.session = session

    async def search(self, query: str) -> str:
        return await mcp_search_docs(self.session, query)

    async def get_page(self, path: str) -> str:
        return await mcp_get_page(self.session, path)


class FilesystemAdapter(DocsAdapter):
    """Lee .mdx directamente del filesystem. Útil para testear cambios locales."""

    MAX_RESULTS = 6
    SNIPPET_RADIUS = 200

    def __init__(self, docs_path: str | Path):
        self.docs_path = Path(docs_path).resolve()
        if not self.docs_path.is_dir():
            raise ValueError(f"docs_path no es un directorio: {self.docs_path}")

    def _titulo(self, texto: str, fallback: str) -> str:
        for linea in texto.splitlines():
            limpio = linea.strip()
            if limpio.startswith("# "):
                return limpio[2:].strip()
        return fallback

    def _path_relativo(self, archivo: Path) -> str:
        rel = archivo.relative_to(self.docs_path).as_posix()
        return rel[:-4] if rel.endswith(".mdx") else rel

    async def search(self, query: str) -> str:
        if not query.strip():
            return "Query vacía."

        tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) >= 3]
        if not tokens:
            return "Query sin términos útiles (mínimo 3 chars por token)."

        scored: list[tuple[int, Path, str]] = []
        for archivo in self.docs_path.rglob("*.mdx"):
            try:
                contenido = archivo.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            contenido_lower = contenido.lower()
            score = sum(contenido_lower.count(tok) for tok in tokens)
            if score == 0:
                continue
            scored.append((score, archivo, contenido))

        if not scored:
            return "No hay resultados para esa búsqueda."

        scored.sort(key=lambda x: x[0], reverse=True)

        items: list[dict] = []
        for score, archivo, contenido in scored[: self.MAX_RESULTS]:
            contenido_lower = contenido.lower()
            primer_match = len(contenido)
            for tok in tokens:
                pos = contenido_lower.find(tok)
                if pos != -1 and pos < primer_match:
                    primer_match = pos
            inicio = max(0, primer_match - self.SNIPPET_RADIUS)
            fin = min(len(contenido), primer_match + self.SNIPPET_RADIUS)
            snippet = contenido[inicio:fin].replace("\n", " ").strip()
            items.append({
                "title": self._titulo(contenido, archivo.stem),
                "path": self._path_relativo(archivo),
                "snippet": snippet,
            })

        return json.dumps(items, ensure_ascii=False, indent=2)

    async def get_page(self, path: str) -> str:
        p = (path or "").strip()
        if not p:
            return "Path vacío."
        if p.startswith("/"):
            p = p[1:]
        if not p.endswith(".mdx"):
            p = p + ".mdx"

        archivo = (self.docs_path / p).resolve()
        try:
            archivo.relative_to(self.docs_path)
        except ValueError:
            return f"Path fuera del docs_path: {path}"

        if not archivo.is_file():
            return f"No existe la página: {path}"

        try:
            return archivo.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            return f"Error leyendo {path}: {e}"


async def ejecutar_tool_adapter(adapter: DocsAdapter, nombre: str, inputs: dict) -> str:
    if nombre == "search_docs":
        return await adapter.search(inputs.get("query", ""))
    if nombre == "get_page":
        return await adapter.get_page(inputs.get("path", ""))
    return f"Tool desconocida: {nombre}"


# --- RAG iterativo con tool use ---


async def responder_con_rag_iterativo(
    client: AsyncAnthropic,
    session: ClientSession | None = None,
    pregunta: str = "",
    clave_rama: str = "",
    nivel: str = "Desconocido",
    historial_establecimiento: list[dict] | None = None,
    model_override: str | None = None,
    system_prompt: str | None = None,
    max_iterations: int | None = None,
    adapter: DocsAdapter | None = None,
) -> tuple[str, int]:
    """
    Loop de agente con tool use. Devuelve (respuesta_cruda, tool_calls_usadas).

    historial se instancia fresco cada llamada para garantizar aislamiento
    entre invocaciones — ningún contexto previo cruza acá.

    Parámetros:
    - system_prompt: si se pasa, overridea SYSTEM_PROMPT por defecto.
    - model_override: overridea MODEL.
    - max_iterations: overridea MAX_RAG_ITERATIONS.
    - adapter: fuente de la doc. Si es None, se construye MCPAdapter(session).
      Los callers existentes pasan `session` y siguen funcionando igual.
    """
    if adapter is None:
        if session is None:
            raise ValueError("Debe pasarse 'session' o 'adapter'.")
        adapter = MCPAdapter(session)

    model = model_override or MODEL
    sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    iter_limit = max_iterations or MAX_RAG_ITERATIONS

    mensaje_inicial = construir_mensaje_inicial(
        pregunta=pregunta,
        clave_rama=clave_rama,
        nivel=nivel,
        historial=historial_establecimiento or [],
    )

    historial: list[dict] = [{"role": "user", "content": mensaje_inicial}]
    tool_calls = 0

    for _ in range(iter_limit):
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=sys_prompt,
            tools=TOOLS_SCHEMA,
            messages=historial,
        )

        if response.stop_reason == "end_turn":
            textos = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return "\n".join(textos).strip(), tool_calls

        if response.stop_reason != "tool_use":
            textos = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return "\n".join(textos).strip(), tool_calls

        historial.append({"role": "assistant", "content": response.content})

        tool_results_content = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_calls += 1
            try:
                resultado = await ejecutar_tool_adapter(adapter, block.name, dict(block.input))
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
        system=sys_prompt,
        messages=historial,
    )
    textos = [b.text for b in final.content if getattr(b, "type", None) == "text"]
    return "\n".join(textos).strip(), tool_calls
