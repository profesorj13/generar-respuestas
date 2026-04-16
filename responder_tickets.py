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
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import gspread
from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from claverrama import inferir_nivel, normalizar, NIVEL_DESCONOCIDO
from core_rag import (
    MCP_URL,
    MCP_SEARCH_TOOL,
    MCP_FILESYSTEM_TOOL,
    MODEL,
    TOKEN_NO_CONSULTA,
    TOKEN_SIN_CONTEXTO,
    responder_con_rag_iterativo,
)

# --- Configuración ---

load_dotenv(Path(__file__).parent / ".env")

CREDENTIALS_DIR = Path(__file__).parent / "credentials"

MAX_RESPUESTA_CHARS = 2000
MAX_HISTORIAL = 3  # tickets previos del mismo establecimiento a inyectar

HEADERS_CLAVERRAMA_TOKENS = ("claverrama", "claverama", "clave rama", "clave_rama", "clave-rama")

CIERRE_LARGO = (
    "\n\n"
    "Si necesitan hacer seguimiento, referencien el número de este ticket "
    "en cualquier nuevo reclamo."
    "\n\n"
    "Si aún no pueden resolver, pueden agendar una tutoría desde "
    "https://calendar.app.google/3jQBnwQ5SQTuGvT59"
)

CIERRE_CORTO = (
    "\n\nPara seguimiento, referencien este ticket en nuevos reclamos."
    "\n\nSi aún no pueden resolver, agenden una tutoría desde "
    "https://calendar.app.google/3jQBnwQ5SQTuGvT59"
)

SIN_CONTEXTO_BODY = (
    "Hola. Este caso no está cubierto por la documentación actual del equipo. "
    "Les pedimos agendar una tutoría en vivo para revisarlo en conjunto desde "
    "https://calendar.app.google/3jQBnwQ5SQTuGvT59"
)

# --- Google Sheets ---

WORKSHEET_NAME = "Tickets - General"

COL_OBSERVACIONES = 21  # U — flag manual de producto (ej: "F" para forzar respuesta)
COL_PREGUNTA = 22       # V
COL_RESPUESTA = 25      # Y
COL_CONTROL = 26        # Z — respuesta final de producto (control humano)
COL_CERRADA_PRODUCTO = 27  # AA — "Respuesta cerrada producto" (Sí/No)
COL_RESPONDIDO = 32        # AF — "Respondido" (Sí/No)
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
RE_THINKING_LEAK = re.compile(r"^.*?(?=Hola[\.\,\s])", flags=re.DOTALL | re.IGNORECASE)


COMPRESSION_SYSTEM = (
    "Sos un editor que reescribe respuestas de soporte de SINIGEP para que "
    "cumplan el límite de 1700 caracteres. Tu única tarea es compactar el texto "
    "que recibís preservando 100% del contenido accionable.\n\n"
    "REGLAS:\n"
    "- Preservá TODAS las acciones, rutas (Menú - Personas - …), botones, "
    "pasos numerados (1., 2., 3.) y links exactos (no acortes URLs).\n"
    "- Eliminá solo frases transicionales, repeticiones y explicaciones "
    "redundantes.\n"
    "- Mantené el saludo 'Hola,' al inicio.\n"
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

    # Cortar "thinking" que el modelo a veces escribe antes del saludo.
    m = re.search(r"Hola[\.\,\s]", texto, flags=re.IGNORECASE)
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
    max_pendientes: int = MAX_PENDIENTES,
    only_row: int | None = None,
    pending_only: bool = False,
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
        if procesadas >= max_pendientes:
            print(f"\nLímite de {max_pendientes} alcanzado, frenando.")
            break

        if only_row is not None and i != only_row:
            continue
        if i < from_row:
            continue

        pregunta = fila[COL_PREGUNTA - 1] if len(fila) >= COL_PREGUNTA else ""
        control = fila[COL_CONTROL - 1] if len(fila) >= COL_CONTROL else ""
        respuesta_existente = fila[COL_RESPUESTA - 1] if len(fila) >= COL_RESPUESTA else ""
        observaciones = fila[COL_OBSERVACIONES - 1] if len(fila) >= COL_OBSERVACIONES else ""

        if not pregunta or control:
            continue

        if pending_only:
            cerrada = fila[COL_CERRADA_PRODUCTO - 1].strip().lower() if len(fila) >= COL_CERRADA_PRODUCTO else ""
            respondido = fila[COL_RESPONDIDO - 1].strip().lower() if len(fila) >= COL_RESPONDIDO else ""
            if cerrada != "no" or respondido != "no":
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
        worksheet.update_cell(i, COL_CERRADA_PRODUCTO, "Sí")

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
    parser.add_argument(
        "--max",
        dest="max_pendientes",
        type=int,
        default=MAX_PENDIENTES,
        help=f"Máximo de filas a procesar en la corrida. Default: {MAX_PENDIENTES}.",
    )
    parser.add_argument(
        "--only-row",
        type=int,
        default=None,
        help="Procesar EXCLUSIVAMENTE esta fila (1-based). Ignora --from-row.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help=(
            "Solo procesar filas donde col AA ('Respuesta cerrada producto') = 'No' "
            "Y col AF ('Respondido') = 'No'. Combinable con --force-regen."
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
                max_pendientes=args.max_pendientes,
                only_row=args.only_row,
                pending_only=args.pending_only,
            )

    print("Listo.")


if __name__ == "__main__":
    asyncio.run(main())
