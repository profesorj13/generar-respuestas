"""
Responder tickets de SINIGEP automáticamente.

Lee preguntas de un Google Sheet, busca en la documentación de Mintlify
via MCP, genera respuestas con Claude API, y las escribe de vuelta en el Sheet.

Uso:
    python responder_tickets.py "https://docs.google.com/spreadsheets/d/SHEET_ID/edit"

Requisitos:
    - ANTHROPIC_API_KEY en variable de entorno
    - credentials/*.json (service account de Google)
    - El Sheet debe estar compartido con el email del service account
"""

import asyncio
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import gspread
from anthropic import Anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# --- Configuración ---

load_dotenv(Path(__file__).parent / ".env")

MCP_URL = "https://educabot-d7d4a6e0.mintlify.app/mcp"

CREDENTIALS_DIR = Path(__file__).parent / "credentials"

SYSTEM_PROMPT = """\
Sos un agente de soporte de SINIGEP (Sistema Nacional Integral de Gestión \
Educativa Provincial). Respondé la consulta del usuario basándote \
EXCLUSIVAMENTE en la documentación proporcionada.

Reglas:
- Respondé lo que puedas con la documentación proporcionada.
- Para lo que NO esté cubierto en la doc, decí: "Para esto te recomendamos contactar a soporte técnico."
- NO inventes pasos ni información que no esté en la doc.
- Usá español rioplatense. Tono cordial y profesional ("Buenos días, ¿cómo están?").
- Si la doc menciona una pantalla o botón, nombralo exactamente como aparece.
- Si la documentación te da múltiples caminos posibles, contestá al usuario con \
"Si estás queriendo hacer eso, entonces [sol A], o si buscás esto, [sol B]".
- Si el ticket NO es una pregunta (es un seguimiento, envío de datos, etc.), respondé: \
"[NO ES CONSULTA] Este ticket parece ser un seguimiento o envío de información, no una consulta técnica."
- IMPORTANTE: Si el ticket es sobre corrección de datos (alta de autoridades, cargos faltantes, \
docentes duplicados, datos erróneos, nombramientos, vinculación de cargos), derivá al sitio \
de solicitudes: https://solicitudes-sinigep.up.railway.app/solicitud — indicá que seleccionen \
la opción correspondiente (autoridades, cargos, etc.). Ejemplo de respuesta: \
"Muchas gracias por enviarnos la información. Para avanzar con este trámite, les pedimos que \
ingresen en https://solicitudes-sinigep.up.railway.app/solicitud y seleccionen la opción \
correspondiente para cargar los datos."\
"""

SIN_INFO = "[SIN INFO EN DOCS]"


# --- Google Sheets ---

def encontrar_credenciales() -> Path:
    """Busca el primer archivo .json en la carpeta credentials/."""
    jsons = list(CREDENTIALS_DIR.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .json en {CREDENTIALS_DIR}. "
            "Descargá el service account key de Google Cloud Console."
        )
    return jsons[0]


WORKSHEET_NAME = "DJ de cursos y docentes"


def conectar_sheet(spreadsheet_url: str) -> gspread.Worksheet:
    """Conecta a Google Sheets y devuelve la hoja por nombre."""
    creds_path = encontrar_credenciales()
    gc = gspread.service_account(filename=str(creds_path))
    sh = gc.open_by_url(spreadsheet_url)
    return sh.worksheet(WORKSHEET_NAME)


# --- MCP (Mintlify) ---

MAX_PAGINAS = 2  # páginas completas a traer


async def buscar_docs(session: ClientSession, pregunta: str) -> str:
    """Busca en la documentación de Mintlify via MCP y devuelve contexto."""
    resultado = await session.call_tool(
        name="search_sinigep_documentación",
        arguments={"query": pregunta},
    )

    if not resultado.content:
        return ""

    # Extraer paths únicos de los snippets
    paginas_vistas = []
    for item in resultado.content:
        if hasattr(item, "text") and item.text:
            match = re.search(r"Page: (.+)", item.text)
            if match:
                path = match.group(1).strip()
                if path not in paginas_vistas:
                    paginas_vistas.append(path)

    # Traer páginas completas (top N únicas)
    contexto_paginas = []
    for path in paginas_vistas[:MAX_PAGINAS]:
        try:
            pagina = await session.call_tool(
                name="get_page_sinigep_documentación",
                arguments={"page": path},
            )
            if pagina.content and hasattr(pagina.content[0], "text"):
                contexto_paginas.append(pagina.content[0].text)
        except Exception:
            pass  # si falla una página, seguimos con las demás

    if contexto_paginas:
        return "\n\n---\n\n".join(contexto_paginas)

    # Fallback a snippets si get_page falló
    textos = []
    for item in resultado.content:
        if hasattr(item, "text") and item.text:
            limpio = re.sub(r"<[^>]+>", "", item.text)
            textos.append(limpio)
    return "\n\n---\n\n".join(textos)


# --- Claude API ---

def generar_respuesta(client: Anthropic, pregunta: str, contexto_docs: str) -> str:
    """Genera una respuesta usando Claude con el contexto de la documentación."""
    if not contexto_docs.strip():
        return SIN_INFO

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"## Documentación relevante:\n{contexto_docs}\n\n"
                    f"## Consulta del usuario:\n{pregunta}"
                ),
            }
        ],
    )
    return response.content[0].text


# --- Procesamiento ---

COL_PREGUNTA = 22   # V
COL_RESPUESTA = 25  # Y
COL_CONTROL = 26    # Z — solo procesar si está vacía
MAX_PENDIENTES = 20


async def procesar_sheet(
    session: ClientSession,
    anthropic_client: Anthropic,
    worksheet: gspread.Worksheet,
):
    """Procesa cada fila del sheet: busca docs, genera respuesta, escribe."""
    filas = worksheet.get_all_values()
    total = len(filas) - 1  # sin header
    procesadas = 0
    sin_info = 0

    print(f"\nTotal de filas (sin header): {total}")

    for i, fila in enumerate(filas[1:], start=2):
        if procesadas >= MAX_PENDIENTES:
            print(f"\nLímite de {MAX_PENDIENTES} alcanzado, frenando.")
            break

        pregunta = fila[COL_PREGUNTA - 1] if len(fila) >= COL_PREGUNTA else ""
        control = fila[COL_CONTROL - 1] if len(fila) >= COL_CONTROL else ""

        # Solo procesar si hay pregunta y col Z está vacía
        if not pregunta or control:
            continue

        print(f"[{procesadas + 1}/{MAX_PENDIENTES}] Fila {i}: {pregunta[:80]}...")

        # Buscar en docs
        contexto = await buscar_docs(session, pregunta)

        # Generar respuesta
        respuesta = generar_respuesta(anthropic_client, pregunta, contexto)

        # Escribir respuesta en col Y
        worksheet.update_cell(i, COL_RESPUESTA, respuesta)

        if respuesta == SIN_INFO:
            sin_info += 1
        procesadas += 1

        # Rate limiting (Google Sheets: 300 req/min)
        await asyncio.sleep(1)

    print(f"\nResultado: {procesadas} procesadas, {sin_info} sin info en docs.")


# --- Main ---

async def main():
    if len(sys.argv) < 2:
        print("Uso: python responder_tickets.py <URL_DEL_GOOGLE_SHEET>")
        print('Ejemplo: python responder_tickets.py "https://docs.google.com/spreadsheets/d/abc123/edit"')
        sys.exit(1)

    sheet_url = sys.argv[1]

    # Conectar a Google Sheets
    print("Conectando a Google Sheets...")
    worksheet = conectar_sheet(sheet_url)
    filas = worksheet.get_all_values()
    print(f"Sheet: {worksheet.title} — {len(filas) - 1} filas")

    # Conectar a Claude API
    anthropic_client = Anthropic()
    print("Claude API OK.")

    # Conectar al MCP de Mintlify
    print(f"Conectando al MCP: {MCP_URL}...")
    async with streamablehttp_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Verificar tools disponibles
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"MCP conectado. Tools: {tool_names}")

            search_tool = next((n for n in tool_names if "search" in n), None)
            if not search_tool:
                print("ERROR: el MCP no tiene tool de search. Abortando.")
                sys.exit(1)

            # Procesar
            await procesar_sheet(session, anthropic_client, worksheet)

    print("Listo.")


if __name__ == "__main__":
    asyncio.run(main())
