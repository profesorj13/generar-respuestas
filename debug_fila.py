"""
Debug de una fila específica del Sheet: corre el RAG iterativo con logging
detallado de cada tool call (search_docs / get_page) y muestra la respuesta
final post-procesada. NO escribe al Sheet.

Uso:
    python debug_fila.py "<SHEET_URL>" 299
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from responder_tickets import (
    conectar_sheet,
    detectar_col_claverrama,
    obtener_historial_establecimiento,
    validar_y_postprocesar,
    COL_PREGUNTA,
    COL_CONTROL,
    COL_RESPUESTA,
)
from core_rag import (
    MCP_URL,
    MCP_SEARCH_TOOL,
    MCP_FILESYSTEM_TOOL,
    MODEL,
    MAX_RAG_ITERATIONS,
    SYSTEM_PROMPT,
    TOOLS_SCHEMA,
    construir_mensaje_inicial,
    ejecutar_tool,
)
from claverrama import inferir_nivel, NIVEL_DESCONOCIDO

load_dotenv(Path(__file__).parent / ".env")


async def run_con_log(
    client: AsyncAnthropic,
    session: ClientSession,
    pregunta: str,
    clave_rama: str,
    nivel: str,
    historial_establecimiento: list[dict],
):
    mensaje_inicial = construir_mensaje_inicial(
        pregunta=pregunta,
        clave_rama=clave_rama,
        nivel=nivel,
        historial=historial_establecimiento,
    )
    print("── MENSAJE INICIAL ──────────────────────────────────────────")
    print(mensaje_inicial)
    print()

    historial = [{"role": "user", "content": mensaje_inicial}]
    tool_calls = 0

    for it in range(MAX_RAG_ITERATIONS):
        response = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS_SCHEMA,
            messages=historial,
        )

        if response.stop_reason != "tool_use":
            textos = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            respuesta = "\n".join(textos).strip()
            print(f"── ITER {it+1} · stop={response.stop_reason} · RESPUESTA FINAL CRUDA ──")
            print(respuesta)
            print()
            return respuesta, tool_calls

        historial.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if getattr(block, "type", None) == "text" and block.text.strip():
                print(f"── ITER {it+1} · THINKING ──")
                print(block.text.strip())
                print()

        tool_results_content = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_calls += 1
            inp = dict(block.input)
            print(f"── ITER {it+1} · TOOL CALL #{tool_calls} ──")
            print(f"  {block.name}({json.dumps(inp, ensure_ascii=False)})")

            try:
                resultado = await ejecutar_tool(session, block.name, inp)
            except Exception as e:
                resultado = f"Error: {e}"

            print(f"  → {len(resultado)} chars")
            preview = resultado[:1200].replace("\n", " ")
            print(f"  preview: {preview}...")
            print()

            tool_results_content.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": resultado[:8000],
            })

        historial.append({"role": "user", "content": tool_results_content})

    # Fallback si pegó el límite de iteraciones.
    historial.append({
        "role": "user",
        "content": (
            "Llegaste al límite de búsquedas. Respondé ahora con lo que tengas, "
            "respetando todas las reglas del prompt. Si no tenés contexto suficiente "
            "devolvé exactamente <<SIN_CONTEXTO>>."
        ),
    })
    final = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=historial,
    )
    textos = [b.text for b in final.content if getattr(b, "type", None) == "text"]
    respuesta = "\n".join(textos).strip()
    print("── MAX ITERATIONS · FALLBACK ──")
    print(respuesta)
    return respuesta, tool_calls


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sheet_url")
    parser.add_argument("row", type=int, help="Número de fila (1-based)")
    args = parser.parse_args()

    ws = conectar_sheet(args.sheet_url)
    filas = ws.get_all_values()
    if args.row < 2 or args.row > len(filas):
        print(f"ERROR: fila {args.row} fuera de rango (2..{len(filas)}).")
        sys.exit(1)

    fila = filas[args.row - 1]
    consulta = fila[COL_PREGUNTA - 1] if len(fila) >= COL_PREGUNTA else ""
    respuesta_existente = fila[COL_RESPUESTA - 1] if len(fila) >= COL_RESPUESTA else ""
    respuesta_producto = fila[COL_CONTROL - 1] if len(fila) >= COL_CONTROL else ""

    col_claverrama = detectar_col_claverrama(filas[0])
    clave_rama = ""
    if col_claverrama and len(fila) >= col_claverrama:
        clave_rama = fila[col_claverrama - 1].strip()
    nivel = inferir_nivel(clave_rama) if clave_rama else NIVEL_DESCONOCIDO

    historial = obtener_historial_establecimiento(
        filas=filas,
        col_claverrama=col_claverrama,
        clave_actual=clave_rama,
        fila_actual=args.row,
    )

    print(f"════════════════════════════════════════════════════════════")
    print(f"FILA {args.row} · clave={clave_rama or '—'} · nivel={nivel}")
    print(f"Historial inyectado: {len(historial)} tickets previos")
    print(f"════════════════════════════════════════════════════════════")
    print()
    print("── CONSULTA ──")
    print(consulta)
    print()

    client = AsyncAnthropic()
    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            assert MCP_SEARCH_TOOL in names and MCP_FILESYSTEM_TOOL in names

            cruda, n_tools = await run_con_log(
                client, session, consulta, clave_rama, nivel, historial,
            )

    final, estado = await validar_y_postprocesar(cruda, client, MODEL)
    print()
    print(f"════════════════════════════════════════════════════════════")
    print(f"TOTAL tool calls: {n_tools} · estado postproc: {estado}")
    print(f"── RESPUESTA FINAL (post-proceso, {len(final)} chars) ──")
    print(final)
    print()
    if respuesta_producto:
        print(f"── RESPUESTA DE PRODUCTO (col Z) ──")
        print(respuesta_producto)
    elif respuesta_existente:
        print(f"── RESPUESTA ACTUAL EN COL Y ──")
        print(respuesta_existente)


if __name__ == "__main__":
    asyncio.run(main())
