"""
Test de debug: corre las 5 filas con peor puntaje con logging detallado
del proceso RAG (queries de búsqueda, resultados MCP, iteraciones).

NO escribe nada en el Google Sheet.

Uso:
    python test_debug.py
"""

import asyncio
import json
import os
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
    construir_mensaje_inicial,
    validar_y_postprocesar,
    ejecutar_tool,
    COL_PREGUNTA,
    COL_CONTROL,
    MCP_URL,
    MCP_SEARCH_TOOL,
    MCP_FILESYSTEM_TOOL,
    SYSTEM_PROMPT,
    TOOLS_SCHEMA,
    MODEL,
    MAX_RAG_ITERATIONS,
)
from claverrama import inferir_nivel, NIVEL_DESCONOCIDO

load_dotenv(Path(__file__).parent / ".env")

FILAS_DEBUG = [72, 69, 68, 77, 73]
SHEET_URL = os.environ.get("SHEET_URL", "")
OUTPUT_DIR = Path(__file__).parent.parent / "inputs" / "debug_logs"


async def responder_con_log(
    client: AsyncAnthropic,
    session: ClientSession,
    pregunta: str,
    clave_rama: str,
    nivel: str,
    historial_establecimiento: list[dict],
) -> dict:
    """RAG iterativo con logging completo de cada paso."""

    mensaje_inicial = construir_mensaje_inicial(
        pregunta=pregunta,
        clave_rama=clave_rama,
        nivel=nivel,
        historial=historial_establecimiento,
    )

    historial = [{"role": "user", "content": mensaje_inicial}]
    tool_calls = 0
    log_pasos = []

    log_pasos.append({
        "paso": 0,
        "tipo": "mensaje_inicial",
        "contenido": mensaje_inicial,
    })

    for iteracion in range(MAX_RAG_ITERATIONS):
        response = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS_SCHEMA,
            messages=historial,
        )

        if response.stop_reason == "end_turn" or response.stop_reason != "tool_use":
            textos = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            respuesta = "\n".join(textos).strip()
            log_pasos.append({
                "paso": iteracion + 1,
                "tipo": "respuesta_final",
                "stop_reason": response.stop_reason,
                "respuesta": respuesta,
            })
            return {
                "respuesta_cruda": respuesta,
                "tool_calls": tool_calls,
                "pasos": log_pasos,
            }

        historial.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if getattr(block, "type", None) == "text" and block.text.strip():
                log_pasos.append({
                    "paso": iteracion + 1,
                    "tipo": "thinking",
                    "texto": block.text.strip(),
                })

        tool_results_content = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_calls += 1

            tool_input = dict(block.input)
            log_paso = {
                "paso": iteracion + 1,
                "tipo": "tool_call",
                "tool": block.name,
                "input": tool_input,
            }

            try:
                resultado = await ejecutar_tool(session, block.name, tool_input)
            except Exception as e:
                resultado = f"Error: {e}"

            resultado_truncado = resultado[:3000]
            log_paso["resultado_preview"] = resultado_truncado[:500]
            log_paso["resultado_chars"] = len(resultado)

            log_pasos.append(log_paso)

            tool_results_content.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": resultado[:8000],
            })

        historial.append({"role": "user", "content": tool_results_content})

    log_pasos.append({"paso": "limite", "tipo": "max_iterations_reached"})

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

    log_pasos.append({
        "paso": "fallback",
        "tipo": "respuesta_final",
        "stop_reason": final.stop_reason,
        "respuesta": respuesta,
    })

    return {
        "respuesta_cruda": respuesta,
        "tool_calls": tool_calls,
        "pasos": log_pasos,
    }


async def main():
    if not SHEET_URL:
        print("ERROR: SHEET_URL no está definida en .env")
        sys.exit(1)

    print(f"Debug de RAG — filas {FILAS_DEBUG}")
    print(f"Logs en: {OUTPUT_DIR}\n")

    print("Conectando a Google Sheets...")
    worksheet = conectar_sheet(SHEET_URL)
    filas = worksheet.get_all_values()
    print(f"Sheet: {worksheet.title} — {len(filas) - 1} filas")

    col_claverrama = detectar_col_claverrama(filas[0] if filas else [])

    anthropic_client = AsyncAnthropic()
    print("Claude API OK.")

    print(f"Conectando al MCP: {MCP_URL}...")
    async with streamablehttp_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"MCP conectado. Tools: {tool_names}\n")

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            for fila_num in FILAS_DEBUG:
                idx = fila_num - 1
                if idx < 1 or idx >= len(filas):
                    continue

                fila = filas[idx]
                consulta = fila[COL_PREGUNTA - 1] if len(fila) >= COL_PREGUNTA else ""
                respuesta_producto = fila[COL_CONTROL - 1] if len(fila) >= COL_CONTROL else ""

                if not consulta:
                    continue

                clave_rama = ""
                if col_claverrama and len(fila) >= col_claverrama:
                    clave_rama = fila[col_claverrama - 1].strip()
                nivel = inferir_nivel(clave_rama) if clave_rama else NIVEL_DESCONOCIDO

                historial = obtener_historial_establecimiento(
                    filas=filas,
                    col_claverrama=col_claverrama,
                    clave_actual=clave_rama,
                    fila_actual=fila_num,
                )

                print(f"{'='*60}")
                print(f"FILA {fila_num} | clave={clave_rama or '—'} nivel={nivel}")
                print(f"Consulta: {consulta[:100]}...")
                print()

                resultado = await responder_con_log(
                    client=anthropic_client,
                    session=session,
                    pregunta=consulta,
                    clave_rama=clave_rama,
                    nivel=nivel,
                    historial_establecimiento=historial,
                )

                for paso in resultado["pasos"]:
                    if paso["tipo"] == "mensaje_inicial":
                        print(f"  [INIT] Mensaje enviado ({len(paso['contenido'])} chars)")
                    elif paso["tipo"] == "thinking":
                        print(f"  [THINK] {paso['texto'][:150]}...")
                    elif paso["tipo"] == "tool_call":
                        print(f"  [TOOL] {paso['tool']}({json.dumps(paso['input'], ensure_ascii=False)})")
                        print(f"         -> {paso['resultado_chars']} chars | preview: {paso['resultado_preview'][:120]}...")
                    elif paso["tipo"] == "respuesta_final":
                        print(f"  [DONE] stop={paso['stop_reason']} | {len(paso['respuesta'])} chars")

                resp_final, estado = await validar_y_postprocesar(
                    resultado["respuesta_cruda"], anthropic_client, MODEL,
                )
                print(f"  [POST] estado={estado} | {len(resp_final)} chars")
                print()

                log_file = OUTPUT_DIR / f"fila_{fila_num}.json"
                log_data = {
                    "fila": fila_num,
                    "clave_rama": clave_rama,
                    "nivel": nivel,
                    "consulta": consulta,
                    "respuesta_ia": resp_final,
                    "estado": estado,
                    "tool_calls": resultado["tool_calls"],
                    "respuesta_producto": respuesta_producto,
                    "pasos": resultado["pasos"],
                }
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)

                print(f"  Log guardado: {log_file}")
                print()
                await asyncio.sleep(1)

    print(f"\n{'='*60}")
    print(f"Debug completo. Logs en {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
