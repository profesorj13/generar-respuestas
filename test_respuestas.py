"""
Test de respuestas: re-genera respuestas para filas específicas del sheet
y las guarda en CSV local para comparar contra las respuestas de producto.

NO escribe nada en el Google Sheet.

Uso:
    python test_respuestas.py
    python test_respuestas.py --desde 85 --hasta 99
    python test_respuestas.py --desde 85 --hasta 99 --concurrencia 2
    python test_respuestas.py --desde 85 --hasta 99 --model claude-haiku-4-5-20251001
"""

import asyncio
import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from responder_tickets import (
    conectar_sheet,
    detectar_col_claverrama,
    obtener_historial_establecimiento,
    responder_con_rag_iterativo,
    validar_y_postprocesar,
    COL_PREGUNTA,
    COL_CONTROL,
    MCP_URL,
    MCP_SEARCH_TOOL,
    MCP_FILESYSTEM_TOOL,
)
from claverrama import inferir_nivel, NIVEL_DESCONOCIDO

load_dotenv(Path(__file__).parent / ".env")

FILA_DESDE = 85
FILA_HASTA = 99
CONCURRENCIA = 1
MAX_RETRIES = 3
RETRY_BASE_DELAY = 65  # segundos — el rate limit es por minuto
SHEET_URL = os.environ.get("SHEET_URL", "")
OUTPUT_PATH = Path(__file__).parent.parent / "inputs" / "test_respuestas_v2.csv"

FIELDNAMES = [
    "fila",
    "consulta",
    "respuesta_ia",
    "estado",
    "tool_calls",
    "chars",
    "respuesta_producto",
]


async def procesar_fila(
    fila_num: int,
    fila: list[str],
    filas: list[list[str]],
    col_claverrama: int | None,
    anthropic_client: AsyncAnthropic,
    session: ClientSession,
    semaphore: asyncio.Semaphore,
    model_override: str | None = None,
) -> dict:
    """Procesa una fila individual, respetando el semáforo de concurrencia."""
    async with semaphore:
        consulta = fila[COL_PREGUNTA - 1] if len(fila) >= COL_PREGUNTA else ""
        respuesta_producto = fila[COL_CONTROL - 1] if len(fila) >= COL_CONTROL else ""

        if not consulta:
            print(f"  [{fila_num}] sin consulta, saltando")
            return None

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

        print(
            f"[{fila_num}] INICIO | clave={clave_rama or '—'} nivel={nivel} "
            f"historial={len(historial)}"
        )
        print(f"  Consulta: {consulta[:80]}...")
        t0 = time.time()

        last_error = None
        for intento in range(MAX_RETRIES + 1):
            try:
                respuesta_cruda, tool_calls = await responder_con_rag_iterativo(
                    client=anthropic_client,
                    session=session,
                    pregunta=consulta,
                    clave_rama=clave_rama,
                    nivel=nivel,
                    historial_establecimiento=historial,
                    model_override=model_override,
                )
                break  # éxito
            except Exception as e:
                last_error = e
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str
                if is_rate_limit and intento < MAX_RETRIES:
                    wait = RETRY_BASE_DELAY * (intento + 1)
                    print(
                        f"  [{fila_num}] Rate limit (intento {intento + 1}/{MAX_RETRIES}), "
                        f"esperando {wait}s..."
                    )
                    await asyncio.sleep(wait)
                    t0 = time.time()  # resetear timer
                    continue
                elapsed = time.time() - t0
                print(f"  [{fila_num}] ERROR ({elapsed:.1f}s): {e}")
                return {
                    "fila": fila_num,
                    "consulta": consulta[:200],
                    "respuesta_ia": f"[ERROR: {e}]",
                    "estado": "ERROR",
                    "tool_calls": 0,
                    "chars": 0,
                    "respuesta_producto": respuesta_producto,
                }
        else:
            # Se agotaron los reintentos
            elapsed = time.time() - t0
            print(f"  [{fila_num}] ERROR tras {MAX_RETRIES} retries ({elapsed:.1f}s): {last_error}")
            return {
                "fila": fila_num,
                "consulta": consulta[:200],
                "respuesta_ia": f"[ERROR tras retries: {last_error}]",
                "estado": "ERROR",
                "tool_calls": 0,
                "chars": 0,
                "respuesta_producto": respuesta_producto,
            }

        respuesta_final, estado = validar_y_postprocesar(respuesta_cruda)
        elapsed = time.time() - t0

        print(
            f"  [{fila_num}] DONE ({elapsed:.1f}s) | estado={estado} "
            f"tool_calls={tool_calls} chars={len(respuesta_final)}"
        )

        return {
            "fila": fila_num,
            "consulta": consulta[:200],
            "respuesta_ia": respuesta_final,
            "estado": estado,
            "tool_calls": tool_calls,
            "chars": len(respuesta_final),
            "respuesta_producto": respuesta_producto,
        }


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test de respuestas IA vs producto.")
    parser.add_argument("--desde", type=int, default=FILA_DESDE)
    parser.add_argument("--hasta", type=int, default=FILA_HASTA)
    parser.add_argument("--filas", type=str, default=None,
                        help="Filas específicas separadas por coma (ej: 136,137,140). Ignora --desde/--hasta.")
    parser.add_argument("--concurrencia", type=int, default=CONCURRENCIA,
                        help="Máximo de filas procesadas en paralelo (default: 1)")
    parser.add_argument("--model", type=str, default=None,
                        help="Modelo a usar (default: el de responder_tickets.py)")
    args = parser.parse_args()

    if args.filas:
        filas_target = [int(f.strip()) for f in args.filas.split(",")]
    else:
        filas_target = list(range(args.desde, args.hasta + 1))

    fila_desde = min(filas_target)
    fila_hasta = max(filas_target)
    concurrencia = args.concurrencia
    model_override = args.model

    if not SHEET_URL:
        print("ERROR: SHEET_URL no está definida en .env")
        sys.exit(1)

    if args.filas:
        print(f"Test de respuestas — filas {args.filas}")
    else:
        print(f"Test de respuestas — filas {fila_desde} a {fila_hasta}")
    print(f"Modelo: {model_override or 'default (sonnet)'}")
    print(f"Concurrencia: {concurrencia} filas en paralelo")
    print(f"Output: {OUTPUT_PATH}\n")

    print("Conectando a Google Sheets...")
    worksheet = conectar_sheet(SHEET_URL)
    filas = worksheet.get_all_values()
    total = len(filas) - 1
    print(f"Sheet: {worksheet.title} — {total} filas")

    col_claverrama = detectar_col_claverrama(filas[0] if filas else [])
    if col_claverrama:
        print(f"Clave rama en columna {col_claverrama}")
    else:
        print("WARN: no se encontró columna clave rama")

    anthropic_client = AsyncAnthropic()
    print("Claude API OK.")

    print(f"Conectando al MCP: {MCP_URL}...")
    async with streamablehttp_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"MCP conectado. Tools: {tool_names}")

            if MCP_SEARCH_TOOL not in tool_names or MCP_FILESYSTEM_TOOL not in tool_names:
                print("ERROR: faltan tools MCP requeridas. Abortando.")
                sys.exit(1)

            semaphore = asyncio.Semaphore(concurrencia)
            t_global = time.time()

            # Preparar tareas para todas las filas válidas
            tasks = []
            for fila_num in filas_target:
                idx = fila_num - 1
                if idx < 1 or idx >= len(filas):
                    print(f"  Fila {fila_num}: fuera de rango, saltando")
                    continue

                tasks.append(
                    procesar_fila(
                        fila_num=fila_num,
                        fila=filas[idx],
                        filas=filas,
                        col_claverrama=col_claverrama,
                        anthropic_client=anthropic_client,
                        session=session,
                        semaphore=semaphore,
                        model_override=model_override,
                    )
                )

            # Ejecutar todas en paralelo (el semáforo limita la concurrencia)
            resultados_raw = await asyncio.gather(*tasks, return_exceptions=True)

            elapsed_total = time.time() - t_global

    # Filtrar resultados válidos y contar estados
    resultados = []
    contadores: dict[str, int] = {}
    for r in resultados_raw:
        if isinstance(r, Exception):
            print(f"  Excepción no capturada: {r}")
            contadores["ERROR"] = contadores.get("ERROR", 0) + 1
            continue
        if r is None:
            continue
        resultados.append(r)
        estado = r["estado"]
        contadores[estado] = contadores.get(estado, 0) + 1

    # Ordenar por número de fila
    resultados.sort(key=lambda x: x["fila"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\n{'='*60}")
    print(f"Resultados: {len(resultados)} filas procesadas en {elapsed_total:.1f}s")
    print(f"Estados: {contadores}")
    print(f"Guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
