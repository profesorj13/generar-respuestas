"""
Extraer tickets con las tres columnas para análisis de calidad de respuestas IA.

Lee col V (consulta) + col Y (respuesta sugerida por IA) + col Z (respuesta de producto)
y guarda en inputs/comparacion_respuestas.csv las filas que tienen las tres columnas completas.

Uso:
    python extraer_respondidos.py "https://docs.google.com/spreadsheets/d/SHEET_ID/edit"
"""

import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
import gspread

load_dotenv(Path(__file__).parent / ".env")

CREDENTIALS_DIR = Path(__file__).parent / "credentials"
WORKSHEET_NAME = "Tickets - General"
OUTPUT_PATH = Path(__file__).parent.parent / "inputs" / "comparacion_respuestas.csv"

COL_CONSULTA = 22         # V — consulta del usuario
COL_RESPUESTA_IA = 25     # Y — respuesta sugerida por la IA
COL_RESPUESTA_PRODUCTO = 26  # Z — respuesta ideal de producto


def encontrar_credenciales() -> Path:
    jsons = list(CREDENTIALS_DIR.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No se encontró .json en {CREDENTIALS_DIR}")
    return jsons[0]


def main():
    if len(sys.argv) < 2:
        print("Uso: python extraer_respondidos.py <URL_DEL_GOOGLE_SHEET>")
        sys.exit(1)

    sheet_url = sys.argv[1]

    # Conectar
    print("Conectando a Google Sheets...")
    gc = gspread.service_account(filename=str(encontrar_credenciales()))
    sh = gc.open_by_url(sheet_url)
    ws = sh.worksheet(WORKSHEET_NAME)

    filas = ws.get_all_values()
    total = len(filas) - 1
    print(f"Sheet: {ws.title} — {total} filas")

    # Extraer filas con las tres columnas completas
    completos = []
    for i, fila in enumerate(filas[1:], start=2):
        consulta = fila[COL_CONSULTA - 1] if len(fila) >= COL_CONSULTA else ""
        respuesta_ia = fila[COL_RESPUESTA_IA - 1] if len(fila) >= COL_RESPUESTA_IA else ""
        respuesta_producto = fila[COL_RESPUESTA_PRODUCTO - 1] if len(fila) >= COL_RESPUESTA_PRODUCTO else ""

        if consulta and respuesta_ia and respuesta_producto:
            completos.append({
                "fila": i,
                "consulta": consulta,
                "respuesta_ia": respuesta_ia,
                "respuesta_producto": respuesta_producto,
            })

    print(f"\nPre-extracción:")
    print(f"  Total filas: {total}")
    print(f"  Con las 3 columnas completas: {len(completos)}")
    print(f"  Sin las 3 columnas: {total - len(completos)}")

    if not completos:
        print("\nNo hay filas con las tres columnas completas. No se genera archivo.")
        return

    # Escribir CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fila", "consulta", "respuesta_ia", "respuesta_producto"])
        writer.writeheader()
        writer.writerows(completos)

    print(f"  Guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
