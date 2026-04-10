"""
Extraer tickets respondidos por producto desde Google Sheets.

Lee col V (consulta) + col Z (respuesta de producto) y guarda
en inputs/tickets_respondidos.csv las filas que tienen respuesta.

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
WORKSHEET_NAME = "DJ de cursos y docentes"
OUTPUT_PATH = Path(__file__).parent.parent / "inputs" / "tickets_respondidos.csv"

COL_CONSULTA = 22  # V
COL_RESPUESTA_PRODUCTO = 26  # Z


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

    # Extraer filas con respuesta de producto
    respondidos = []
    for i, fila in enumerate(filas[1:], start=2):
        consulta = fila[COL_CONSULTA - 1] if len(fila) >= COL_CONSULTA else ""
        respuesta = fila[COL_RESPUESTA_PRODUCTO - 1] if len(fila) >= COL_RESPUESTA_PRODUCTO else ""

        if consulta and respuesta:
            respondidos.append({
                "fila": i,
                "consulta": consulta,
                "respuesta_producto": respuesta,
            })

    # Escribir CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fila", "consulta", "respuesta_producto"])
        writer.writeheader()
        writer.writerows(respondidos)

    print(f"\nResultado:")
    print(f"  Total filas: {total}")
    print(f"  Con respuesta de producto: {len(respondidos)}")
    print(f"  Sin respuesta: {total - len(respondidos)}")
    print(f"  Guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
