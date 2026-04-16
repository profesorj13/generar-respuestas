"""
Limpiar preámbulo de "thinking" previo a "Buenos días" en respuestas ya escritas.

Recorre la hoja "Tickets - general" desde una fila de inicio y, para cada
respuesta en la columna Y, recorta todo lo que esté antes de "Buenos días".

Excepción: si la respuesta arranca con el marcador "[EXCEDE LIMITE ... chars]"
se deja intacta — ese prefijo lo agrega el pipeline y no queremos tocarlo.

Uso:
    python limpiar_preambulo.py "https://docs.google.com/spreadsheets/d/SHEET_ID/edit"
    python limpiar_preambulo.py "<URL>" --desde 260           # default 260
    python limpiar_preambulo.py "<URL>" --dry-run             # no escribe, solo reporta
"""

import argparse
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import gspread

load_dotenv(Path(__file__).parent / ".env")

CREDENTIALS_DIR = Path(__file__).parent / "credentials"
WORKSHEET_NAME = "Tickets - general"
COL_RESPUESTA = 25  # Y
FILA_INICIO_DEFAULT = 260

RE_SALUDO = re.compile(r"Buenos d[ií]as[\.\,\s]", flags=re.IGNORECASE)
MARCADOR_EXCEDE = "[EXCEDE LIMITE"


def encontrar_credenciales() -> Path:
    jsons = list(CREDENTIALS_DIR.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No se encontró .json en {CREDENTIALS_DIR}")
    return jsons[0]


def _normalizar(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def conectar_sheet(url: str, nombre: str) -> gspread.Worksheet:
    gc = gspread.service_account(filename=str(encontrar_credenciales()))
    sh = gc.open_by_url(url)
    try:
        return sh.worksheet(nombre)
    except gspread.exceptions.WorksheetNotFound:
        objetivo = _normalizar(nombre)
        for ws in sh.worksheets():
            if _normalizar(ws.title) == objetivo:
                return ws
        titulos = ", ".join(f'"{ws.title}"' for ws in sh.worksheets())
        raise SystemExit(
            f'No encontré la hoja "{nombre}". Hojas disponibles: {titulos}\n'
            f"Pasala con --hoja \"<nombre exacto>\"."
        )


def recortar(texto: str) -> str | None:
    """Devuelve el texto recortado, o None si no hay que tocar nada."""
    if not texto:
        return None
    if texto.lstrip().startswith(MARCADOR_EXCEDE):
        return None
    m = RE_SALUDO.search(texto)
    if not m or m.start() == 0:
        return None
    return texto[m.start():].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL del Google Sheet")
    parser.add_argument("--desde", type=int, default=FILA_INICIO_DEFAULT,
                        help=f"Fila desde la que empezar (default {FILA_INICIO_DEFAULT})")
    parser.add_argument("--hoja", default=WORKSHEET_NAME,
                        help=f'Nombre de la hoja (default "{WORKSHEET_NAME}")')
    parser.add_argument("--dry-run", action="store_true",
                        help="No escribe, solo lista lo que se cambiaría")
    args = parser.parse_args()

    ws = conectar_sheet(args.url, args.hoja)
    total_filas = ws.row_count

    if args.desde > total_filas:
        print(f"La fila de inicio ({args.desde}) excede el total ({total_filas}).")
        sys.exit(0)

    rango = f"Y{args.desde}:Y{total_filas}"
    valores = ws.get(rango, value_render_option="FORMATTED_VALUE")

    cambios: list[tuple[int, str]] = []
    for offset, fila in enumerate(valores):
        texto = fila[0] if fila else ""
        recortado = recortar(texto)
        if recortado is None:
            continue
        fila_num = args.desde + offset
        cambios.append((fila_num, recortado))

    print(f"Hoja: {ws.title}")
    print(f"Rango analizado: {rango} ({len(valores)} filas)")
    print(f"Filas a modificar: {len(cambios)}")

    if not cambios:
        return

    for fila_num, nuevo in cambios[:5]:
        preview = nuevo[:80].replace("\n", " ")
        print(f"  fila {fila_num}: {preview}…")
    if len(cambios) > 5:
        print(f"  … y {len(cambios) - 5} más")

    if args.dry_run:
        print("\n[dry-run] no se escribió nada.")
        return

    updates = [
        {"range": f"Y{fila_num}", "values": [[nuevo]]}
        for fila_num, nuevo in cambios
    ]
    ws.batch_update(updates, value_input_option="USER_ENTERED")
    print(f"Actualizadas {len(cambios)} filas.")


if __name__ == "__main__":
    main()
