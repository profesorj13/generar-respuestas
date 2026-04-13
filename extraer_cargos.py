"""
Bajar la hoja "lista para abap" del sheet de cargos por nivel y guardarla
como tabla markdown en context/cargos-por-nivel.md.

Uso:
    python extraer_cargos.py

El archivo generado lo consume responder_tickets.py como contexto fijo en
cada llamada al modelo.
"""

from pathlib import Path

from dotenv import load_dotenv
import gspread

load_dotenv(Path(__file__).parent / ".env")

CREDENTIALS_DIR = Path(__file__).parent / "credentials"
SHEET_URL = "https://docs.google.com/spreadsheets/d/1BhbyQMUycVlnfnYQpHzW1xBmy5jUXM1JQTzx5iDz1iE/edit"
WORKSHEET_NAME = "Lista para ABAP"
OUTPUT_PATH = Path(__file__).parent / "context" / "cargos-por-nivel.md"
COLUMNAS_RELEVANTES = [
    "Nivel",
    "descripcion",
    "rama",
    "descripcion2",
    "esautoridad",
    "eslegal",
    "horaspermitidas",
]


def encontrar_credenciales() -> Path:
    jsons = list(CREDENTIALS_DIR.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No se encontró .json en {CREDENTIALS_DIR}")
    return jsons[0]


def escapar_celda(valor: str) -> str:
    return valor.replace("|", "\\|").replace("\n", " ").strip()


def filas_a_markdown(filas: list[list[str]]) -> str:
    if not filas:
        return "_(hoja vacía)_\n"

    header_completo = [escapar_celda(c) for c in filas[0]]
    try:
        indices = [header_completo.index(col) for col in COLUMNAS_RELEVANTES]
    except ValueError as e:
        raise ValueError(
            f"No se encontró columna esperada en la hoja. Headers: {header_completo}"
        ) from e

    header = [header_completo[i] for i in indices]
    ancho_total = len(header_completo)

    lineas = []
    lineas.append("| " + " | ".join(header) + " |")
    lineas.append("| " + " | ".join(["---"] * len(header)) + " |")

    for fila in filas[1:]:
        fila_norm = [escapar_celda(c) for c in fila[:ancho_total]]
        fila_norm += [""] * (ancho_total - len(fila_norm))
        if not any(fila_norm):
            continue
        fila_filtrada = [fila_norm[i] for i in indices]
        lineas.append("| " + " | ".join(fila_filtrada) + " |")

    return "\n".join(lineas) + "\n"


def main():
    print("Conectando a Google Sheets...")
    gc = gspread.service_account(filename=str(encontrar_credenciales()))
    sh = gc.open_by_url(SHEET_URL)
    ws = sh.worksheet(WORKSHEET_NAME)

    filas = ws.get_all_values()
    print(f"Hoja: {ws.title} — {len(filas)} filas (incluye header)")

    tabla = filas_a_markdown(filas)

    contenido = (
        "# Catálogo de cargos por nivel (SINIGEP)\n\n"
        f"Fuente: hoja `{WORKSHEET_NAME}` del sheet de cargos. "
        "Generado por `extraer_cargos.py`. No editar a mano.\n\n"
        f"{tabla}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(contenido, encoding="utf-8")
    print(f"Guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
