"""
Helpers para normalizar y clasificar claverramas de SINIGEP.

La claverrama identifica al establecimiento y codifica el nivel en el sufijo
(o en el prefijo, para escuelas técnicas). Ejemplos:

    A-360P   → Primario
    A-503M   → Medio / Secundario
    A-36J    → Inicial / Jardín
    A-793S   → Superior
    ET454    → Técnico
    A-1124   → Desconocido (sin sufijo)

El mapping está pendiente de validación final con Santi — ver plan.
"""

import re


NIVEL_INICIAL = "Inicial"
NIVEL_PRIMARIO = "Primario"
NIVEL_MEDIO = "Medio"
NIVEL_SUPERIOR = "Superior"
NIVEL_TECNICO = "Técnico"
NIVEL_DESCONOCIDO = "Desconocido"


_PATRON_A = re.compile(r"^A-?(\d+)([JPMS]?)$")
_PATRON_ET = re.compile(r"^ET-?\d+$")

_SUFIJO_A_NIVEL = {
    "J": NIVEL_INICIAL,
    "P": NIVEL_PRIMARIO,
    "M": NIVEL_MEDIO,
    "S": NIVEL_SUPERIOR,
}


def normalizar(clave) -> str:
    if not clave:
        return ""
    return re.sub(r"\s+", "", str(clave)).upper()


def inferir_nivel(clave) -> str:
    norm = normalizar(clave)
    if not norm:
        return NIVEL_DESCONOCIDO

    if _PATRON_ET.match(norm):
        return NIVEL_TECNICO

    match = _PATRON_A.match(norm)
    if match:
        sufijo = match.group(2)
        return _SUFIJO_A_NIVEL.get(sufijo, NIVEL_DESCONOCIDO)

    return NIVEL_DESCONOCIDO
