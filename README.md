# generar-respuestas

Automatización de respuestas a tickets de SINIGEP usando documentación de Mintlify + Claude API.

## Scripts

| Script | Qué hace |
|---|---|
| `responder_tickets.py` | Lee preguntas de un Google Sheet, usa RAG iterativo contra la doc de Mintlify (tool use con 5 vueltas máximo), genera respuestas con Claude y las escribe en el Sheet. Inyecta claverrama + nivel + historial del establecimiento. El catálogo de cargos se consulta on-demand via `search_docs` contra la doc (no se inyecta inline). Post-procesa formato, appendea cierre estándar y valida longitud ≤ 2000 chars. |
| `extraer_respondidos.py` | Extrae tickets ya respondidos por producto desde el Sheet a un CSV (col V + col Y + col Z). |
| `extraer_cargos.py` | **(Mantenimiento, opcional)** Baja la hoja `Lista para ABAP` del sheet de cargos y la guarda en `context/cargos-por-nivel.md`. Se usa como insumo para mantener las páginas de cargos publicadas en Mintlify. **No afecta la corrida de `responder_tickets.py`** — que ya no lee ese archivo. |
| `claverrama.py` | Helper: `inferir_nivel()` mapea sufijo de la claverrama (J/P/M/S/ET) a nivel educativo. |

## Setup

1. Crear un virtualenv e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Crear archivo `.env` con tu API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. Colocar el JSON del service account de Google en `credentials/`:
   ```
   credentials/tu-service-account.json
   ```

4. Compartir el Google Sheet con el email del service account.

## Uso

```bash
# Generar respuestas automáticas
python responder_tickets.py "https://docs.google.com/spreadsheets/d/SHEET_ID/edit"

# Reescribir respuestas ya existentes (p. ej. tras cambiar el prompt)
python responder_tickets.py "<URL>" --force-regen

# Extraer tickets ya respondidos a CSV (análisis de calidad)
python extraer_respondidos.py "https://docs.google.com/spreadsheets/d/SHEET_ID/edit"
```

### Mantenimiento del catálogo de cargos (opcional)

```bash
python extraer_cargos.py
```

Baja la hoja `Lista para ABAP` del sheet de cargos a `context/cargos-por-nivel.md`. Este archivo **no se usa en runtime** — sirve como insumo para mantener las páginas de cargos publicadas en Mintlify, que son las que consulta el RAG via `search_docs`. Correr solo cuando Santi actualice la hoja oficial.

## Columnas del Sheet

- **V (22)**: Consulta del usuario
- **Y (25)**: Respuesta generada por el bot — se saltan filas con Y no vacía salvo `--force-regen`
- **Z (26)**: Respuesta final de producto (control humano) — si tiene algo, la fila se saltea
- **Claverrama**: se detecta automáticamente por header usando substring matching contra `("claverrama", "claverama", "clave rama", "clave_rama", "clave-rama")`. Tolera prefijos (ej: `Educación - Claverama`) y la escritura con una o dos `r`. El índice puede variar — el script no lo hardcodea.

## Cómo funciona el RAG iterativo

El script expone dos tools a Claude (wrappers sobre el MCP de Mintlify):

- `search_docs(query)` — devuelve lista compacta (title, path, snippet) de resultados, backed by `search_sinigep_documentación`.
- `get_page(path)` — trae una página completa, backed by `query_docs_filesystem_sinigep_documentación` usando `cat <path>.mdx` contra el filesystem virtual de la doc.

Claude itera hasta 5 tool calls por ticket decidiendo cuándo buscar más o leer una página entera. Cuando converge, emite la respuesta. Si el ticket no es consulta devuelve `<<NO_ES_CONSULTA>>`; si la doc no cubre el caso devuelve `<<SIN_CONTEXTO>>`.

El historial de mensajes se instancia fresco en cada ticket para garantizar aislamiento — nunca cruza contexto entre tickets.

## Contexto inyectado en el prompt

Por cada ticket, el script le pasa al modelo en el primer mensaje user:

- `Clave rama` y `Nivel inferido` (Inicial / Primario / Medio / Superior / Técnico / Desconocido).
- `Historial reciente del establecimiento` — hasta 3 tickets previos del mismo establecimiento (misma claverrama) que ya tengan respuesta en col Z o col Y.
- La consulta del usuario.

El catálogo de cargos por nivel **no se inyecta** — vive como páginas en la doc de Mintlify y Claude las trae on-demand via `search_docs("cargos <nivel>")` cuando el ticket menciona un cargo/rol/autoridad. El `SYSTEM_PROMPT` obliga al modelo a disparar esa búsqueda antes de responder sobre cargos. La doc general de Mintlify también se trae bajo demanda con `search_docs` y `get_page`.

## Post-procesado

Después de que Claude genera la respuesta, el script:

1. Detecta `<<NO_ES_CONSULTA>>` y `<<SIN_CONTEXTO>>` y los reemplaza por texto de revisión / derivación a tutoría.
2. Limpia markers internos, markdown (`**`, `__`, `##`), caracteres `<>`, exclamaciones (`!`, `¡`) y emojis.
3. Appendea el cierre estándar fijo (no generado por el LLM — se repetía literal en 14/21 respuestas de producto revisadas).
4. Valida que la respuesta final sea ≤ 2000 caracteres (límite BA Colaborativa). Si excede, la marca con `[EXCEDE LIMITE]` para revisión manual en vez de recortar.

## Configuración

- `MAX_PENDIENTES = 20` — máximo de tickets a procesar por corrida
- `MAX_RAG_ITERATIONS = 5` — tope duro de tool calls por ticket
- `MAX_RESPUESTA_CHARS = 2000` — límite de BA Colaborativa
- `MAX_HISTORIAL = 3` — tickets previos del mismo establecimiento a inyectar
- `MODEL = "claude-sonnet-4-20250514"`
- `WORKSHEET_NAME = "DJ de cursos y docentes"` — pestaña del Sheet principal
- `extraer_cargos.py` (mantenimiento, no runtime) lee la hoja `Lista para ABAP` del sheet de cargos (URL hardcoded en el script) y escribe `context/cargos-por-nivel.md` como insumo para actualizar Mintlify.

## TODOs diferidos (fase 2)

- **Plantillas pre-aprobadas por patrón:** clasificar tickets en ~6 patrones recurrentes y llenar slots. Diferido hasta validar si el RAG iterativo ya cierra el gap.
- **Soporte multimodal:** tickets con imágenes adjuntas (capturas) requiere cambiar el pipeline de ingesta (no viene del Sheet).
