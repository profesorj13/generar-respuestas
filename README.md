# generar-respuestas

Automatización de respuestas a tickets de SINIGEP usando documentación de Mintlify + Claude API.

## Scripts

| Script | Qué hace |
|---|---|
| `responder_tickets.py` | Lee preguntas de un Google Sheet, busca en la doc de Mintlify via MCP, genera respuestas con Claude y las escribe en el Sheet |
| `extraer_respondidos.py` | Extrae tickets ya respondidos por producto desde el Sheet a un CSV |

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

# Extraer tickets ya respondidos a CSV
python extraer_respondidos.py "https://docs.google.com/spreadsheets/d/SHEET_ID/edit"
```

## Columnas del Sheet

- **V (22)**: Consulta del usuario
- **Y (25)**: Respuesta generada por el bot
- **Z (26)**: Respuesta de producto (control) — solo procesa filas donde Z está vacía

## Configuración

- `MAX_PENDIENTES = 20` — máximo de tickets a procesar por corrida
- `MAX_PAGINAS = 2` — páginas de Mintlify a traer como contexto
- `WORKSHEET_NAME = "DJ de cursos y docentes"` — pestaña del Sheet
