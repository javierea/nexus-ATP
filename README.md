# rg_atp_pipeline (Etapa 4)

Proyecto base para la descarga y procesamiento de PDFs ATP. En esta etapa se incorpora la extracción de texto crudo, métricas de calidad y marcado para OCR.

## Requisitos
- Python 3.11

## Instalación en Windows (Conda)
```bash
conda create -n rg-atp python=3.11
conda activate rg-atp

# Librerías requeridas
pip install pydantic PyYAML pypdf requests typer

# Instalación del proyecto (editable)
pip install -e .
```

## Librerías requeridas
- pydantic
- PyYAML
- pypdf
- requests
- typer

## Instalación (editable)
```bash
pip install -e .
```

## Dependencias de testing
```bash
pip install -e ".[test]"
```

## UI Streamlit (opcional)
```bash
pip install -e ".[ui]"
python -m rg_atp_pipeline ui-streamlit
```

## Comandos (Etapa 2)
```bash
python -m rg_atp_pipeline --help
python -m rg_atp_pipeline init
python -m rg_atp_pipeline validate
python -m rg_atp_pipeline show-config
python -m rg_atp_pipeline show-state
python -m rg_atp_pipeline plan
python -m rg_atp_pipeline fetch --mode both
python -m rg_atp_pipeline fetch --mode new --year 2026 --n-start 1 --n-end 5
python -m rg_atp_pipeline fetch --mode old --old-start 1 --old-end 10 --dry-run
python -m rg_atp_pipeline fetch --mode both --skip-existing
python -m rg_atp_pipeline extract --status DOWNLOADED --limit 50
python -m rg_atp_pipeline extract --doc-key RG-2024-001 --force
python -m rg_atp_pipeline extract --only-needs-ocr
python -m rg_atp_pipeline split-rgs --input-text data/text/COMPENDIO-2024.txt --logical-page-offset 45
python -m rg_atp_pipeline split-rgs --input-text data/text/COMPENDIO-2024.txt --page-marker-pattern '^===PAGE\s+(\d+)===$'
python -m rg_atp_pipeline structure --limit 50
python -m rg_atp_pipeline structure --doc-key RG-2024-001 --force
python -m rg_atp_pipeline structure --include-needs-ocr --no-export-json
python -m rg_atp_pipeline audit-compendio --pdf-path data/compendio-legislativo-al-31-12-2024.pdf
python -m rg_atp_pipeline audit-compendio --only-missing-downloads
python -m rg_atp_pipeline seed-norms
python -m rg_atp_pipeline seed-common-aliases --seed-path data/state/seeds/common_aliases.yml
python -m rg_atp_pipeline upload-norm --norm-key LEY-83-F --file path.pdf --authoritative
python -m rg_atp_pipeline resolve-norm --text "Dec. Ley 2444/62"
python -m rg_atp_pipeline merge-norm --from UNK-D3BE5096 --to LEY-83-F --dry-run
python -m rg_atp_pipeline merge-norm --from UNK-D3BE5096 --to LEY-83-F --apply
python -m rg_atp_pipeline citations --llm off
python -m rg_atp_pipeline citations --llm verify --ollama-model qwen2.5:7b-instruct
python -m rg_atp_pipeline relations --llm off
python -m rg_atp_pipeline relations --llm verify --prompt-version reltype-v1
python -m rg_atp_pipeline ui --host 127.0.0.1 --port 8000
python -m rg_atp_pipeline ui-streamlit --host 127.0.0.1 --port 8501
```

## Notas
- `config.yml` y `data/state/state.json` se crean si faltan con valores por defecto.
- El comando `plan` solo imprime URLs candidatas sin hacer requests.
- `fetch` realiza HEAD/GET según configuración, guarda PDFs versionados en `data/raw_pdfs/` y actualiza SQLite en `data/state/rg_atp.sqlite`.
- `fetch` puede omitir entradas ya descargadas con `--skip-existing` si el PDF local sigue disponible.
- `extract` genera texto crudo por página en `data/text/`, calcula métricas y marca `NEEDS_OCR` cuando corresponde.
- `split-rgs` (post-extract / pre-structure) detecta inicios de RG por encabezado (`RESOLUCIÓN GENERAL Nº ...`) y valida con `VISTO:` o `VISTO Y CONSIDERANDO`, exportando bloques individuales a `data/pre_structure/`. Soporta numeración `XXXX`, `XXXX/AA` y `XXXX/AAAA`, y offset de páginas lógicas del índice (ej. índice 1 = PDF 46 => `--logical-page-offset 45`). El patrón de página es configurable con `--page-marker-pattern`; por defecto usa `===== PÁGINA XXX/770 =====`.
- `structure` segmenta el texto crudo en unidades normativas (ARTÍCULO/ANEXO/secciones) y guarda unidades en SQLite, con export JSON opcional en `data/structured/`.
- `audit-compendio` extrae referencias a RG desde el compendio legislativo, normaliza claves (`RES-AAAA-NN-20-1` u `OLD-N`) y exporta CSV/JSON en `data/audit/` con comparación contra `data/state/rg_atp.sqlite`. También genera `missing_downloads_*.csv` y soporta `--only-missing-downloads` para exportar únicamente ese listado.
- `seed-norms` carga el catálogo inicial de normas y aliases desde `data/state/seeds/norms.yml`.
- `seed-common-aliases` carga aliases comunes versionados (CTP / Ley Tarifaria) desde `data/state/seeds/common_aliases.yml`; es idempotente y se puede ejecutar múltiples veces sin duplicar filas.
- `upload-norm` permite subir manualmente un PDF asociado a una norma, versionándolo por SHA256 en `data/raw_pdfs/` y registrando la fuente.
- `resolve-norm` resuelve un texto libre contra aliases conocidos y sugiere crear placeholder si no encuentra match.
- `merge-norm` fusiona una norma origen (p. ej. placeholder `UNK-*`) sobre una norma canónica sin cambiar esquema. Soporta `--dry-run` (default) para contar filas potencialmente afectadas y `--apply` para ejecutar el merge transaccional (`BEGIN IMMEDIATE`, `foreign_keys=ON`, `busy_timeout=30000`).
- `merge-norm` actualiza `citation_links` y `relation_extractions`, mueve aliases desde `norm_aliases` (con inserción idempotente `INSERT OR IGNORE`), agrega alias de trazabilidad con la clave origen y deja la norma origen sin borrar por defecto.
- `merge-norm` imprime un resumen JSON con `rows_affected` por tabla, `aliases_moved` y `errors` para auditoría.
- `citations` (Etapa 4) detecta referencias normativas en RGs, verifica opcionalmente con Ollama local y resuelve contra el catálogo `norms` (no clasifica relaciones deroga/modifica).
- `relations` (Etapa 4.1) tipifica relaciones normativas desde `citations` + `citation_links` ya resueltos (deroga/modifica/sustituye/incorpora/reglamenta/según, etc.), con validación LLM opcional y modo conservador (`UNKNOWN` cuando no es claro). Por defecto cierra el input a la última `citations.extract_version` (o `--citation-extract-version` para fijarla explícitamente).
- Stage 4.1 también materializa relaciones intra-norma (`intra_norm_relations`) para navegación interna de artículos en consultas GraphRAG.
- Para diagnóstico manual de errores `UNIQUE constraint failed` en Etapa 4.1, ejecutar `sqlite3 data/state/rg_atp.sqlite < docs/stage4_1_sqlite_diagnostico.sql` y compartir el output.
- `graphrag-validate` ejecuta una suite de validación GraphRAG-ready (KPIs + queries de negocio/aceptación).
- La página Audit de Streamlit incluye una sección opcional para depurar `missing_downloads` con Ollama local y exportar resultados por veredicto.
- `ui` levanta una interfaz mínima para revisar inventario, ver config/estado y ejecutar fetches manuales o programados.
- `ui-streamlit` levanta un panel Streamlit con dashboard, acciones de pipeline, módulos de configuración y la página Audit para correr la auditoría del compendio.
