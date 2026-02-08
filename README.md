# rg_atp_pipeline (Etapa 2)

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
python -m rg_atp_pipeline ui --host 127.0.0.1 --port 8000
```

## Notas
- `config.yml` y `data/state/state.json` se crean si faltan con valores por defecto.
- El comando `plan` solo imprime URLs candidatas sin hacer requests.
- `fetch` realiza HEAD/GET según configuración, guarda PDFs versionados en `data/raw_pdfs/` y actualiza SQLite en `data/state/rg_atp.sqlite`.
- `fetch` puede omitir entradas ya descargadas con `--skip-existing` si el PDF local sigue disponible.
- `extract` genera texto crudo por página en `data/text/`, calcula métricas y marca `NEEDS_OCR` cuando corresponde.
- `ui` levanta una interfaz mínima para revisar inventario, ver config/estado y ejecutar fetches manuales o programados.
