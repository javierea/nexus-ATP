# rg_atp_pipeline (Etapa 1)

Proyecto base para la descarga y procesamiento de PDFs ATP. En esta etapa se incorpora la descarga, versionado y registro en SQLite (sin extracción de texto u OCR).

## Requisitos
- Python 3.11

## Instalación (editable)
```bash
pip install -e .
```

## Dependencias de testing
```bash
pip install -e ".[test]"
```

## Comandos (Etapa 1)
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
```

## Notas
- `config.yml` y `data/state/state.json` se crean si faltan con valores por defecto.
- El comando `plan` solo imprime URLs candidatas sin hacer requests.
- `fetch` realiza HEAD/GET según configuración, guarda PDFs versionados en `data/raw_pdfs/` y actualiza SQLite en `data/state/rg_atp.sqlite`.
