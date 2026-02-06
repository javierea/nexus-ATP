# rg_atp_pipeline (Etapa 0)

Proyecto base para la descarga y procesamiento de PDFs ATP. En esta etapa solo se crea el andamiaje (estructura, configuración, estado, CLI y planificación de URLs) sin realizar requests.

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

## Comandos (Etapa 0)
```bash
python -m rg_atp_pipeline --help
python -m rg_atp_pipeline init
python -m rg_atp_pipeline validate
python -m rg_atp_pipeline show-config
python -m rg_atp_pipeline show-state
python -m rg_atp_pipeline plan
```

## Notas
- `config.yml` y `data/state/state.json` se crean si faltan con valores por defecto.
- El comando `plan` solo imprime URLs candidatas sin hacer requests.
