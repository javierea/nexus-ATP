"""Minimal web UI for rg_atp_pipeline."""

from __future__ import annotations

import html
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import load_config
from .fetcher import FetchOptions, run_fetch, run_manual_fetch
from .paths import config_path, data_dir, state_path
from .state import load_state
from .storage_sqlite import DocumentStore
from .text_extractor import ExtractOptions, run_extract


def run_ui(host: str, port: int) -> None:
    """Run the minimal web UI server."""
    server = ThreadingHTTPServer((host, port), _RGHandler)
    logging.getLogger("rg_atp_pipeline.ui").info("UI disponible en http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return
    finally:
        server.server_close()


class _RGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self._send_text("Not Found", HTTPStatus.NOT_FOUND)
            return
        self._render_home(message=None)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/fetch":
            self._handle_fetch()
            return
        if parsed.path == "/manual-fetch":
            self._handle_manual_fetch()
            return
        if parsed.path == "/extract":
            self._handle_extract()
            return
        if parsed.path == "/delete-record":
            self._handle_delete_record()
            return
        self._send_text("Not Found", HTTPStatus.NOT_FOUND)

    def _handle_fetch(self) -> None:
        data = self._read_form()
        mode = _first(data, "mode", "both")
        year = _parse_int(_first(data, "year", "")) if _first(data, "year", "") else None
        n_start = _parse_int(_first(data, "n_start", "")) if _first(data, "n_start", "") else None
        n_end = _parse_int(_first(data, "n_end", "")) if _first(data, "n_end", "") else None
        old_start = _parse_int(_first(data, "old_start", "")) if _first(data, "old_start", "") else None
        old_end = _parse_int(_first(data, "old_end", "")) if _first(data, "old_end", "") else None
        dry_run = "dry_run" in data
        max_downloads = (
            _parse_int(_first(data, "max_downloads", ""))
            if _first(data, "max_downloads", "")
            else None
        )
        skip_existing = "skip_existing" in data

        config = load_config(config_path())
        state = load_state(state_path())
        store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
        summary = run_fetch(
            config,
            state,
            store,
            data_dir(),
            FetchOptions(
                mode=mode,
                year=year,
                n_start=n_start,
                n_end=n_end,
                old_start=old_start,
                old_end=old_end,
                dry_run=dry_run,
                max_downloads=max_downloads,
                skip_existing=skip_existing,
            ),
            logging.getLogger("rg_atp_pipeline.ui"),
        )
        message = f"Fetch completado: {summary.as_dict()}"
        self._render_home(message=message)

    def _handle_manual_fetch(self) -> None:
        data = self._read_form()
        url = _first(data, "manual_url", "").strip()
        if not url:
            self._render_home(message="URL manual vacía. Intenta nuevamente.")
            return
        year = _parse_int(_first(data, "manual_year", "")) if _first(data, "manual_year", "") else None
        number = (
            _parse_int(_first(data, "manual_number", ""))
            if _first(data, "manual_number", "")
            else None
        )
        config = load_config(config_path())
        state = load_state(state_path())
        store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
        summary = run_manual_fetch(
            config,
            state,
            store,
            data_dir(),
            url,
            year,
            number,
            logging.getLogger("rg_atp_pipeline.ui"),
        )
        message = f"Fetch manual completado: {summary.as_dict()}"
        self._render_home(message=message)

    def _handle_delete_record(self) -> None:
        data = self._read_form()
        doc_key = _first(data, "doc_key", "").strip()
        if not doc_key:
            self._render_home(message="Doc key vacío. Intenta nuevamente.")
            return
        store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
        deleted = store.delete_record(doc_key)
        message = f"Registro eliminado ({doc_key})." if deleted else f"No se encontró {doc_key}."
        self._render_home(message=message)

    def _handle_extract(self) -> None:
        data = self._read_form()
        doc_key = _first(data, "doc_key", "").strip() or None
        status = _first(data, "status", "DOWNLOADED")
        limit = _parse_int(_first(data, "limit", "")) if _first(data, "limit", "") else None
        force = "force" in data
        only_text = "only_text" in data
        only_needs_ocr = "only_needs_ocr" in data
        config = load_config(config_path())
        store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
        summary = run_extract(
            config,
            store,
            data_dir(),
            ExtractOptions(
                status=status,
                limit=limit,
                doc_key=doc_key,
                force=force,
                only_text=only_text,
                only_needs_ocr=only_needs_ocr,
            ),
            logging.getLogger("rg_atp_pipeline.ui"),
        )
        message = f"Extracción completada: {summary.as_dict()}"
        self._render_home(message=message)

    def _render_home(self, message: str | None) -> None:
        store = DocumentStore(data_dir() / "state" / "rg_atp.sqlite")
        records = store.list_records(limit=200)
        config = load_config(config_path())
        state = load_state(state_path())
        body = _render_page(records, config.model_dump(), state.model_dump(mode="json"), message)
        self._send_html(body)

    def _read_form(self) -> dict[str, list[str]]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return parse_qs(raw, keep_blank_values=True)

    def _send_html(self, body: str) -> None:
        content = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_text(self, body: str, status: HTTPStatus) -> None:
        content = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        logging.getLogger("rg_atp_pipeline.ui").info(format, *args)


def _render_page(
    records: list,
    config: dict,
    state: dict,
    message: str | None,
) -> str:
    rows = "\n".join(_render_record_row(record) for record in records)
    message_html = (
        f"<p class='message'>{html.escape(message)}</p>" if message else ""
    )
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Inventario RGs</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
    th {{ background: #f5f5f5; text-align: left; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ border: 1px solid #ddd; padding: 12px; border-radius: 4px; }}
    label {{ display: block; margin-top: 8px; }}
    input[type="text"], input[type="number"] {{ width: 100%; padding: 6px; }}
    .message {{ background: #eef6ff; padding: 8px; border-left: 4px solid #3b82f6; }}
    .small {{ font-size: 12px; color: #666; }}
    .delete-form {{ margin: 0; }}
    .delete-button {{ background: #ef4444; color: white; border: none; padding: 6px 10px; cursor: pointer; }}
    .delete-button:hover {{ background: #dc2626; }}
  </style>
</head>
<body>
  <h1>Inventario de RGs</h1>
  {message_html}
  <table>
    <thead>
      <tr>
        <th>Doc Key</th>
        <th>Familia</th>
        <th>Año</th>
        <th>Número</th>
        <th>Status</th>
        <th>Última revisión</th>
        <th>Última descarga</th>
        <th>Text status</th>
        <th>Needs OCR</th>
        <th>URL</th>
        <th>Acciones</th>
      </tr>
    </thead>
    <tbody>
      {rows if rows else "<tr><td colspan='11'>Sin registros.</td></tr>"}
    </tbody>
  </table>

  <div class="grid">
    <div class="panel">
      <h2>Config actual</h2>
      <pre>{html.escape(json.dumps(config, indent=2, ensure_ascii=False))}</pre>
    </div>
    <div class="panel">
      <h2>Estado</h2>
      <pre>{html.escape(json.dumps(state, indent=2, ensure_ascii=False))}</pre>
    </div>
  </div>

  <div class="grid" style="margin-top: 16px;">
    <div class="panel">
      <h2>Fetch programado</h2>
      <form method="post" action="/fetch">
        <label>Modo
          <select name="mode">
            <option value="both">both</option>
            <option value="new">new</option>
            <option value="old">old</option>
          </select>
        </label>
        <label>Año (solo new)
          <input type="number" name="year" min="1900" max="2100" />
        </label>
        <label>N inicio (new)
          <input type="number" name="n_start" min="1" />
        </label>
        <label>N fin (new)
          <input type="number" name="n_end" min="1" />
        </label>
        <label>Old inicio
          <input type="number" name="old_start" min="1" />
        </label>
        <label>Old fin
          <input type="number" name="old_end" min="1" />
        </label>
        <label>Máximo descargas
          <input type="number" name="max_downloads" min="1" />
        </label>
        <label><input type="checkbox" name="dry_run" /> Dry run</label>
        <label><input type="checkbox" name="skip_existing" /> Omitir existentes</label>
        <button type="submit">Ejecutar fetch</button>
      </form>
      <p class="small">Esto usa las mismas opciones del CLI.</p>
    </div>
    <div class="panel">
      <h2>Fetch manual</h2>
      <form method="post" action="/manual-fetch">
        <label>URL del PDF
          <input type="text" name="manual_url" placeholder="https://..." />
        </label>
        <label>Año (opcional)
          <input type="number" name="manual_year" min="1900" max="2100" />
        </label>
        <label>Número (opcional)
          <input type="number" name="manual_number" min="1" />
        </label>
        <button type="submit">Descargar URL</button>
      </form>
      <p class="small">Registra la URL en el inventario con familia MANUAL.</p>
    </div>
    <div class="panel">
      <h2>Extracción de texto</h2>
      <form method="post" action="/extract">
        <label>Doc key (opcional)
          <input type="text" name="doc_key" placeholder="DOC-KEY" />
        </label>
        <label>Status
          <input type="text" name="status" value="DOWNLOADED" />
        </label>
        <label>Límite
          <input type="number" name="limit" min="1" />
        </label>
        <label><input type="checkbox" name="force" /> Reprocesar</label>
        <label><input type="checkbox" name="only_text" /> Solo EXTRACTED</label>
        <label><input type="checkbox" name="only_needs_ocr" /> Solo NEEDS_OCR</label>
        <button type="submit">Ejecutar extract</button>
      </form>
      <p class="small">Extrae texto crudo y marca NEEDS_OCR según config.</p>
    </div>
  </div>
</body>
</html>"""


def _render_record_row(record) -> str:
    needs_ocr = "Sí" if record.text_status == "NEEDS_OCR" else "No"
    return (
        "<tr>"
        f"<td>{html.escape(record.doc_key)}</td>"
        f"<td>{html.escape(record.doc_family)}</td>"
        f"<td>{html.escape(str(record.year or ''))}</td>"
        f"<td>{html.escape(str(record.number or ''))}</td>"
        f"<td>{html.escape(record.status)}</td>"
        f"<td>{html.escape(record.last_checked_at)}</td>"
        f"<td>{html.escape(record.last_downloaded_at or '')}</td>"
        f"<td>{html.escape(record.text_status)}</td>"
        f"<td>{html.escape(needs_ocr)}</td>"
        f"<td><a href='{html.escape(record.url)}' target='_blank'>link</a></td>"
        "<td>"
        "<form method='post' action='/delete-record' class='delete-form'>"
        f"<input type='hidden' name='doc_key' value='{html.escape(record.doc_key)}' />"
        "<button type='submit' class='delete-button' "
        "onclick=\"return confirm('¿Eliminar este registro?');\">Eliminar</button>"
        "</form>"
        "</td>"
        "</tr>"
    )


def _first(data: dict[str, list[str]], key: str, default: str) -> str:
    values = data.get(key)
    return values[0] if values else default


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None
