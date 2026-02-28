import logging
from logging.handlers import RotatingFileHandler

from rg_atp_pipeline.logging_utils import SafeRotatingFileHandler


def test_safe_rotating_file_handler_swallows_permission_error(monkeypatch, tmp_path):
    log_path = tmp_path / "app.log"
    handler = SafeRotatingFileHandler(log_path, maxBytes=1, backupCount=1, encoding="utf-8")
    logger = logging.getLogger("test.safe.rotate")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    def broken_rollover(self):
        raise PermissionError("locked")

    monkeypatch.setattr(RotatingFileHandler, "doRollover", broken_rollover)

    logger.info("message")

    assert log_path.exists()
