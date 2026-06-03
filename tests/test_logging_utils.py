from __future__ import annotations

import logging

from pyroller.logging_utils import configure_logging


def test_configure_logging_builds_console_only_config(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(logging.config, "dictConfig", lambda config: captured.update(config))

    result = configure_logging(level="warning", log_file=None)

    assert result is None
    assert captured["handlers"]["console"]["level"] == "WARNING"
    assert captured["root"]["handlers"] == ["console"]
    assert captured["root"]["level"] == "DEBUG"


def test_configure_logging_creates_file_handler_config(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    log_file = tmp_path / "logs" / "run.log"
    monkeypatch.setattr(logging.config, "dictConfig", lambda config: captured.update(config))

    result = configure_logging(level="INFO", log_file=log_file)

    assert result == log_file
    assert log_file.parent.exists()
    assert captured["handlers"]["file"]["filename"] == str(log_file)
    assert captured["handlers"]["file"]["encoding"] == "utf-8"
    assert captured["root"]["handlers"] == ["console", "file"]
