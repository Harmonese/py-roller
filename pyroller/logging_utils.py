from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional


def configure_logging(level: str = "INFO", log_file: Optional[Path] = None) -> Optional[Path]:
    resolved_log_file: Optional[Path] = log_file
    handlers: dict[str, dict[str, object]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level.upper(),
            "formatter": "standard",
        }
    }
    root_handlers = ["console"]

    if resolved_log_file is not None:
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(resolved_log_file),
            "encoding": "utf-8",
        }
        root_handlers.append("file")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": handlers,
            "root": {
                "level": "DEBUG",
                "handlers": root_handlers,
            },
        }
    )
    return resolved_log_file
