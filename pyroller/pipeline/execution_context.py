from __future__ import annotations

import json
import logging
from typing import Any

from pyroller.transcriber.base import Transcriber
from pyroller.transcriber.registry import build_transcriber, resolve_transcriber_backend, sanitize_transcriber_config

logger = logging.getLogger("pyroller.pipeline")


class PipelineExecutionContext:
    def __init__(self) -> None:
        self._transcriber_key: tuple[str, str, str] | None = None
        self._transcriber: Transcriber | None = None

    def _make_transcriber_key(self, *, language: str, backend_name: str, config: dict[str, Any]) -> tuple[str, str, str]:
        return (
            language,
            backend_name,
            json.dumps(config, sort_keys=True, ensure_ascii=False, default=str),
        )

    def get_transcriber(self, *, language: str, backend_name: str | None, config: dict[str, Any]) -> Transcriber:
        effective_language, chosen_backend = resolve_transcriber_backend(language, backend_name)
        sanitized = sanitize_transcriber_config(chosen_backend, config)
        key = self._make_transcriber_key(language=effective_language, backend_name=chosen_backend, config=sanitized)
        if self._transcriber is None or self._transcriber_key != key:
            if self._transcriber is not None and self._transcriber_key is not None:
                logger.info(
                    "Execution context switching transcriber: old_backend=%s old_language=%s new_backend=%s new_language=%s",
                    self._transcriber_key[1],
                    self._transcriber_key[0],
                    key[1],
                    key[0],
                )
            self.close_transcriber()
            logger.info("Execution context building transcriber backend=%s language=%s", chosen_backend, effective_language)
            self._transcriber = build_transcriber(
                language=effective_language,
                backend_name=chosen_backend,
                config=sanitized,
            )
            self._transcriber_key = key
        return self._transcriber

    def close_transcriber(self) -> None:
        transcriber = self._transcriber
        transcriber_key = self._transcriber_key
        self._transcriber = None
        self._transcriber_key = None
        if transcriber is None:
            return
        try:
            if transcriber_key is not None:
                logger.info(
                    "Execution context closing transcriber backend=%s language=%s",
                    transcriber_key[1],
                    transcriber_key[0],
                )
            else:
                logger.info("Execution context closing transcriber backend=<unknown> language=<unknown>")
            transcriber.close()
        except Exception:
            logger.exception("Execution context failed to close transcriber cleanly")

    def close(self) -> None:
        self.close_transcriber()
