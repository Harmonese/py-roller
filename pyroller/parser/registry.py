from __future__ import annotations

import inspect
import logging
from typing import Any

from pyroller.parser.base import LyricsParser
from pyroller.parser.en_arpabet import EnglishARPAbetParser
from pyroller.parser.mul_ipa import MultilingualIPAParser
from pyroller.parser.zh_pinyin import ChinesePinyinParser
from pyroller.parser.zh_router_pinyin import ZhRouterPinyinParser

logger = logging.getLogger("pyroller.parser")

_SUPPORTED_LANGUAGES = {"zh", "en", "mul"}
_DEFAULT_PARSER_BY_LANGUAGE = {
    "zh": "zh_router_pinyin",
    "en": "en_arpabet",
    "mul": "mul_ipa",
}

_PARSER_FACTORIES = {
    "zh_pinyin": ChinesePinyinParser,
    "zh_router_pinyin": ZhRouterPinyinParser,
    "en_arpabet": EnglishARPAbetParser,
    "mul_ipa": MultilingualIPAParser,
}

_PARSER_REQUIREMENTS = {
    "zh_pinyin": ("pypinyin",),
    "zh_router_pinyin": ("pypinyin",),
    "en_arpabet": (),
    "mul_ipa": (),
}

_FALLBACK_LANGUAGE = "mul"


def resolve_parser_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in _SUPPORTED_LANGUAGES:
        return normalized
    logger.error("Unsupported language=%s for parser. Falling back to language=%s.", language, _FALLBACK_LANGUAGE)
    return _FALLBACK_LANGUAGE


def list_available_parser_backends(language: str) -> tuple[str, ...]:
    effective_language = resolve_parser_language(language)
    backend = _DEFAULT_PARSER_BY_LANGUAGE[effective_language]
    return (backend,)


def resolve_parser_backend(language: str) -> str:
    return _DEFAULT_PARSER_BY_LANGUAGE[resolve_parser_language(language)]


def get_parser_requirements(language: str) -> tuple[str, ...]:
    backend = resolve_parser_backend(language)
    return _PARSER_REQUIREMENTS.get(backend, ())


def get_lyrics_parser(language: str, config: dict[str, Any] | None = None) -> LyricsParser:
    effective_language = resolve_parser_language(language)
    backend_name = _DEFAULT_PARSER_BY_LANGUAGE[effective_language]
    factory = _PARSER_FACTORIES[backend_name]
    init_config = {key: value for key, value in dict(config or {}).items() if value is not None}
    init_config.pop("backend", None)
    signature = inspect.signature(factory.__init__)
    accepted = {name for name in signature.parameters if name != "self"}
    filtered_config = {key: value for key, value in init_config.items() if key in accepted}
    return factory(**filtered_config)
