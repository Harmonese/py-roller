from __future__ import annotations

import pytest

from pyroller.parser.en_arpabet import EnglishARPAbetParser
from pyroller.parser.mul_ipa import MultilingualIPAParser
from pyroller.parser.registry import (
    get_lyrics_parser,
    get_parser_requirements,
    list_available_parser_backends,
    resolve_parser_backend,
    resolve_parser_language,
)
from pyroller.parser.zh_router_pinyin import ZhRouterPinyinParser


@pytest.mark.parametrize(
    ("language", "backend", "parser_type"),
    [
        ("zh", "zh_router_pinyin", ZhRouterPinyinParser),
        ("en", "en_arpabet", EnglishARPAbetParser),
        ("mul", "mul_ipa", MultilingualIPAParser),
    ],
)
def test_parser_registry_maps_languages_to_default_backends(language: str, backend: str, parser_type: type) -> None:
    assert resolve_parser_backend(language) == backend
    assert list_available_parser_backends(language) == (backend,)
    assert isinstance(get_lyrics_parser(language), parser_type)


def test_parser_registry_falls_back_to_multilingual_for_unknown_language() -> None:
    assert resolve_parser_language("xx") == "mul"
    assert resolve_parser_backend("xx") == "mul_ipa"
    assert isinstance(get_lyrics_parser("xx"), MultilingualIPAParser)


def test_parser_requirements_are_declared_by_effective_backend() -> None:
    assert get_parser_requirements("zh") == ("pypinyin",)
    assert get_parser_requirements("en") == ()
    assert get_parser_requirements("xx") == ()
