from __future__ import annotations

import json
import os
from pathlib import Path

_locale: str | None = None
_translations: dict[str, str] = {}


def _detect_locale() -> str:
    env_lang = os.environ.get("PYROLLER_LANG")
    if env_lang:
        return env_lang.split("_")[0].split(".")[0].lower()
    for var in ("LC_ALL", "LANG", "LANGUAGE"):
        val = os.environ.get(var, "")
        if val:
            lang = val.split("_")[0].split(".")[0].lower()
            if lang in ("zh", "en"):
                return lang
    return "en"


def _load_translations(locale: str) -> dict[str, str]:
    if locale == "en":
        return {}
    locale_file = Path(__file__).parent / "resources" / "locales" / f"{locale}.json"
    if not locale_file.exists():
        return {}
    with open(locale_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _(text: str) -> str:
    global _locale, _translations
    if _locale is None:
        _locale = _detect_locale()
        _translations = _load_translations(_locale)
        _patch_argparse()
    return _translations.get(text, text)


def _patch_argparse() -> None:
    """Replace argparse's built-in gettext function so its own strings (like
    'usage:', 'positional arguments:', 'show this help message and exit')
    are also translated."""
    try:
        import argparse
        argparse._ = _
    except Exception:
        pass
