from __future__ import annotations

import json
import os
from pathlib import Path

_locale: str | None = None
_translations: dict[str, str] = {}
_SUPPORTED_LOCALES = frozenset({"zh", "zh_hant", "zh_hant_hk", "ja", "ko", "pl", "pt", "sk", "en"})


def _detect_locale() -> str:
    env_lang = os.environ.get("PYROLLER_LANG")
    if env_lang:
        normalized = _normalize_locale(env_lang)
        return normalized if normalized in _SUPPORTED_LOCALES else "en"
    for var in ("LC_ALL", "LANGUAGE", "LANG"):
        for value in _locale_candidates(os.environ.get(var, "")):
            result = _normalize_locale(value)
            if result in _SUPPORTED_LOCALES:
                return result
    return "en"


def _locale_candidates(raw: str) -> list[str]:
    if not raw:
        return []
    return [item for item in raw.split(":") if item]


def _normalize_locale(raw: str) -> str:
    """Normalize a POSIX or BCP 47 locale string to an internal locale key."""
    s = raw.split(".")[0].replace("-", "_").lower()
    parts = s.split("_")

    if parts[0] != "zh":
        return parts[0]

    # zh_Hant, zh-Hant → zh_hant
    # zh_Hant_HK, zh-Hant-HK → zh_hant_hk
    # zh_TW, zh_MO → zh_hant (Taiwan / Macau use Traditional)
    # zh_HK → zh_hant_hk
    # zh_CN, zh_SG → zh (Simplified)
    sub = "_".join(parts[1:])

    if "hant" in sub:
        if "hk" in sub.lower():
            return "zh_hant_hk"
        return "zh_hant"

    if any(r in parts for r in ("hk",)):
        return "zh_hant_hk"
    if any(r in parts for r in ("tw", "mo")):
        return "zh_hant"

    return "zh"


def _load_translations(locale: str) -> dict[str, str]:
    if locale == "en":
        return {}
    base = Path(__file__).parent / "resources" / "locales"
    locale_file = base / f"{locale}.json"
    if not locale_file.exists():
        # zh_hant_hk falls back to zh_hant if the HK variant is missing
        if "_" in locale:
            parent = locale.rsplit("_", 1)[0]
            parent_file = base / f"{parent}.json"
            if parent_file.exists():
                with open(parent_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        return {}
    with open(locale_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _(text: str) -> str:
    global _locale, _translations
    if _locale is None:
        _locale = _detect_locale()
        _translations = _load_translations(_locale)
    return _translations.get(text, text)


def install_argparse_i18n() -> None:
    """Replace argparse's built-in gettext function so its own strings (like
    'usage:', 'positional arguments:', 'show this help message and exit')
    are also translated.

    argparse exposes no public hook for this; keep the private patch explicit
    and limited to CLI parser construction instead of hiding it inside _().
    """
    try:
        import argparse
        argparse._ = _
    except Exception:
        pass
