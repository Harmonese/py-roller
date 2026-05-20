#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "pyroller"
LOCALES_DIR = PACKAGE_ROOT / "resources" / "locales"

ARGPARSE_KEYS = {
    "usage: ",
    "positional arguments",
    "options",
    "optional arguments",
    "show this help message and exit",
    "and",
    "or",
}

ALLOWED_UNTRANSLATED_KEYS = {
    "%s%s",
    "%s%s - %s",
    "+ {}",
    " / {}",
    "{} {}",
    "{}\n\n{}\n\n{}",
    "  [{:<4}] {:<18} {}",
    "  [{tag:<5}] #{index:03d} {stem} :: {msg}",
    "  {label:<22}: {path}",
    "  py-roller install",
    "[ERROR] {}",
    "OK",
    "py-roller doctor",
    "py-roller install",
    "torchaudio {}",
}

SUBPROCESS_I18N_RE = re.compile(r"_i18n\('([^']*)'\)")


def collect_message_keys() -> set[str]:
    keys = set(ARGPARSE_KEYS)
    for path in PACKAGE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "_":
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                keys.add(arg.value)

        # Some install validation messages live inside a subprocess script
        # string and call _i18n() after f-string brace escaping.
        for match in SUBPROCESS_I18N_RE.finditer(source):
            keys.add(match.group(1).replace("{{", "{").replace("}}", "}"))
    return keys


def load_locale(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
        raise ValueError(f"{path} must contain a JSON object of string keys and string values")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Check py-roller locale JSON coverage against code message keys.")
    parser.add_argument(
        "--strict-untranslated",
        action="store_true",
        help="also fail when a locale value is identical to its English key",
    )
    args = parser.parse_args()

    expected = collect_message_keys()
    failed = False
    for locale_path in sorted(LOCALES_DIR.glob("*.json")):
        data = load_locale(locale_path)
        keys = set(data)
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        untranslated = sorted(
            key
            for key, value in data.items()
            if key in expected and value == key and key not in ALLOWED_UNTRANSLATED_KEYS
        )

        if missing or extra or (args.strict_untranslated and untranslated):
            failed = True
        print(
            f"{locale_path.name}: keys={len(keys)} expected={len(expected)} "
            f"missing={len(missing)} extra={len(extra)} untranslated={len(untranslated)}"
        )
        for label, items in (("missing", missing), ("extra", extra)):
            for item in items[:20]:
                print(f"  {label}: {item!r}")
            if len(items) > 20:
                print(f"  {label}: ... {len(items) - 20} more")
        if args.strict_untranslated:
            for item in untranslated[:20]:
                print(f"  untranslated: {item!r}")
            if len(untranslated) > 20:
                print(f"  untranslated: ... {len(untranslated) - 20} more")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
