from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from utils import app_config


_LANGUAGE_ORDER = ("en", "zh", "ja")
_DEFAULT_LANGUAGES = {
    "en": "English",
    "zh": "简体中文",
    "ja": "日本語",
}
_CACHE: dict[str, dict[str, Any]] = {}


def _candidate_paths(language: str) -> list[Path]:
    filename = f"{language}.json"
    paths: list[Path] = []
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        paths.append(base / "i18n" / filename)
        paths.append(Path(sys.executable).parent / "i18n" / filename)
    paths.append(Path(__file__).resolve().parents[1] / "i18n" / filename)
    paths.append(Path.cwd() / "i18n" / filename)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_language(language: str) -> dict[str, Any]:
    language = app_config.normalize_language(language) or "en"
    if language in _CACHE:
        return _CACHE[language]

    for path in _candidate_paths(language):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _CACHE[language] = data
                return data

    data = {
        "language": language,
        "display_name": _DEFAULT_LANGUAGES.get(language, language),
        "namespaces": {},
    }
    _CACHE[language] = data
    return data


def clear_cache() -> None:
    _CACHE.clear()


def available_languages() -> dict[str, str]:
    languages: dict[str, str] = {}
    for code in _LANGUAGE_ORDER:
        data = load_language(code)
        display_name = data.get("display_name", _DEFAULT_LANGUAGES[code])
        languages[code] = str(display_name)
    return languages


def language_display_to_code() -> dict[str, str]:
    return {display: code for code, display in available_languages().items()}


def language_code_to_display() -> dict[str, str]:
    return available_languages()


def translate(namespace: str, key: str) -> str:
    language = app_config.get_language()
    candidates = [language]
    if language != "en":
        candidates.append("en")

    for candidate in candidates:
        data = load_language(candidate)
        namespaces = data.get("namespaces", {})
        namespace_data = namespaces.get(namespace, {}) if isinstance(namespaces, dict) else {}
        if isinstance(namespace_data, dict) and key in namespace_data:
            return str(namespace_data[key])
    return key
