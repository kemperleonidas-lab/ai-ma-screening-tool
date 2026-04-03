"""Helpers for mapping free text to canonical sector labels."""

from __future__ import annotations

from utils.constants import CANONICAL_SECTORS

_CANONICAL_SET = frozenset(CANONICAL_SECTORS)


def normalize_to_canonical_sector(text: str, default: str = "Industrials") -> str:
    """Match GPT or fuzzy output to exactly one label in :data:`CANONICAL_SECTORS`."""

    cleaned = " ".join(str(text).strip().split())
    if cleaned in _CANONICAL_SET:
        return cleaned
    lower = cleaned.lower()
    for choice in sorted(_CANONICAL_SET, key=len, reverse=True):
        if choice.lower() == lower or choice.lower() in lower:
            return choice
    return default if default in _CANONICAL_SET else "Industrials"
