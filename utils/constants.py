"""Project-wide constants."""

from __future__ import annotations

# Canonical sector labels — single source of truth for Chroma metadata, SIC mapping, and classifiers.
CANONICAL_SECTORS: list[str] = [
    "Technology Hardware",
    "Technology Software",
    "Pharma",
    "Financial Services",
    "Retail",
    "Energy",
    "Industrials",
    "Healthcare",
    "Telecommunications",
    "Consumer Goods",
    "Real Estate",
    "Metals and Mining",
]

# Default sector tag for software/semiconductor peer companies in rag/build_database.py.
DEFAULT_PEER_SOFTWARE_SECTOR = "Technology Software"

# ---------------------------------------------------------------------------
# Sector-specific thresholds for standalone red flag checks.
# Each key matches a canonical sector name exactly.
# Metrics not listed in a sector entry fall back to DEFAULT_THRESHOLDS.
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict = {
    "gross_margin_min": 20.0,       # % — below this is a red flag
    "operating_margin_min": 0.0,    # % — below this (negative) is a red flag
    "current_ratio_min": 1.0,       # ratio
    "debt_to_equity_max": 3.0,      # ratio — above this is a red flag
    "cash_conversion_min": 0.8,     # ratio
    "roe_min": 0.05,                # ratio (not %)
    "free_cash_flow_min": 0.0,      # $M — below this is a red flag
}

SECTOR_THRESHOLDS: dict[str, dict] = {
    "Technology Software": {
        "gross_margin_min": 50.0,
        "operating_margin_min": 0.0,
        "current_ratio_min": 1.0,
        "debt_to_equity_max": 3.0,
        "cash_conversion_min": 0.8,
        "roe_min": 0.05,
        "free_cash_flow_min": 0.0,
    },
    "Technology Hardware": {
        "gross_margin_min": 30.0,
        "operating_margin_min": 0.0,
        "current_ratio_min": 1.0,
        "debt_to_equity_max": 3.0,
        "cash_conversion_min": 0.8,
        "roe_min": 0.05,
        "free_cash_flow_min": 0.0,
    },
    "Pharma": {
        "gross_margin_min": 50.0,
        "operating_margin_min": -10.0,  # R&D-heavy, negative ops margin acceptable
        "current_ratio_min": 1.2,
        "debt_to_equity_max": 3.0,
        "cash_conversion_min": 0.7,
        "roe_min": 0.03,
        "free_cash_flow_min": 0.0,
    },
    "Financial Services": {
        "gross_margin_min": 20.0,       # net interest margin proxy
        "operating_margin_min": 0.0,
        "current_ratio_min": 1.0,
        "debt_to_equity_max": 15.0,     # leverage is structural in banking
        "cash_conversion_min": 0.7,
        "roe_min": 0.05,
        "free_cash_flow_min": 0.0,
    },
    "Retail": {
        "gross_margin_min": 15.0,       # structurally thin margins
        "operating_margin_min": 0.0,
        "current_ratio_min": 0.8,       # negative working capital common
        "debt_to_equity_max": 4.0,
        "cash_conversion_min": 0.7,
        "roe_min": 0.05,
        "free_cash_flow_min": 0.0,
    },
    "Energy": {
        "gross_margin_min": 15.0,       # commodity exposure compresses margins
        "operating_margin_min": 0.0,
        "current_ratio_min": 1.0,
        "debt_to_equity_max": 2.5,      # capital-intensive, lower tolerance
        "cash_conversion_min": 0.8,
        "roe_min": 0.04,
        "free_cash_flow_min": 0.0,
    },
    "Industrials": {
        "gross_margin_min": 20.0,
        "operating_margin_min": 0.0,
        "current_ratio_min": 1.0,
        "debt_to_equity_max": 3.0,
        "cash_conversion_min": 0.8,
        "roe_min": 0.05,
        "free_cash_flow_min": 0.0,
    },
    "Healthcare": {
        "gross_margin_min": 30.0,
        "operating_margin_min": 0.0,
        "current_ratio_min": 1.0,
        "debt_to_equity_max": 3.0,
        "cash_conversion_min": 0.8,
        "roe_min": 0.04,
        "free_cash_flow_min": 0.0,
    },
    "Telecommunications": {
        "gross_margin_min": 30.0,
        "operating_margin_min": 0.0,
        "current_ratio_min": 0.8,       # capex-heavy, lower current ratio tolerated
        "debt_to_equity_max": 4.0,      # high leverage structural in telecoms
        "cash_conversion_min": 0.8,
        "roe_min": 0.04,
        "free_cash_flow_min": 0.0,
    },
    "Consumer Goods": {
        "gross_margin_min": 25.0,
        "operating_margin_min": 0.0,
        "current_ratio_min": 1.0,
        "debt_to_equity_max": 3.0,
        "cash_conversion_min": 0.8,
        "roe_min": 0.05,
        "free_cash_flow_min": 0.0,
    },
    "Real Estate": {
        "gross_margin_min": 25.0,
        "operating_margin_min": 0.0,
        "current_ratio_min": 0.8,
        "debt_to_equity_max": 6.0,      # REITs are structurally highly leveraged
        "cash_conversion_min": 0.7,
        "roe_min": 0.03,
        "free_cash_flow_min": 0.0,
    },
    "Metals and Mining": {
        "gross_margin_min": 15.0,       # commodity margins structurally thin
        "operating_margin_min": 0.0,
        "current_ratio_min": 1.0,
        "debt_to_equity_max": 2.5,
        "cash_conversion_min": 0.8,
        "roe_min": 0.04,
        "free_cash_flow_min": 0.0,
    },
}


def get_thresholds(sector: str) -> dict:
    """Return threshold config for a sector, falling back to DEFAULT_THRESHOLDS for missing keys."""
    sector_specific = SECTOR_THRESHOLDS.get(sector, {})
    return {**DEFAULT_THRESHOLDS, **sector_specific}
