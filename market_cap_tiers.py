"""M&A-oriented market cap size tiers (billions USD)."""

from __future__ import annotations

from typing import Optional


def classify_market_cap_billions(market_cap_billions: Optional[float]) -> Optional[str]:
    """Return size_tier string or None if cap is missing or not numeric.

    Boundaries:
    - large_cap: above $100B
    - mid_cap: $5B–$100B inclusive
    - small_cap: $500M–$5B (0.5 <= x < 5)
    - micro_cap: below $500M
    """

    if market_cap_billions is None:
        return None
    try:
        b = float(market_cap_billions)
    except (TypeError, ValueError):
        return None

    if b > 100:
        return "large_cap"
    if b >= 5:
        return "mid_cap"
    if b >= 0.5:
        return "small_cap"
    return "micro_cap"
