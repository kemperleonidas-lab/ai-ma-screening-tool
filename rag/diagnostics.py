"""Diagnostics utilities for peer cohort coverage (non-RAG)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _norm_tier(tier: Any) -> str:
    t = str(tier or "").strip()
    return t if t else "unknown"


def peer_cohort_counts_by_sector_and_tier(
    peer_registry_path: str | Path,
) -> Dict[str, Dict[str, int]]:
    """
    Return counts of peers per sector and size_tier from rag/peer_registry.json.

    Output shape:
      { "Technology Software": { "large_cap": 4, "mid_cap": 5, ... }, ... }
    """
    path = Path(peer_registry_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    if not isinstance(data, dict):
        return {}

    for sector, peers in data.items():
        if not isinstance(peers, list):
            continue
        # Skip Damodaran list stored in registry.
        if sector == "industry_averages":
            continue
        for p in peers:
            if not isinstance(p, dict):
                continue
            tier = _norm_tier(p.get("size_tier"))
            counts[str(sector)][tier] += 1

    # Convert nested defaultdict to plain dict
    return {s: dict(tiers) for s, tiers in counts.items()}


def thin_peer_cohorts(
    peer_registry_path: str | Path,
    *,
    min_peers: int = 5,
) -> List[Tuple[str, str, int]]:
    """
    Return list of (sector, size_tier, count) where count < min_peers.
    """
    counts = peer_cohort_counts_by_sector_and_tier(peer_registry_path)
    out: List[Tuple[str, str, int]] = []
    for sector, tiers in counts.items():
        for tier, n in tiers.items():
            if n < min_peers:
                out.append((sector, tier, int(n)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out

