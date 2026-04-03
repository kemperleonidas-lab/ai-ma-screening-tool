from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import logging
from typing import Any, Dict

import requests


SEC_USER_AGENT = "financial-analyzer contact@example.com"
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "application/json",
}

APPLE_CIK = "0000320193"
COMPANY_FACTS_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{APPLE_CIK}.json"


logger = logging.getLogger(__name__)


def _get_10k_yearly_maxes(tag_data: Dict[str, Any]) -> Dict[int, float]:
    """Return per-fiscal-year max values for a single tag (10-K only)."""

    units = tag_data.get("units")
    if not isinstance(units, dict):
        return {}

    per_year_max: Dict[int, float] = {}

    for _unit_name, datapoints in units.items():
        if not isinstance(datapoints, list):
            continue
        for point in datapoints:
            if not isinstance(point, dict):
                continue
            if point.get("form") != "10-K":
                continue

            fy_val = point.get("fy")
            try:
                fy = int(fy_val)
            except (TypeError, ValueError):
                continue

            val = point.get("val")
            try:
                numeric_val = float(val)
            except (TypeError, ValueError):
                continue

            current_max = per_year_max.get(fy)
            if current_max is None or numeric_val > current_max:
                per_year_max[fy] = numeric_val

    return per_year_max


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Fetching Apple (AAPL) company facts from SEC EDGAR.")
    try:
        resp = requests.get(COMPANY_FACTS_URL, headers=SEC_HEADERS, timeout=60)
    except requests.RequestException as exc:
        logger.error("Network error fetching company facts: %s", exc)
        return

    if resp.status_code != 200:
        logger.error("HTTP %s fetching company facts: %s", resp.status_code, resp.text)
        return

    try:
        data = resp.json()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to parse company facts JSON: %s", exc)
        return

    facts = data.get("facts") or {}
    us_gaap = facts.get("us-gaap") or {}
    if not isinstance(us_gaap, dict):
        logger.error("No 'us-gaap' section found in Apple company facts.")
        return

    substrings = ("revenue", "sales", "gross", "cost")

    print("=== Apple (AAPL) XBRL Revenue / Sales / Gross / Cost Tags ===")
    for tag_name, tag_data in sorted(us_gaap.items()):
        if not isinstance(tag_name, str) or not isinstance(tag_data, dict):
            continue
        lower_name = tag_name.lower()
        if not any(s in lower_name for s in substrings):
            continue

        yearly = _get_10k_yearly_maxes(tag_data)
        if yearly:
            latest_year = max(yearly.keys())
            value = yearly[latest_year]
            print(f"{tag_name}: {value} ({latest_year})")
        else:
            print(f"{tag_name}: no 10-K annual values")

    print("=== End of Apple XBRL Diagnostic ===")


if __name__ == "__main__":
    main()

