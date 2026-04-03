"""Agent 1: 10-K parser using SEC EDGAR XBRL company facts API.

This agent:

- Accepts a full :class:`FinancialState` object as input.
- Reads the stock ticker from ``state["ticker"]``.
- Uses the SEC EDGAR search-index API to resolve the company's CIK.
- Uses the SEC EDGAR company facts API to fetch the latest 10-K XBRL data.
- Extracts key financial figures into the ``financial_data`` structure.
- Computes a ``parsing_confidence`` score based on coverage of target fields.
- Returns only the fields it owns:
  ``documents_parsed``, ``financial_data``, ``parsing_confidence``, ``errors``,
  ``sic_code``, ``sic_description``, and optionally ``company_name`` / ``industry``
  when resolved from EDGAR or SIC/GPT sector detection.

All HTTP calls use the required SEC User-Agent header.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
import yfinance as yf
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from state import FinancialState
from utils.constants import CANONICAL_SECTORS
from utils.market_cap_tiers import classify_market_cap_billions
from utils.sector_utils import normalize_to_canonical_sector

logger = logging.getLogger(__name__)

SEC_USER_AGENT = "financial-analyzer contact@example.com"
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "application/json",
}

BROWSE_EDGAR_URL_TEMPLATE = (
    "https://www.sec.gov/cgi-bin/browse-edgar?"
    "action=getcompany&company=&CIK={ticker}&type=10-K&dateb=&"
    "owner=include&count=10&search_text=&output=atom"
)
SEARCH_URL_TEMPLATE = (
    "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"
)
COMPANY_FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"

def _safe_get_json(url: str, description: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    """Helper to GET JSON from a URL with SEC headers and basic error handling."""

    logger.info("Requesting %s from %s", description, url)
    try:
        response = requests.get(url, headers=SEC_HEADERS, timeout=30)
        logger.info(
            "HTTP %s response for %s: %s",
            response.status_code,
            description,
            response.text,
        )
        if response.status_code != 200:
            msg = f"Failed to fetch {description}: HTTP {response.status_code}"
            logger.error(msg)
            errors.append(msg)
            return None
        try:
            data = response.json()
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Failed to parse JSON for {description}: {exc}"
            logger.exception(msg)
            errors.append(msg)
            return None

        logger.info("Successfully fetched %s JSON.", description)
        return data

    except requests.RequestException as exc:  # pragma: no cover - defensive
        msg = f"Network error while fetching {description}: {exc}"
        logger.exception(msg)
        errors.append(msg)
        return None


def _extract_cik_from_search_results(data: Dict[str, Any]) -> Optional[str]:
    """Extract a CIK value from SEC search-index results using hits._source."""

    logger.info("Attempting to extract CIK from search-index response (hits array).")

    try:
        hits_root = data.get("hits", {})
        hits = hits_root.get("hits", []) or []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            source = hit.get("_source") or {}
            if not isinstance(source, dict):
                continue
            period = source.get("period_of_report")
            entity_id = source.get("entity_id")
            logger.info(
                "Hit with period_of_report=%r, entity_id=%r", period, entity_id
            )
            if isinstance(entity_id, (str, int)):
                s = str(entity_id).strip()
                if s.isdigit():
                    logger.info("Selected CIK from entity_id: %s", s)
                    return s
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Error while extracting CIK from search-index hits: %s", exc)

    logger.warning("No CIK found in search-index hits array.")
    return None


def _fetch_cik_for_ticker(ticker: str, errors: List[str]) -> Optional[str]:
    """Resolve a stock ticker to a CIK using SEC APIs.

    First tries the browse-edgar Atom XML endpoint, then falls back to the
    search-index JSON endpoint if necessary.
    """

    logger.info("Resolving CIK for ticker %s via SEC APIs.", ticker)

    # First attempt: browse-edgar Atom XML.
    browse_url = BROWSE_EDGAR_URL_TEMPLATE.format(ticker=ticker.upper())
    try:
        browse_response = requests.get(browse_url, headers=SEC_HEADERS, timeout=30)
        logger.info(
            "HTTP %s response from browse-edgar for ticker %s: %s",
            browse_response.status_code,
            ticker,
            browse_response.text,
        )
        if browse_response.status_code == 200:
            import re
            import xml.etree.ElementTree as ET

            try:
                root = ET.fromstring(browse_response.text)
                cik_candidates: List[str] = []
                for elem in root.iter():
                    if elem.tag.endswith("id") and elem.text:
                        match = re.search(r"CIK=(\d+)", elem.text)
                        if match:
                            cik_candidates.append(match.group(1))
                if cik_candidates:
                    cik = sorted(cik_candidates, key=lambda s: (len(s), s))[0]
                    logger.info("Resolved CIK from browse-edgar Atom feed: %s", cik)
                    return cik
            except Exception as exc:  # pragma: no cover - defensive
                msg = f"Failed to parse browse-edgar Atom XML for ticker {ticker}: {exc}"
                logger.exception(msg)
                errors.append(msg)
        else:
            msg = (
                f"browse-edgar returned HTTP {browse_response.status_code} "
                f"for ticker {ticker}"
            )
            logger.error(msg)
            errors.append(msg)
    except requests.RequestException as exc:  # pragma: no cover - defensive
        msg = f"Network error while calling browse-edgar for ticker {ticker}: {exc}"
        logger.exception(msg)
        errors.append(msg)

    # Second attempt: search-index JSON.
    logger.info("Falling back to search-index API for ticker %s.", ticker)
    search_url = SEARCH_URL_TEMPLATE.format(ticker=ticker.upper())
    data = _safe_get_json(
        search_url, f"search-index results for ticker {ticker}", errors
    )
    if data is None:
        return None

    return _extract_cik_from_search_results(data)


def _fetch_company_facts(cik: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    """Fetch company facts JSON from the SEC company facts API."""

    cik_padded = cik.zfill(10)
    logger.info("Fetching company facts for CIK %s (padded: %s).", cik, cik_padded)
    url = COMPANY_FACTS_URL_TEMPLATE.format(cik=cik_padded)
    return _safe_get_json(url, f"company facts for CIK {cik_padded}", errors)


def _fetch_submissions_json(cik: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    """Fetch company metadata (including SIC) from SEC submissions API."""

    cik_padded = cik.zfill(10)
    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik_padded)
    return _safe_get_json(url, f"submissions for CIK {cik_padded}", errors)


def _parse_sic_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("none", "null", ""):
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _sic_code_to_sector(sic: int) -> Optional[str]:
    """Map numeric SIC to canonical sector; more-specific rules first."""

    # Technology Hardware (before Industrials overlap on 3510–3599)
    if (
        (3571 <= sic <= 3579)
        or (3669 <= sic <= 3679)
        or sic == 3812
        or sic == 3674
    ):
        return "Technology Hardware"
    # Technology Software (7371, 7372 fall inside 7370–7379)
    if 7370 <= sic <= 7379:
        return "Technology Software"
    if 2830 <= sic <= 2836:
        return "Pharma"
    if (6020 <= sic <= 6099) or (6110 <= sic <= 6199):
        return "Financial Services"
    if 5200 <= sic <= 5999:
        return "Retail"
    # Energy (spot codes that fall inside 1000–1499 must win over Metals)
    if sic == 1311 or (1381 <= sic <= 1389) or sic == 2911 or sic == 2990:
        return "Energy"
    # Industrials: 3440–3499 and 3510–3599 excluding Hardware window 3571–3579
    if 3440 <= sic <= 3499:
        return "Industrials"
    if 3510 <= sic <= 3570:
        return "Industrials"
    if 3580 <= sic <= 3599:
        return "Industrials"
    if 8000 <= sic <= 8099:
        return "Healthcare"
    if 4810 <= sic <= 4899:
        return "Telecommunications"
    if (2000 <= sic <= 2099) or (2100 <= sic <= 2199):
        return "Consumer Goods"
    if 6500 <= sic <= 6599:
        return "Real Estate"
    if 1000 <= sic <= 1499:
        return "Metals and Mining"
    return None


def _normalize_gpt_sector(text: str) -> str:
    """Match GPT output to an allowed canonical sector label."""

    return normalize_to_canonical_sector(text, default="Industrials")


def _classify_sector_gpt(
    company_name: str,
    sic_description: str,
    errors: List[str],
) -> str:
    """Single GPT-4o-mini call when SIC does not match the static mapping."""

    sector_list = ", ".join(CANONICAL_SECTORS)
    prompt = (
        f"Based on the company name {company_name} and SEC SIC description "
        f"{sic_description}, classify this company into exactly one of these sectors: "
        f"{sector_list}. "
        "Return only the sector name, nothing else."
    )
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        sector = _normalize_gpt_sector(raw)
        logger.info("GPT sector classifier raw=%r normalized=%s", raw, sector)
        return sector
    except Exception as exc:
        logger.exception("GPT sector classification failed: %s", exc)
        errors.append(f"GPT sector classification failed: {exc}")
        return "Industrials"


def _official_company_name_from_facts(facts: Dict[str, Any]) -> Optional[str]:
    """Return the registrant display name from the SEC company facts API payload.

    The JSON includes a top-level ``entityName`` field (official filer name).
    """

    name = facts.get("entityName")
    if isinstance(name, str):
        s = name.strip()
        if s:
            return s
    return None


def _fetch_market_cap_data(
    ticker: str, errors: List[str]
) -> Tuple[Optional[float], Optional[str]]:
    """Fetch market cap from Yahoo Finance and derive size tier.

    Returns
    -------
    Tuple[Optional[float], Optional[str]]
        (market_cap_billions, size_tier) where both are None on failure.
    """

    try:
        stock = yf.Ticker(str(ticker).strip().upper())
        info = stock.info or {}
        market_cap_raw = info.get("marketCap")
        if market_cap_raw is None:
            msg = (
                f"Info: Yahoo Finance market cap unavailable for ticker {ticker}; "
                "setting market_data fields to None."
            )
            logger.warning(msg)
            errors.append(msg)
            return None, None

        market_cap_billions = round(float(market_cap_raw) / 1_000_000_000.0, 2)
        size_tier = classify_market_cap_billions(market_cap_billions)
        return market_cap_billions, size_tier
    except Exception as exc:  # pragma: no cover - defensive
        msg = (
            f"Info: Yahoo Finance market cap fetch failed for ticker {ticker}: {exc}; "
            "setting market_data fields to None."
        )
        logger.warning(msg)
        errors.append(msg)
        return None, None


def _get_10k_yearly_maxes(tag_data: Dict[str, Any]) -> Dict[int, float]:
    """Return per-fiscal-year max values for a single tag (10-K only)."""

    units = tag_data.get("units")
    if not isinstance(units, dict):
        return {}

    per_year_max: Dict[int, float] = {}

    for unit_name, datapoints in units.items():
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


def _get_latest_10k_value(tag_data: Dict[str, Any]) -> Optional[float]:
    """From a single tag's data, get the largest value for the latest 10-K year."""

    per_year_max = _get_10k_yearly_maxes(tag_data)
    if not per_year_max:
        return None

    latest_year = max(per_year_max.keys())
    return per_year_max[latest_year]


def _get_tag_value(
    us_gaap: Dict[str, Any],
    primary_tag: str,
    alt_tags: List[str],
    display_name: str,
    errors: List[str],
) -> Optional[float]:
    """Lookup a primary tag, falling back to alternative tags if needed."""

    tags_to_try = [primary_tag] + alt_tags
    for tag in tags_to_try:
        tag_data = us_gaap.get(tag)
        if not isinstance(tag_data, dict):
            continue

        value = _get_latest_10k_value(tag_data)
        if value is not None:
            logger.info("Extracted %s from tag %s: %s", display_name, tag, value)
            return value

    msg = f"Missing XBRL data for {display_name} (tried: {', '.join(tags_to_try)})."
    logger.warning(msg)
    errors.append(msg)
    return None


def _get_latest_10k_max_across_tags(
    us_gaap: Dict[str, Any],
    tags: List[str],
    display_name: str,
    errors: List[str],
) -> Optional[float]:
    """Get the largest 10-K value for the latest fiscal year across multiple tags.

    For each tag we compute per-fiscal-year maxima, then:
    - Combine across tags taking the max per fiscal year.
    - Choose the latest fiscal year.
    - Return the largest value for that year.

    This is especially important for revenue, where consolidated totals may
    appear under different tag names or alongside smaller segment amounts.
    """

    combined_per_year: Dict[int, float] = {}

    for tag in tags:
        tag_data = us_gaap.get(tag)
        if not isinstance(tag_data, dict):
            continue

        yearly = _get_10k_yearly_maxes(tag_data)
        for fy, value in yearly.items():
            current = combined_per_year.get(fy)
            if current is None or value > current:
                combined_per_year[fy] = value

    if not combined_per_year:
        msg = f"Missing XBRL data for {display_name} (tried: {', '.join(tags)})."
        logger.warning(msg)
        errors.append(msg)
        return None

    latest_year = max(combined_per_year.keys())
    value = combined_per_year[latest_year]
    logger.info(
        "Extracted %s for latest fiscal year %s across tags %s: %s",
        display_name,
        latest_year,
        tags,
        value,
    )
    return value


def _diagnose_revenue_and_gross_tags(ticker: str, facts: Dict[str, Any]) -> None:
    """Temporary diagnostic: log all revenue / sales / gross / cost tags for AAPL.

    For ticker 'AAPL' only, this walks the us-gaap section and logs every tag whose
    name contains any of: 'Revenue', 'Sales', 'Gross', 'Cost' (case-insensitive),
    along with the most recent annual 10-K value if available.
    """

    if str(ticker).upper() != "AAPL":
        return

    logger.info("=== AAPL XBRL revenue/gross diagnostic start ===")
    facts_root = facts.get("facts") or {}
    us_gaap = facts_root.get("us-gaap") or {}
    if not isinstance(us_gaap, dict):
        logger.info("No us-gaap section available for diagnostic.")
        logger.info("=== AAPL XBRL revenue/gross diagnostic end ===")
        return

    substrings = ("revenue", "sales", "gross", "cost")
    for tag_name, tag_data in us_gaap.items():
        if not isinstance(tag_name, str) or not isinstance(tag_data, dict):
            continue
        lower_name = tag_name.lower()
        if not any(s in lower_name for s in substrings):
            continue

        yearly = _get_10k_yearly_maxes(tag_data)
        if yearly:
            latest_year = max(yearly.keys())
            value = yearly[latest_year]
            logger.info(
                "AAPL DIAG tag=%s latest_fy=%s value=%s",
                tag_name,
                latest_year,
                value,
            )
        else:
            logger.info("AAPL DIAG tag=%s has no 10-K annual values.", tag_name)

    logger.info("=== AAPL XBRL revenue/gross diagnostic end ===")


def _get_revenue_candidates_for_latest_year(
    us_gaap: Dict[str, Any],
    revenue_tags: List[str],
) -> Tuple[Optional[int], List[Tuple[float, str]]]:
    """Get (latest_fiscal_year, list of (value, tag) for that year, sorted by value descending)."""
    combined_per_year: Dict[int, Dict[str, float]] = {}  # fy -> {tag: value}

    for tag in revenue_tags:
        tag_data = us_gaap.get(tag)
        if not isinstance(tag_data, dict):
            continue
        yearly = _get_10k_yearly_maxes(tag_data)
        for fy, value in yearly.items():
            if fy not in combined_per_year:
                combined_per_year[fy] = {}
            combined_per_year[fy][tag] = value

    if not combined_per_year:
        return None, []

    latest_year = max(combined_per_year.keys())
    by_tag = combined_per_year[latest_year]
    candidates = [(v, tag) for tag, v in by_tag.items()]
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return latest_year, candidates


def _build_financial_data_from_facts(
    facts: Dict[str, Any],
    errors: List[str],
    industry: str = "",
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    """Construct the financial_data structure from company facts.

    industry is used to prefer RevenueFromContractWithCustomerExcludingAssessedTax
    for technology companies when both Revenues and that tag are available.
    """

    logger.info("Building financial_data structure from company facts.")

    financial_data: Dict[str, Dict[str, Any]] = {
        "balance_sheet": {},
        "income_statement": {},
        "cash_flow": {},
        "market_data": {},
        "meta": {},
    }

    facts_root = facts.get("facts") or {}
    us_gaap = facts_root.get("us-gaap") or {}
    if not isinstance(us_gaap, dict):
        msg = "Unexpected company facts format: 'facts.us-gaap' is missing or not an object."
        logger.error(msg)
        errors.append(msg)
        return financial_data, 0, 0

    targets = [
        # Income statement (other than revenues, which is handled specially below)
        (
            "income_statement",
            "net_income_loss",
            "NetIncomeLoss",
            [],
            "Net income (loss)",
        ),
        # Balance sheet (assets and some items; liabilities/equity handled specially)
        (
            "balance_sheet",
            "assets",
            "Assets",
            [],
            "Total assets",
        ),
        (
            "balance_sheet",
            "assets_current",
            "AssetsCurrent",
            [],
            "Current assets",
        ),
        (
            "balance_sheet",
            "liabilities_current",
            "LiabilitiesCurrent",
            [],
            "Current liabilities",
        ),
        (
            "balance_sheet",
            "long_term_debt",
            "LongTermDebtNoncurrent",
            [
                "LongTermDebt",
                "LongTermNotesPayable",
            ],
            "Long-term debt (noncurrent preferred)",
        ),
        (
            "balance_sheet",
            "stockholders_equity",
            "StockholdersEquity",
            [],
            "Stockholders' equity",
        ),
        (
            "balance_sheet",
            "cash_and_cash_equivalents",
            "CashAndCashEquivalentsAtCarryingValue",
            [],
            "Cash and cash equivalents",
        ),
        # Cash flow statement
        (
            "cash_flow",
            "net_cash_from_operating_activities",
            "NetCashProvidedByUsedInOperatingActivities",
            [],
            "Net cash from operating activities",
        ),
        (
            "cash_flow",
            "net_cash_from_investing_activities",
            "NetCashProvidedByUsedInInvestingActivities",
            [],
            "Net cash from investing activities",
        ),
        (
            "cash_flow",
            "net_cash_from_financing_activities",
            "NetCashProvidedByUsedInFinancingActivities",
            [],
            "Net cash from financing activities",
        ),
        # capital_expenditures handled specially below
    ]

    # Revenues: fetch all tag values for latest FY; apply tech preference and sanity check.
    revenue_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ]
    total_fields = len(targets) + 4  # include revenues, gross_profit, operating_income, capex
    found_fields = 0

    latest_fy, revenue_candidates = _get_revenue_candidates_for_latest_year(us_gaap, revenue_tags)
    revenue_candidates_sorted: List[float] = []  # values only, descending (for sanity check)
    revenues_value: Optional[float] = None

    if latest_fy is not None and revenue_candidates:
        revenue_candidates_sorted = [v for v, _ in revenue_candidates]
        by_tag = {tag: v for v, tag in revenue_candidates}
        ind_l = str(industry).strip().lower()
        is_technology = ind_l == "technology" or "technology" in ind_l
        if (
            is_technology
            and "Revenues" in by_tag
            and "RevenueFromContractWithCustomerExcludingAssessedTax" in by_tag
        ):
            revenues_value = by_tag["RevenueFromContractWithCustomerExcludingAssessedTax"]
            logger.info(
                "Preferring RevenueFromContractWithCustomerExcludingAssessedTax for technology company (latest FY %s): %s",
                latest_fy,
                revenues_value,
            )
        else:
            revenues_value = revenue_candidates_sorted[0]
            logger.info(
                "Revenues for latest fiscal year %s (max across tags): %s",
                latest_fy,
                revenues_value,
            )
    else:
        msg = f"Missing XBRL data for Revenues (tried: {', '.join(revenue_tags)})."
        logger.warning(msg)
        errors.append(msg)

    if revenues_value is not None:
        financial_data["income_statement"]["revenues"] = revenues_value
        found_fields += 1

    # Persist fiscal-year context for downstream reporting (best-effort).
    financial_data["meta"]["fiscal_year"] = latest_fy
    financial_data["meta"]["period_basis"] = "10-K"

    # Gross profit: prefer direct tags, otherwise compute as revenues - COGS.
    gross_profit_tags = [
        "GrossProfit",
        "GrossProfitLoss",
    ]
    gross_value = _get_latest_10k_max_across_tags(
        us_gaap=us_gaap,
        tags=gross_profit_tags,
        display_name="Gross profit",
        errors=errors,
    )
    if gross_value is None and revenues_value is not None:
        cogs_tags = [
            "CostOfRevenue",
            "CostOfGoodsSoldAndServicesSold",
            "CostOfGoodsSold",
        ]
        cogs_value = _get_latest_10k_max_across_tags(
            us_gaap=us_gaap,
            tags=cogs_tags,
            display_name="Cost of revenue / COGS",
            errors=errors,
        )
        if cogs_value is not None:
            gross_value = revenues_value - cogs_value
            logger.info(
                "Computed Gross profit as Revenues - COGS: %s - %s = %s",
                revenues_value,
                cogs_value,
                gross_value,
            )
    if gross_value is not None:
        financial_data["income_statement"]["gross_profit"] = gross_value
        found_fields += 1

    # Sanity check: gross profit cannot exceed revenue; if revenue > 1.5 * gross profit, flag and try next largest revenue tag.
    if revenues_value is not None and gross_value is not None and revenue_candidates_sorted:
        # revenue_candidates_sorted is descending (largest first); "next" = next index = next smaller value
        try:
            idx = revenue_candidates_sorted.index(revenues_value)
        except ValueError:
            idx = 0
        while revenues_value > 1.5 * gross_value:
            idx += 1
            if idx >= len(revenue_candidates_sorted):
                break
            revenues_value = revenue_candidates_sorted[idx]
            financial_data["income_statement"]["revenues"] = revenues_value
            logger.info("Using next largest revenue candidate: %s", revenues_value)

    # Operating income: combine several related tags and take max for latest FY.
    operating_income_tags = [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "OperatingIncome",
    ]
    operating_value = _get_latest_10k_max_across_tags(
        us_gaap=us_gaap,
        tags=operating_income_tags,
        display_name="Operating income (loss)",
        errors=errors,
    )
    if operating_value is not None:
        financial_data["income_statement"]["operating_income_loss"] = operating_value
        found_fields += 1

    # Interest expense: try tags in priority order; use latest 10-K annual value.
    interest_expense_tags = [
        "InterestExpense",
        "InterestAndDebtExpense",
        "InterestExpenseDebt",
    ]
    interest_expense_value: Optional[float] = None
    for tag in interest_expense_tags:
        per_year = _get_10k_yearly_maxes(us_gaap.get(tag, {}))
        if per_year:
            latest_year = max(per_year.keys())
            interest_expense_value = per_year[latest_year]
            logger.info(
                "Interest expense for latest fiscal year %s from tag %s: %s",
                latest_year,
                tag,
                interest_expense_value,
            )
            break
    # Store silently as None if all tags are missing.
    financial_data["income_statement"]["interest_expense"] = interest_expense_value
    if interest_expense_value is not None:
        found_fields += 1

    # Capital expenditures: use multiple capex-related tags.
    capex_tags = [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurringObligation",
        "PaymentsForCapitalImprovements",
        "AcquisitionOfPropertyPlantAndEquipment",
    ]
    capex_value = _get_latest_10k_max_across_tags(
        us_gaap=us_gaap,
        tags=capex_tags,
        display_name="Capital expenditures",
        errors=errors,
    )
    if capex_value is not None:
        financial_data["cash_flow"]["capital_expenditures"] = capex_value
        found_fields += 1

    # Total liabilities: combine several approaches.
    total_liab_value: Optional[float] = None
    used_total_liab_method: Optional[str] = None

    # 1) Direct Liabilities tag (try first; don't warn yet).
    liabilities_direct_per_year = _get_10k_yearly_maxes(us_gaap.get("Liabilities", {}))
    if liabilities_direct_per_year:
        latest_liab_year = max(liabilities_direct_per_year.keys())
        total_liab_value = liabilities_direct_per_year[latest_liab_year]
        used_total_liab_method = "liabilities_direct"

    # 2) LiabilitiesAndStockholdersEquity - StockholdersEquity.
    if total_liab_value is None:
        total_le_and_eq = _get_10k_yearly_maxes(
            us_gaap.get("LiabilitiesAndStockholdersEquity", {})
        )
        equity_per_year: Dict[int, float] = {}
        for eq_tag in (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ):
            eq_per_year_for_tag = _get_10k_yearly_maxes(us_gaap.get(eq_tag, {}))
            for fy, eq_val in eq_per_year_for_tag.items():
                prev = equity_per_year.get(fy)
                if prev is None or eq_val > prev:
                    equity_per_year[fy] = eq_val

        liabilities_diff_per_year: Dict[int, float] = {}
        for fy, total_val in total_le_and_eq.items():
            eq_val = equity_per_year.get(fy)
            if eq_val is None:
                continue
            liabilities_diff_per_year[fy] = max(
                liabilities_diff_per_year.get(fy, 0.0), total_val - eq_val
            )

        if liabilities_diff_per_year:
            latest_liab_year = max(liabilities_diff_per_year.keys())
            total_liab_value = liabilities_diff_per_year[latest_liab_year]
            used_total_liab_method = "liabilities_and_equity_diff"

    # 3) LiabilitiesCurrent + LiabilitiesNoncurrent.
    if total_liab_value is None:
        liab_current_per_year = _get_10k_yearly_maxes(us_gaap.get("LiabilitiesCurrent", {}))
        liab_noncurrent_per_year = _get_10k_yearly_maxes(us_gaap.get("LiabilitiesNoncurrent", {}))
        common_years = set(liab_current_per_year.keys()) & set(liab_noncurrent_per_year.keys())
        if common_years:
            latest_liab_year = max(common_years)
            total_liab_value = liab_current_per_year[latest_liab_year] + liab_noncurrent_per_year[latest_liab_year]
            used_total_liab_method = "liabilities_current_plus_noncurrent"

    # 4) Derived fallback: assets - stockholders_equity.
    if total_liab_value is None:
        assets_per_year = _get_10k_yearly_maxes(us_gaap.get("Assets", {}))
        equity_per_year: Dict[int, float] = {}
        for eq_tag in (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ):
            eq_per_year_for_tag = _get_10k_yearly_maxes(us_gaap.get(eq_tag, {}))
            for fy, eq_val in eq_per_year_for_tag.items():
                prev = equity_per_year.get(fy)
                if prev is None or eq_val > prev:
                    equity_per_year[fy] = eq_val

        common_years = set(assets_per_year.keys()) & set(equity_per_year.keys())
        if common_years:
            latest_liab_year = max(common_years)
            total_liab_value = assets_per_year[latest_liab_year] - equity_per_year[latest_liab_year]
            used_total_liab_method = "assets_minus_equity_fallback"

    if total_liab_value is not None:
        financial_data["balance_sheet"]["total_liabilities"] = total_liab_value
        found_fields += 1
        if used_total_liab_method == "assets_minus_equity_fallback":
            logger.info("Total liabilities derived from assets minus equity (fallback)")
    else:
        tags_to_try = [
            "Liabilities",
            "LiabilitiesAndStockholdersEquity - StockholdersEquity",
            "LiabilitiesCurrent + LiabilitiesNoncurrent",
            "Assets - StockholdersEquity",
        ]
        msg = f"Missing XBRL data for Total liabilities (tried: {', '.join(tags_to_try)})."
        logger.warning(msg)
        errors.append(msg)

    # Stockholders' equity: multiple equity tags.
    equity_tags = [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]
    equity_value = _get_latest_10k_max_across_tags(
        us_gaap=us_gaap,
        tags=equity_tags,
        display_name="Stockholders' equity",
        errors=errors,
    )
    if equity_value is not None:
        financial_data["balance_sheet"]["stockholders_equity"] = equity_value
        found_fields += 1

    for statement, key_name, primary_tag, alt_tags, display_name in targets:
        value = _get_tag_value(us_gaap, primary_tag, alt_tags, display_name, errors)
        if value is not None:
            financial_data[statement][key_name] = value
            found_fields += 1

    # Convert all numeric values from raw dollars to millions.
    for statement_name in ("balance_sheet", "income_statement", "cash_flow"):
        statement_data = financial_data.get(statement_name, {})
        for key, val in list(statement_data.items()):
            try:
                statement_data[key] = float(val) / 1_000_000.0
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping non-numeric value for %s.%s during millions conversion: %r",
                    statement_name,
                    key,
                    val,
                )

    # Guardrail: if LTD is parsed as exactly 0.0 for a large company, it's very likely
    # a tag-mapping/parsing failure rather than a true zero debt level.
    bs = financial_data.get("balance_sheet") or {}
    if isinstance(bs, dict):
        try:
            assets_millions = float(bs.get("assets") or 0.0)
        except (TypeError, ValueError):
            assets_millions = 0.0
        try:
            ltd_millions = float(bs.get("long_term_debt") or 0.0)
        except (TypeError, ValueError):
            ltd_millions = 0.0

        if assets_millions > 10_000 and ltd_millions == 0.0:
            msg = (
                "WARNING: long_term_debt parsed as 0.0 despite assets > $10B "
                "(fallback tag may still be missing); investigate XBRL LTD tags."
            )
            logger.warning(msg)
            errors.append(msg)

    logger.info(
        "Finished building financial_data (scaled to millions): found %d of %d target fields.",
        found_fields,
        total_fields,
    )
    return financial_data, found_fields, total_fields


def run_parser_agent(state: FinancialState) -> Dict[str, Any]:
    """Run the SEC EDGAR-based parser agent using the ticker in `state`.

    Parameters
    ----------
    state:
        The full :class:`FinancialState` dictionary for the current analysis run.

    Returns
    -------
    Dict[str, Any]
        A partial state update: ``documents_parsed``, ``financial_data``,
        ``parsing_confidence``, ``errors``, and optionally ``company_name`` when
        it was missing and resolved from company facts ``entityName``.
    """

    # Start from any existing errors so we don't drop upstream issues.
    errors: List[str] = list(state.get("errors", []))  # type: ignore[arg-type]

    ticker = state.get("ticker")  # type: ignore[assignment]
    logger.info("Running parser agent for ticker: %s", ticker)

    financial_data: Dict[str, Dict[str, Any]] = {
        "balance_sheet": {},
        "income_statement": {},
        "cash_flow": {},
        "market_data": {},
    }
    documents_parsed = False
    parsing_confidence = 0.0

    if not ticker:
        msg = "No ticker provided in state; cannot query SEC EDGAR."
        logger.error(msg)
        errors.append(msg)
        return {
            "documents_parsed": documents_parsed,
            "financial_data": financial_data,
            "parsing_confidence": parsing_confidence,
            "errors": errors,
        }

    # 1) Resolve CIK from ticker.
    cik = _fetch_cik_for_ticker(str(ticker), errors)
    if not cik:
        msg = f"Unable to resolve CIK for ticker {ticker}."
        logger.error(msg)
        errors.append(msg)
        return {
            "documents_parsed": documents_parsed,
            "financial_data": financial_data,
            "parsing_confidence": parsing_confidence,
            "errors": errors,
        }

    # 2) Fetch company facts for that CIK.
    facts = _fetch_company_facts(cik, errors)
    if facts is None:
        msg = f"Unable to fetch company facts for CIK {cik}."
        logger.error(msg)
        errors.append(msg)
        return {
            "documents_parsed": documents_parsed,
            "financial_data": financial_data,
            "parsing_confidence": parsing_confidence,
            "errors": errors,
        }

    # 2b) If no company name was provided, use the official name from EDGAR company facts.
    name_updates: Dict[str, Any] = {}
    existing_name = str(state.get("company_name") or "").strip()
    if not existing_name:
        official = _official_company_name_from_facts(facts)
        if official:
            name_updates["company_name"] = official
            logger.info("Set company_name from SEC company facts entityName: %s", official)
        else:
            logger.warning(
                "company_name was empty and entityName could not be read from company facts."
            )

    company_display_name = (
        name_updates.get("company_name")
        or existing_name
        or _official_company_name_from_facts(facts)
        or str(ticker)
    )

    # 2c) SEC submissions: SIC + sector (mapping or GPT); respect user-provided industry.
    user_industry = str(state.get("industry") or "").strip()
    sic_code_str = ""
    sic_desc_str = ""
    mapped_sector: Optional[str] = None
    submissions = _fetch_submissions_json(cik, errors)
    if submissions:
        sic_raw = submissions.get("sic")
        desc_raw = submissions.get("sicDescription")
        if desc_raw is not None:
            sic_desc_str = str(desc_raw).strip()
        if sic_raw is not None and str(sic_raw).strip() != "":
            sic_code_str = str(sic_raw).strip()
        sic_int = _parse_sic_int(sic_raw)
        logger.info(
            "SEC submissions: sic_code=%s sicDescription=%r",
            sic_code_str,
            sic_desc_str,
        )
        if sic_int is not None:
            mapped_sector = _sic_code_to_sector(sic_int)
            if mapped_sector:
                logger.info("Detected sector from SIC %s: %s", sic_int, mapped_sector)
            else:
                logger.info("SIC %s did not match a known mapping; may use GPT fallback.", sic_int)
    else:
        logger.warning("Submissions JSON unavailable; sic_code/sic_description left empty.")

    sector_updates: Dict[str, Any] = {
        "sic_code": sic_code_str,
        "sic_description": sic_desc_str,
    }
    if user_industry:
        effective_industry = user_industry
        logger.info("Keeping user-provided industry (not overriding): %s", user_industry)
    elif mapped_sector:
        sector_updates["industry"] = mapped_sector
        effective_industry = mapped_sector
        logger.info("Set industry from SIC mapping: %s", mapped_sector)
    else:
        gpt_sector = _classify_sector_gpt(
            company_display_name,
            sic_desc_str or "Unknown",
            errors,
        )
        sector_updates["industry"] = gpt_sector
        effective_industry = gpt_sector
        logger.info("Set industry from GPT sector classifier: %s", gpt_sector)

    # 3) Optional diagnostic for AAPL revenue / gross tags before normal extraction.
    _diagnose_revenue_and_gross_tags(str(ticker), facts)

    # 4) Build financial_data from the XBRL facts.
    financial_data, found_fields, total_fields = _build_financial_data_from_facts(
        facts, errors, industry=effective_industry
    )

    # 4b) Fetch current Yahoo Finance market cap after XBRL extraction.
    market_cap_billions, size_tier = _fetch_market_cap_data(str(ticker), errors)
    market_data = financial_data.setdefault("market_data", {})
    market_data["market_cap_billions"] = market_cap_billions
    market_data["size_tier"] = size_tier

    # 5) Compute parsing confidence.
    if total_fields > 0:
        parsing_confidence = min(1.0, found_fields / total_fields)
    else:
        parsing_confidence = 0.0

    documents_parsed = found_fields > 0

    logger.info(
        "Parser agent completed for ticker %s. documents_parsed=%s, "
        "found_fields=%d/%d, parsing_confidence=%.3f",
        ticker,
        documents_parsed,
        found_fields,
        total_fields,
        parsing_confidence,
    )

    out: Dict[str, Any] = {
        "documents_parsed": documents_parsed,
        "financial_data": financial_data,
        "parsing_confidence": parsing_confidence,
        "errors": errors,
    }
    out.update(name_updates)
    out.update(sector_updates)
    return out


