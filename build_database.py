from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

"""Build a ChromaDB vector database of peer company financial metrics.

This script:

- Defines peer tickers for Technology Software and Technology Hardware (canonical sectors).
- Reuses the existing EDGAR XBRL parsing logic (Agent 1) to fetch each
  company's latest 10-K financial data.
- Reuses the metrics agent (Agent 2) to compute financial ratios.
- Formats a human-readable metrics summary for each company.
- Stores each summary as a document in a local ChromaDB vector store with
  appropriate metadata.
- Maintains a simple JSON registry of successfully processed tickers per sector.

All SEC requests go through the existing parser agent, which already sets
the required User-Agent header.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import yfinance as yf

from state import get_initial_state
from agents.parser import run_parser_agent
from agents.metrics import run_metrics_agent
from utils.constants import CANONICAL_SECTORS, DEFAULT_PEER_SOFTWARE_SECTOR
from utils.market_cap_tiers import classify_market_cap_billions


logger = logging.getLogger(__name__)

# Ensure default peer sector stays aligned with the canonical list.
assert DEFAULT_PEER_SOFTWARE_SECTOR in CANONICAL_SECTORS

PEER_HARDWARE_SECTOR = "Technology Hardware"
assert PEER_HARDWARE_SECTOR in CANONICAL_SECTORS

# Retail peers (canonical sector: Retail).
PEER_RETAIL_SECTOR = "Retail"
assert PEER_RETAIL_SECTOR in CANONICAL_SECTORS

# Industrials peers (canonical sector: Industrials).
PEER_INDUSTRIALS_SECTOR = "Industrials"
assert PEER_INDUSTRIALS_SECTOR in CANONICAL_SECTORS

# Software / cloud / semiconductor design peers (canonical sector: Technology Software).
PEER_TICKERS_TECHNOLOGY_SOFTWARE: Dict[str, str] = {
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    # Added as additional benchmark peers (company name resolved from SEC facts).
    "WDAY": "",
    "INFA": "",
    "GTLB": "",
    "PCTY": "",
    "BRZE": "",
    "SPSC": "",
    "VRNT": "",
    "DOMO": "",
    "RELY": "",
}

# Hardware / equipment peers (canonical sector: Technology Hardware).
PEER_TICKERS_TECHNOLOGY_HARDWARE: Dict[str, str] = {
    "DELL": "Dell Technologies",
    "HPQ": "HP Inc.",
    "QCOM": "Qualcomm",
    "AVGO": "Broadcom",
    "TXN": "Texas Instruments",
    "INTC": "Intel",
    "STX": "Seagate Technology",
    "WDC": "Western Digital",
    "IBM": "IBM",
    "NVDA": "NVIDIA",
    # Added as additional benchmark peers (company name resolved from SEC facts).
    "ZBRA": "",
    "LSCC": "",
    "NOVT": "",
    "OSIS": "",
    "BHE": "",
    "PLPC": "",
    "DAKT": "",
    "VICR": "",
}

# Retail peer tickers (company name resolved from SEC facts).
PEER_TICKERS_RETAIL: Dict[str, str] = {
    "WMT": "",
    "AMZN": "",
    "COST": "",
    "HD": "",
    "TGT": "",
    "DG": "",
    "KSS": "",
    "GAP": "",
    "CATO": "",
    "URBN": "",
    "BOOT": "",
}

# Industrials peer tickers (company name resolved from SEC facts).
PEER_TICKERS_INDUSTRIALS: Dict[str, str] = {
    "HON": "",
    "CAT": "",
    "EMR": "",
    "ROK": "",
    "WTS": "",
    "IIIN": "",
    "HAYN": "",
    "NNBR": "",
    "MLI": "",
    "APOG": "",
}

# Consumers goods peers (canonical sector: Consumer Goods).
PEER_CONSUMER_GOODS_SECTOR = "Consumer Goods"
assert PEER_CONSUMER_GOODS_SECTOR in CANONICAL_SECTORS

# Consumer goods / consumer staples / household and apparel / consumer packaged goods
# peers. Company names are resolved from SEC entityName when the dict value is "".
PEER_TICKERS_CONSUMER_GOODS: Dict[str, str] = {
    "PG": "",
    "CL": "",
    "CHD": "",
    "SPB": "",
    "ENR": "",
    "PBH": "",
    "CENT": "",
    "ODC": "",
    "SKIN": "",
}

# Telecommunications peers (canonical sector: Telecommunications).
PEER_TELECOMMUNICATIONS_SECTOR = "Telecommunications"
assert PEER_TELECOMMUNICATIONS_SECTOR in CANONICAL_SECTORS

# Telecommunications peers: use SEC entityName as authoritative company_name when dict value is "".
PEER_TICKERS_TELECOMMUNICATIONS: Dict[str, str] = {
    "T": "",
    "VZ": "",
    "TMUS": "",
    "CHTR": "",
    "CNSL": "",
    "SHEN": "",
    "OOMA": "",
    "IRDM": "",
    "CABO": "",
}

DEFAULT_SECTOR = DEFAULT_PEER_SOFTWARE_SECTOR

RAG_DIR = Path("rag")
CHROMA_DIR = RAG_DIR / "chroma_db"
REGISTRY_PATH = RAG_DIR / "peer_registry.json"


def _load_registry(path: Path) -> Dict[str, Any]:
    """Load an existing peer registry file or return an empty structure."""

    if not path.exists():
        logger.info("Peer registry file does not exist yet at %s; creating fresh.", path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("Peer registry file at %s is not a dict; resetting.", path)
            return {}
        return data
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load peer registry from %s: %s", path, exc)
        return {}


def _save_registry(path: Path, registry: Dict[str, Any]) -> None:
    """Persist the peer registry to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    logger.info("Saved peer registry to %s", path)


def _format_value(value: Any, suffix: str = "", is_percent: bool = False) -> str:
    """Format a numeric or None value for human-readable output."""

    if value is None:
        return "N/A"

    try:
        num = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if is_percent:
        return f"{num:.2f}%"

    if suffix:
        return f"{num:,.2f}{suffix}"

    return f"{num:,.2f}"


def _build_peer_chunk(
    company_name: str,
    ticker: str,
    sector: str,
    metrics: Dict[str, Any],
) -> str:
    """Create a human-readable peer metrics text chunk for a company."""

    gross_margin = _format_value(metrics.get("gross_margin"), is_percent=True)
    operating_margin = _format_value(metrics.get("operating_margin"), is_percent=True)
    net_margin = _format_value(metrics.get("net_margin"), is_percent=True)
    current_ratio = _format_value(metrics.get("current_ratio"))
    debt_to_equity = _format_value(metrics.get("debt_to_equity"))
    roe = _format_value(metrics.get("return_on_equity"))
    free_cash_flow = _format_value(metrics.get("free_cash_flow"), suffix="M")
    cash_conversion = _format_value(metrics.get("cash_conversion_ratio"))
    interest_coverage = _format_value(metrics.get("interest_coverage_ratio"))

    chunk = (
        f"Company: {company_name} ({ticker}) | "
        f"Sector: {sector} | "
        f"Gross Margin: {gross_margin} | "
        f"Operating Margin: {operating_margin} | "
        f"Net Margin: {net_margin} | "
        f"Current Ratio: {current_ratio} | "
        f"Debt to Equity: {debt_to_equity} | "
        f"ROE: {roe} | "
        f"Free Cash Flow: ${free_cash_flow} | "
        f"Cash Conversion: {cash_conversion} | "
        f"Interest Coverage: {interest_coverage}"
    )

    return chunk


def _build_vector_store() -> Chroma:
    """Create or open the ChromaDB vector store."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="peer_metrics",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vector_store


def _classify_market_cap(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    """Fetch Yahoo Finance market cap in billions and classify size tier."""

    try:
        info = yf.Ticker(str(ticker).strip().upper()).info or {}
        market_cap_raw = info.get("marketCap")
        if market_cap_raw is None:
            return None, None

        market_cap_billions = round(float(market_cap_raw) / 1_000_000_000.0, 2)
        tier = classify_market_cap_billions(market_cap_billions)
        return market_cap_billions, tier
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Market cap classification failed for ticker %s (continuing): %s",
            ticker,
            exc,
        )
        return None, None


def process_peer_company(
    ticker: str,
    company_name: str,
    sector: str,
    vector_store: Chroma,
) -> Tuple[bool, Dict[str, Any], str]:
    """Fetch, compute, and store metrics for a single peer company.

    Returns (success, metadata, failure_reason) where `metadata` contains at
    least ticker, company_name, and sector. On success, `failure_reason` is
    an empty string.
    """

    logger.info("Processing peer company %s (%s)", company_name, ticker)

    # Initialize a minimal state and run Agent 1 (parser) then Agent 2 (metrics).
    state = get_initial_state(
        company_name=company_name,
        industry=sector,
        ticker=ticker,
    )

    parser_update = run_parser_agent(state)
    state.update(parser_update)

    # Accuracy guard: after the SEC parser runs, `state["company_name"]` should be the
    # canonical registrant display name (entityName) when the input was blank.
    resolved_company_name = str(state.get("company_name") or "").strip()
    if not resolved_company_name:
        reason = f"SEC company_name resolution failed for ticker={ticker}; refusing to store peer."
        logger.error(reason)
        return (
            False,
            {"ticker": ticker, "company_name": resolved_company_name, "sector": sector},
            reason,
        )

    metrics_update = run_metrics_agent(state)
    state.update(metrics_update)

    financial_data = state.get("financial_data") or {}
    metrics = state.get("metrics") or {}
    errors = state.get("errors") or []

    if not financial_data or not metrics:
        reason = (
            f"No financial_data or metrics produced. Errors: {errors}"
        )
        logger.error("FAILED [%s]: %s", ticker, reason)
        return (
            False,
            {"ticker": ticker, "company_name": resolved_company_name, "sector": sector},
            reason,
        )

    # Validate that key margins are within reasonable bounds.
    gross_margin = metrics.get("gross_margin")
    operating_margin = metrics.get("operating_margin")
    net_margin = metrics.get("net_margin")

    def _out_of_bounds(value: Any) -> bool:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return v > 100.0 or v < -100.0

    if any(_out_of_bounds(v) for v in (gross_margin, operating_margin, net_margin)):
        reason = (
            "Out-of-bounds margins: "
            f"gross_margin={gross_margin!r}, "
            f"operating_margin={operating_margin!r}, "
            f"net_margin={net_margin!r}"
        )
        logger.error("FAILED [%s]: %s", ticker, reason)
        return (
            False,
            {"ticker": ticker, "company_name": resolved_company_name, "sector": sector},
            reason,
        )

    chunk = _build_peer_chunk(resolved_company_name, ticker, sector, metrics)
    market_cap_billions, size_tier = _classify_market_cap(ticker)
    metadata = {
        "ticker": ticker,
        "company_name": resolved_company_name,
        "sector": sector,
        # Fiscal year is not explicitly tracked yet; placeholder for future enrichment.
        "fiscal_year": None,
        "market_cap_billions": market_cap_billions,
        "size_tier": size_tier,
        "last_classified": date.today().isoformat(),
    }

    logger.info("Storing peer document for %s (%s) in ChromaDB.", resolved_company_name, ticker)
    try:
        vector_store.add_texts(
            texts=[chunk],
            metadatas=[metadata],
            ids=[f"{ticker}"],
        )
    except Exception as exc:  # pragma: no cover - defensive
        reason = f"Failed to add peer document to ChromaDB: {exc}"
        logger.error("FAILED [%s]: %s", ticker, reason)
        logger.exception(
            "Failed to add peer document for %s (%s) to ChromaDB: %s",
            resolved_company_name,
            ticker,
            exc,
        )
        return False, metadata, reason

    return True, metadata, ""


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting build_database script for peer metrics.")

    vector_store = _build_vector_store()
    registry = _load_registry(REGISTRY_PATH)

    peer_groups: Tuple[Tuple[str, Dict[str, str]], ...] = (
        (DEFAULT_PEER_SOFTWARE_SECTOR, PEER_TICKERS_TECHNOLOGY_SOFTWARE),
        (PEER_HARDWARE_SECTOR, PEER_TICKERS_TECHNOLOGY_HARDWARE),
        (PEER_RETAIL_SECTOR, PEER_TICKERS_RETAIL),
        (PEER_INDUSTRIALS_SECTOR, PEER_TICKERS_INDUSTRIALS),
        (PEER_CONSUMER_GOODS_SECTOR, PEER_TICKERS_CONSUMER_GOODS),
        (PEER_TELECOMMUNICATIONS_SECTOR, PEER_TICKERS_TELECOMMUNICATIONS),
    )

    # Fresh peer lists each run (avoids stale tickers e.g. after moving INTC to Hardware).
    for sector_name, _ in peer_groups:
        registry[sector_name] = []

    # Remove previously-stored peer docs for these tickers so sector retagging doesn't leave duplicates.
    all_peer_ids: List[str] = []
    for _, peers in peer_groups:
        all_peer_ids.extend(list(peers.keys()))
    all_peer_ids = sorted(set(all_peer_ids))
    if all_peer_ids:
        try:
            vector_store.delete(ids=all_peer_ids)
            logger.info("Deleted %d previous peer documents from ChromaDB.", len(all_peer_ids))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Peer doc deletion skipped/failed (continuing): %s", exc)

    success_count = 0
    failure_count = 0
    failures: Dict[str, str] = {}

    for sector, peers in peer_groups:
        for ticker, company_name in peers.items():
            try:
                success, metadata, reason = process_peer_company(
                    ticker=ticker,
                    company_name=company_name,
                    sector=sector,
                    vector_store=vector_store,
                )
                if success:
                    success_count += 1
                    sector_key = metadata.get("sector", sector)
                    registry.setdefault(sector_key, [])
                    registry[sector_key].append(
                        {
                            "ticker": ticker,
                            "company_name": metadata.get("company_name", company_name),
                            "market_cap_billions": metadata.get("market_cap_billions"),
                            "size_tier": metadata.get("size_tier"),
                            "last_classified": metadata.get("last_classified"),
                        }
                    )
                    logger.info("Successfully processed peer %s (%s).", company_name, ticker)
                else:
                    failure_count += 1
                    failure_reason = reason or "Unknown processing failure."
                    failures[ticker] = failure_reason
                    logger.error("FAILED [%s]: %s", ticker, failure_reason)
            except Exception as exc:  # pragma: no cover - defensive
                failure_count += 1
                failure_reason = f"Unexpected error: {exc}"
                failures[ticker] = failure_reason
                logger.error("FAILED [%s]: %s", ticker, failure_reason)
                logger.exception(
                    "Unexpected error while processing peer %s (%s): %s",
                    company_name,
                    ticker,
                    exc,
                )
                continue

    # Persist registry only; ChromaDB persistence is handled by langchain-chroma.
    # Remove legacy non-canonical sector key from older runs.
    if "Technology" in registry:
        logger.info("Removing legacy peer_registry key 'Technology' (canonical: %s).", DEFAULT_SECTOR)
        del registry["Technology"]

    _save_registry(REGISTRY_PATH, registry)

    # Failure report.
    print("=== Failure Report ===")
    if failures:
        for ticker, reason in failures.items():
            print(f"{ticker}: {reason}")
    else:
        print("No failures.")

    # Final summary.
    print("=== Peer Metrics Build Summary ===")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process:     {failure_count}")
    print("Current peer registry:")
    print(json.dumps(registry, indent=2, sort_keys=True))
    print("=== End of Summary ===")


if __name__ == "__main__":
    main()

