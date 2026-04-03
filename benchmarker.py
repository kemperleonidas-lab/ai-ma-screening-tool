"""Agent 3: Benchmark analyzer — compares target company metrics to peers via ChromaDB and LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from state import FinancialState
from rag.query import query_industry_averages, query_peers

logger = logging.getLogger(__name__)

METRIC_KEYS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "current_ratio",
    "debt_to_equity",
    "return_on_equity",
    "free_cash_flow",
    "cash_conversion_ratio",
    "interest_coverage_ratio",
]


def _build_query_string(industry: str, metrics: Dict[str, Any]) -> str:
    """Build a search query from industry and key metric names/values for semantic retrieval."""
    parts = [industry]
    for key in METRIC_KEYS:
        val = metrics.get(key)
        if val is not None:
            parts.append(f"{key} {val}")
    return " ".join(str(p) for p in parts)


def _filter_peers_excluding_target(docs: List[Any], target_ticker: str) -> List[Any]:
    """Drop documents whose metadata ticker matches the target company (case-insensitive)."""

    if not (target_ticker or "").strip():
        return docs
    want = target_ticker.strip().upper()
    out: List[Any] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        peer_t = str(meta.get("ticker", "")).strip().upper()
        if peer_t == want:
            logger.info("Excluding target company from peer set (self-match): ticker=%r", peer_t)
            continue
        out.append(d)
    return out


def _extract_peers_list(docs: List[Any]) -> List[Dict[str, Any]]:
    """Build list of peer company dicts (ticker, company_name, sector) from retrieved documents."""
    peers: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        ticker = meta.get("ticker")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        peers.append({
            "ticker": ticker,
            "company_name": meta.get("company_name", ticker),
            "sector": meta.get("sector", ""),
        })
    return peers


def run_benchmarker_agent(state: FinancialState) -> Dict[str, Any]:
    """
    Compare target company metrics to industry averages and direct peers.

    Query 1: industry_average docs only -> populate state.industry_benchmark.
    Query 2: direct peer company docs only -> populate state.benchmark_analysis.
    """
    metrics = state.get("metrics") or {}
    industry = (state.get("industry") or "").strip() or "Technology Software"
    target_ticker = (state.get("ticker") or "").strip()
    market_data = ((state.get("financial_data") or {}).get("market_data") or {})
    target_size_tier = str(market_data.get("size_tier") or "").strip() or None
    errors: List[str] = list(state.get("errors", []))  # type: ignore[arg-type]

    industry_benchmark: Dict[str, Any] = {}
    benchmark_analysis: Dict[str, Any] = {}
    peer_comparison: List[Dict[str, Any]] = []

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Ensure all benchmarked keys exist in the payload (even if None) so the LLM
    # is consistently asked to assess them.
    normalized_metrics: Dict[str, Any] = dict(metrics) if isinstance(metrics, dict) else {}
    for k in METRIC_KEYS:
        normalized_metrics.setdefault(k, None)
    target_metrics_str = json.dumps(normalized_metrics, indent=2)

    # -------------------------
    # Query 1: Industry averages
    # -------------------------
    docs_industry = []
    try:
        docs_industry = query_industry_averages(sector=industry, n_results=3)
    except Exception as exc:
        logger.exception("Chroma industry-average query failed: %s", exc)
        errors.append(f"Industry-average peer query failed: {exc}")

    if not docs_industry:
        msg = f"Industry-average comparison unavailable: no industry_average documents found for sector={industry!r}."
        logger.info(msg)
        errors.append(msg)
    else:
        industry_texts = [getattr(d, "page_content", "") or "" for d in docs_industry]
        industry_context = "\n\n---\n\n".join(industry_texts)

        system_content_industry = (
            "You are a financial analyst. You compare a target company's metrics to sector industry-average data. "
            "Return ONLY valid JSON with no markdown or explanation. "
            "Use exactly this structure:\n"
            '{"industry_benchmark": {"<metric_name>": {"industry_avg": <number or null>, "target_value": <number or null>, "assessment": "ABOVE"|"IN LINE"|"BELOW"|null}, ...}}\n'
            "For each metric in the target company's metrics JSON, compute the industry average by parsing numeric values from the provided industry-average document text. "
            "Compute the industry_avg as the average across all valid numeric values found in the provided documents for that metric. "
            "If fewer than 2 valid data points are available for a metric, set industry_avg to null and assessment to null for that metric. "
            "Set target_value from the target metrics JSON for each metric when possible. "
            "Classify assessment as ABOVE (target > industry_avg), BELOW (target < industry_avg), or IN LINE (target equals industry_avg). "
            "If industry_avg is null, set assessment to null. "
            "Return only the JSON object."
        )
        user_content_industry = (
            "Industry average documents (one block per document):\n\n"
            f"{industry_context}\n\n"
            "Target company metrics (JSON):\n"
            f"{target_metrics_str}\n\n"
            "Return one JSON object with industry_benchmark as described."
        )

        content = ""
        try:
            response = llm.invoke(
                [SystemMessage(content=system_content_industry), HumanMessage(content=user_content_industry)]
            )
            content = response.content if hasattr(response, "content") else str(response)
            if "```json" in content:
                content = content.split("```json", 1)[-1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[-1].split("```", 1)[0].strip()
            content = content.replace("'", '"')
            content = re.sub(r",\s*(?=[}\]])", "", content)
            while True:
                new_content = re.sub(r"(\d),(\d{3})", r"\1\2", content)
                if new_content == content:
                    break
                content = new_content
            data = json.loads(content)
            if isinstance(data.get("industry_benchmark"), dict):
                industry_benchmark = data["industry_benchmark"]
        except Exception as exc:
            logger.exception("Industry-average benchmark GPT failed: %s", exc)
            errors.append(f"Industry-average benchmark GPT failed: {exc}")

    # -------------------------
    # Query 2: Direct peer companies
    # -------------------------
    docs_peers = []
    peer_retrieval_mode = "sector_only"
    try:
        peer_query_result = query_peers(
            sector=industry,
            ticker_to_exclude=target_ticker,
            n_results=5,
            size_tier=target_size_tier,
        )
        docs_peers = list(peer_query_result.get("docs", []) or [])
        peer_retrieval_mode = str(peer_query_result.get("retrieval_mode") or "sector_only")
    except Exception as exc:
        logger.exception("Chroma peer query failed: %s", exc)
        errors.append(f"Peer-company query failed: {exc}")

    if peer_retrieval_mode == "sector_fallback" and target_size_tier:
        errors.append(
            "Size-matched peer cohort has fewer than 3 companies for "
            f"sector={industry} tier={target_size_tier} — benchmarked against full sector peer set instead."
        )

    if not docs_peers:
        msg = f"Direct peer comparison unavailable: no peer company documents found for sector={industry!r} (excluding ticker={target_ticker!r})."
        logger.info(msg)
        errors.append(msg)
    else:
        peer_comparison = _extract_peers_list(docs_peers)
        peer_texts = [getattr(d, "page_content", "") or "" for d in docs_peers]
        peer_context = "\n\n---\n\n".join(peer_texts)

        system_content_peer = (
            "You are a financial analyst. You compare a target company's metrics to peer companies. "
            "Return ONLY valid JSON with no markdown or explanation. "
            "Use exactly this structure:\n"
            '{"benchmark_analysis": {"<metric_name>": {"peer_average": <number or null>, "target_value": <number or null>, "assessment": "STRONG"|"AVERAGE"|"WEAK"|null}, ...}}\n'
            "For each metric in the target company, compute a peer average by parsing numeric values from the provided peer document text. "
            "When computing peer averages, EXCLUDE outlier peer values that are clearly distorted: "
            "for debt_to_equity, exclude any peer value strictly below -5 or strictly above 20; "
            "for return_on_equity (ROE), exclude any peer value strictly below -1 or strictly above 5; "
            "for free_cash_flow, exclude any peer value that is negative when forming the peer average. "
            "After applying these exclusions, if fewer than 2 valid peer data points remain for a metric, "
            "set peer_average to null and assessment to null for that metric. "
            "Otherwise set target_value from the target metrics JSON and classify assessment as STRONG (target > peer_average), "
            "AVERAGE (target equals peer_average), or WEAK (target < peer_average). "
            "Only include metrics that appear in the target metrics JSON. "
            "Return only the JSON object."
        )
        user_content_peer = (
            "Peer company documents (one block per company):\n\n"
            f"{peer_context}\n\n"
            "Target company metrics (JSON):\n"
            f"{target_metrics_str}\n\n"
            "Return one JSON object with benchmark_analysis as described."
        )

        content = ""
        try:
            response = llm.invoke([SystemMessage(content=system_content_peer), HumanMessage(content=user_content_peer)])
            content = response.content if hasattr(response, "content") else str(response)
            if "```json" in content:
                content = content.split("```json", 1)[-1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[-1].split("```", 1)[0].strip()
            content = content.replace("'", '"')
            content = re.sub(r",\s*(?=[}\]])", "", content)
            while True:
                new_content = re.sub(r"(\d),(\d{3})", r"\1\2", content)
                if new_content == content:
                    break
                content = new_content
            data = json.loads(content)
            if isinstance(data.get("benchmark_analysis"), dict):
                benchmark_analysis = data["benchmark_analysis"]
        except Exception as exc:
            logger.exception("Peer benchmark GPT failed: %s", exc)
            errors.append(f"Peer benchmark GPT failed: {exc}")

    # Omit metrics where the industry average is missing/null.
    if isinstance(industry_benchmark, dict):
        for metric_name in list(industry_benchmark.keys()):
            details = industry_benchmark.get(metric_name)
            if not isinstance(details, dict):
                del industry_benchmark[metric_name]
                continue
            if details.get("industry_avg") is None:
                del industry_benchmark[metric_name]

    logger.info(
        "Benchmarker completed: industry_metrics=%d peer_metrics=%d peers=%d",
        len(industry_benchmark),
        len(benchmark_analysis),
        len(peer_comparison),
    )

    # Strict size-tier policy: warn when cohort is thin (do not fall back here).
    MIN_SIZE_TIER_PEERS = 5
    if target_size_tier and peer_retrieval_mode == "size_matched" and len(peer_comparison) < MIN_SIZE_TIER_PEERS:
        errors.append(
            f"Thin size-tier cohort: only {len(peer_comparison)} peers for sector={industry} tier={target_size_tier} "
            f"(minimum recommended is {MIN_SIZE_TIER_PEERS})."
        )
    return {
        "industry_benchmark": industry_benchmark,
        "benchmark_analysis": benchmark_analysis,
        "peer_comparison": peer_comparison,
        "benchmark_peer_retrieval_mode": peer_retrieval_mode,
        "benchmark_size_tier": target_size_tier,
        "errors": errors,
    }
