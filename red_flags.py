"""Agent 4: red flag detector for M&A screening."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

from state import FinancialState
from utils.constants import get_thresholds

logger = logging.getLogger(__name__)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _severity_multiplier(threshold: float, actual: float, direction: str) -> float:
    """
    Scale flag weight based on how far the metric is from its threshold.
    direction='below' means lower is worse (e.g. gross_margin, current_ratio).
    direction='above' means higher is worse (e.g. debt_to_equity).
    Returns a multiplier between 1.0 and 1.5.
    """
    if threshold == 0:
        return 1.0
    if direction == "below":
        distance = (threshold - actual) / abs(threshold)
    else:
        distance = (actual - threshold) / abs(threshold)
    return round(1.0 + min(0.5, max(0.0, distance * 0.5)), 4)


def _standalone_checks(
    metrics: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> Tuple[List[str], Set[str], float]:
    """
    Run all standalone threshold checks using sector-specific thresholds.
    Returns (flag_strings, flagged_metric_names, total_points).
    """
    flags: List[str] = []
    flagged: Set[str] = set()
    points: float = 0.0

    # --- Operating margin (weight 3.0) ---
    operating_margin = _to_float(metrics.get("operating_margin"))
    threshold_op = thresholds["operating_margin_min"]
    if operating_margin is not None and operating_margin < threshold_op:
        mult = _severity_multiplier(threshold_op if threshold_op != 0 else -0.01,
                                     operating_margin, "below")
        contrib = round(3.0 * mult, 2)
        points += contrib
        flagged.add("operating_margin")
        flags.append(
            f"Operating margin of {operating_margin:.2f}% is below the "
            f"{threshold_op:.1f}% sector threshold, indicating loss-making or "
            f"severely pressured core operations — requires review of cost "
            f"structure and path to profitability. (score contribution: {contrib:.2f})"
        )

    # --- Free cash flow (weight 2.0) ---
    free_cash_flow = _to_float(metrics.get("free_cash_flow"))
    if free_cash_flow is not None and free_cash_flow < thresholds["free_cash_flow_min"]:
        mult = _severity_multiplier(1.0, abs(free_cash_flow) / max(1.0, abs(free_cash_flow) + 1.0), "above")
        contrib = round(2.0 * mult, 2)
        points += contrib
        flagged.add("free_cash_flow")
        flags.append(
            f"Free cash flow of ${free_cash_flow:,.0f}M is negative, indicating "
            f"cash burn — requires runway and funding needs assessment. "
            f"(score contribution: {contrib:.2f})"
        )

    # --- Current ratio (weight 2.0) ---
    current_ratio = _to_float(metrics.get("current_ratio"))
    threshold_cr = thresholds["current_ratio_min"]
    if current_ratio is not None and current_ratio < threshold_cr:
        mult = _severity_multiplier(threshold_cr, current_ratio, "below")
        contrib = round(2.0 * mult, 2)
        points += contrib
        flagged.add("current_ratio")
        flags.append(
            f"Current ratio of {current_ratio:.2f} is below the {threshold_cr:.1f} "
            f"sector threshold, signaling short-term liquidity risk — requires "
            f"working capital stress testing. (score contribution: {contrib:.2f})"
        )

    # --- Debt to equity (weight 2.0) ---
    debt_to_equity = _to_float(metrics.get("debt_to_equity"))
    threshold_de = thresholds["debt_to_equity_max"]
    if debt_to_equity is not None and debt_to_equity > threshold_de:
        mult = _severity_multiplier(threshold_de, debt_to_equity, "above")
        contrib = round(2.0 * mult, 2)
        points += contrib
        flagged.add("debt_to_equity")
        flags.append(
            f"Debt-to-equity of {debt_to_equity:.2f} exceeds the {threshold_de:.1f} "
            f"sector threshold, indicating high leverage — requires covenant, "
            f"refinancing, and solvency diligence. (score contribution: {contrib:.2f})"
        )

    # --- Gross margin (weight 1.5) ---
    gross_margin = _to_float(metrics.get("gross_margin"))
    threshold_gm = thresholds["gross_margin_min"]
    if gross_margin is not None and gross_margin < threshold_gm:
        mult = _severity_multiplier(threshold_gm, gross_margin, "below")
        contrib = round(1.5 * mult, 2)
        points += contrib
        flagged.add("gross_margin")
        flags.append(
            f"Gross margin of {gross_margin:.2f}% is below the {threshold_gm:.1f}% "
            f"sector threshold, indicating potential commoditization or structural "
            f"cost pressure — requires due diligence on pricing power. "
            f"(score contribution: {contrib:.2f})"
        )

    # --- Cash conversion ratio (weight 1.5) ---
    cash_conversion = _to_float(metrics.get("cash_conversion_ratio"))
    threshold_cc = thresholds["cash_conversion_min"]
    if cash_conversion is not None and cash_conversion < threshold_cc:
        mult = _severity_multiplier(threshold_cc, cash_conversion, "below")
        contrib = round(1.5 * mult, 2)
        points += contrib
        flagged.add("cash_conversion_ratio")
        flags.append(
            f"Cash conversion ratio of {cash_conversion:.2f} is below the "
            f"{threshold_cc:.1f} sector threshold, suggesting earnings quality "
            f"concerns — operating cash flow may not support reported profits. "
            f"(score contribution: {contrib:.2f})"
        )

    # --- Return on equity (weight 1.0) ---
    roe = _to_float(metrics.get("return_on_equity"))
    threshold_roe = thresholds["roe_min"]
    if roe is not None and roe < threshold_roe:
        mult = _severity_multiplier(threshold_roe, roe, "below")
        contrib = round(1.0 * mult, 2)
        points += contrib
        flagged.add("return_on_equity")
        flags.append(
            f"Return on equity of {roe:.4f} is below the {threshold_roe:.2f} "
            f"sector threshold, indicating weak shareholder returns — requires "
            f"evaluation of capital allocation efficiency. "
            f"(score contribution: {contrib:.2f})"
        )

    return flags, flagged, points


def _benchmark_checks(
    metrics: Dict[str, Any],
    benchmark_analysis: Dict[str, Any],
    industry_benchmark: Dict[str, Any],
    flagged_already: Set[str],
    peer_count: int,
) -> Tuple[List[str], float]:
    """
    Tiered benchmark confirmation checks.

    Tier 1 — Both signals confirm (WEAK vs peers AND BELOW vs industry):
        Gap >= threshold: 0.5 pts. Gap < threshold: 0.25 pts.

    Tier 2 — Single signal only, large gap:
        WEAK vs peers only, gap >= 2x margin threshold: 0.3 pts.
        BELOW vs industry only, no peer data: 0.25 pts.

    Peer count < 3 suppresses peer-based signals entirely.
    """
    flags: List[str] = []
    points: float = 0.0

    margin_metrics = {"gross_margin", "operating_margin", "net_margin"}
    ratio_metrics = {"current_ratio", "debt_to_equity", "return_on_equity",
                     "cash_conversion_ratio"}

    MARGIN_GAP_THRESHOLD = 15.0
    RATIO_GAP_THRESHOLD = 0.3

    confirmed_weak_count = 0

    all_metric_names = set(list(benchmark_analysis.keys()) + list(industry_benchmark.keys()))

    for metric_name in all_metric_names:
        if metric_name in flagged_already:
            continue

        peer_details = benchmark_analysis.get(metric_name) if isinstance(benchmark_analysis, dict) else None
        industry_details = industry_benchmark.get(metric_name) if isinstance(industry_benchmark, dict) else None

        peer_assessment = str((peer_details or {}).get("assessment", "")).upper() if isinstance(peer_details, dict) else ""
        industry_assessment = str((industry_details or {}).get("assessment", "")).upper() if isinstance(industry_details, dict) else ""

        target_value = _to_float((peer_details or industry_details or {}).get("target_value"))
        peer_average = _to_float((peer_details or {}).get("peer_average")) if isinstance(peer_details, dict) else None
        industry_avg = _to_float((industry_details or {}).get("industry_avg")) if isinstance(industry_details, dict) else None

        if metric_name in margin_metrics:
            gap_threshold = MARGIN_GAP_THRESHOLD
        elif metric_name in ratio_metrics:
            gap_threshold = RATIO_GAP_THRESHOLD
        else:
            continue

        peer_gap = abs(target_value - peer_average) if (target_value is not None and peer_average is not None) else None
        industry_gap = abs(target_value - industry_avg) if (target_value is not None and industry_avg is not None) else None

        has_peer = peer_count >= 3 and peer_assessment == "WEAK" and peer_gap is not None
        has_industry = industry_assessment in ("BELOW", "WEAK") and industry_gap is not None

        contrib = 0.0
        tier = None

        if has_peer and has_industry:
            # Tier 1: both confirm
            confirmed_weak_count += 1
            contrib = 0.5 if (peer_gap >= gap_threshold or industry_gap >= gap_threshold) else 0.25
            tier = 1
        elif has_peer and not has_industry and peer_gap is not None and peer_gap >= gap_threshold * 2:
            # Tier 2a: peer only, very large gap
            contrib = 0.3
            tier = 2
        elif has_industry and not has_peer and industry_gap is not None and industry_gap >= gap_threshold:
            # Tier 2b: industry only, no valid peer data
            contrib = 0.25
            tier = 2

        if contrib > 0 and tier is not None:
            points += contrib
            peer_str = f"WEAK vs peers (gap={peer_gap:.2f})" if peer_gap is not None and has_peer else "no peer signal"
            ind_str = f"BELOW vs industry (gap={industry_gap:.2f})" if industry_gap is not None and has_industry else "no industry signal"
            flags.append(
                f"Benchmark weakness on {metric_name}: {peer_str}, {ind_str} "
                f"[Tier {tier} confirmation] — may affect valuation in M&A negotiations. "
                f"(score contribution: {contrib:.2f})"
            )

    # Systemic penalty: 4+ metrics confirmed weak across both benchmarks
    if confirmed_weak_count >= 4:
        points += 1.0
        flags.append(
            f"Systemic underperformance: {confirmed_weak_count} metrics confirmed "
            f"weak across both peer and industry benchmarks — suggests structural "
            f"competitiveness issues beyond isolated metric weakness. "
            f"(score contribution: 1.00)"
        )

    return flags, points


def run_red_flags_agent(state: FinancialState) -> Dict[str, Any]:
    """Detect M&A red flags using weighted, sector-calibrated scoring."""

    errors: List[str] = list(state.get("errors", []))
    metrics = state.get("metrics") or {}
    benchmark_analysis = state.get("benchmark_analysis") or {}
    industry_benchmark = state.get("industry_benchmark") or {}
    industry = str(state.get("industry") or "")
    peer_count = len(state.get("peer_comparison") or [])

    thresholds = get_thresholds(industry)

    standalone_flags, flagged_metrics, standalone_points = _standalone_checks(
        metrics, thresholds
    )

    benchmark_flags, benchmark_points = _benchmark_checks(
        metrics, benchmark_analysis, industry_benchmark, flagged_metrics, peer_count
    )

    red_flags = standalone_flags + benchmark_flags
    risk_score = round(min(10.0, standalone_points + benchmark_points), 1)

    logger.info(
        "Red flags agent: %d flags, standalone_pts=%.2f, benchmark_pts=%.2f, "
        "risk_score=%.1f, sector=%s",
        len(red_flags), standalone_points, benchmark_points, risk_score, industry,
    )

    return {
        "red_flags": red_flags,
        "risk_score": risk_score,
        "errors": errors,
    }

