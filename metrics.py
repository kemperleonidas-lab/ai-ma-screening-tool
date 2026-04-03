"""Agent 2: metrics calculator.

This agent:

- Reads ``financial_data`` from :class:`FinancialState`.
- Computes a set of core financial ratios and metrics.
- Handles missing data and division-by-zero gracefully by returning ``None``
  and appending warnings to ``errors`` instead of raising.
- Returns only the fields it owns: ``metrics`` and ``errors``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from state import FinancialState

logger = logging.getLogger(__name__)


def _safe_divide(
    numerator: Optional[float],
    denominator: Optional[float],
    label: str,
    errors: List[str],
) -> Optional[float]:
    """Divide numerator by denominator with graceful handling of edge cases."""

    if numerator is None or denominator is None:
        msg = f"Cannot compute {label}: missing numerator or denominator."
        logger.warning(msg)
        errors.append(msg)
        return None

    try:
        if denominator == 0:
            msg = f"Cannot compute {label}: denominator is zero."
            logger.warning(msg)
            errors.append(msg)
            return None
        return numerator / denominator
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Error computing {label}: {exc}"
        logger.exception(msg)
        errors.append(msg)
        return None


def run_metrics_agent(state: FinancialState) -> Dict[str, Any]:
    """Compute derived financial metrics from `financial_data` in the state."""

    # Start from existing errors so we don't lose prior context.
    errors: List[str] = list(state.get("errors", []))  # type: ignore[arg-type]

    financial_data = state.get("financial_data", {})  # type: ignore[assignment]
    logger.info("Running metrics agent with financial_data keys: %s", list(financial_data.keys()))

    income = financial_data.get("income_statement", {}) or {}
    balance = financial_data.get("balance_sheet", {}) or {}
    cash_flow = financial_data.get("cash_flow", {}) or {}

    revenues = income.get("revenues")
    gross_profit = income.get("gross_profit")
    operating_income = income.get("operating_income_loss")
    net_income = income.get("net_income_loss")

    # Absolute scale (millions) for reporting.
    metrics: Dict[str, Any] = {}
    metrics["revenues"] = revenues
    metrics["total_assets"] = balance.get("assets")

    assets_current = balance.get("assets_current")
    liabilities_current = balance.get("liabilities_current")
    total_liabilities = balance.get("total_liabilities")
    stockholders_equity = balance.get("stockholders_equity")

    operating_cash_flow = cash_flow.get("net_cash_from_operating_activities")
    capital_expenditures = cash_flow.get("capital_expenditures")

    # Margins (as percentages, 2 decimals).
    gross_margin_ratio = _safe_divide(gross_profit, revenues, "gross_margin", errors)
    if gross_margin_ratio is not None:
        metrics["gross_margin"] = round(gross_margin_ratio * 100.0, 2)
    else:
        metrics["gross_margin"] = None

    operating_margin_ratio = _safe_divide(
        operating_income, revenues, "operating_margin", errors
    )
    if operating_margin_ratio is not None:
        metrics["operating_margin"] = round(operating_margin_ratio * 100.0, 2)
    else:
        metrics["operating_margin"] = None

    net_margin_ratio = _safe_divide(net_income, revenues, "net_margin", errors)
    if net_margin_ratio is not None:
        metrics["net_margin"] = round(net_margin_ratio * 100.0, 2)
    else:
        metrics["net_margin"] = None

    # Free cash flow = operating cash flow - capital expenditures.
    if operating_cash_flow is None or capital_expenditures is None:
        msg = (
            "Cannot compute free_cash_flow: missing operating_cash_flow "
            "or capital_expenditures."
        )
        logger.warning(msg)
        errors.append(msg)
        metrics["free_cash_flow"] = None
    else:
        try:
            metrics["free_cash_flow"] = round(
                operating_cash_flow - capital_expenditures, 2
            )
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Error computing free_cash_flow: {exc}"
            logger.exception(msg)
            errors.append(msg)
            metrics["free_cash_flow"] = None

    # Liquidity & leverage ratios.
    current_ratio = _safe_divide(
        assets_current, liabilities_current, "current_ratio", errors
    )
    metrics["current_ratio"] = (
        round(current_ratio, 2) if current_ratio is not None else None
    )

    debt_to_equity = _safe_divide(
        total_liabilities, stockholders_equity, "debt_to_equity", errors
    )
    metrics["debt_to_equity"] = (
        round(debt_to_equity, 2) if debt_to_equity is not None else None
    )

    return_on_equity = _safe_divide(
        net_income, stockholders_equity, "return_on_equity", errors
    )
    metrics["return_on_equity"] = (
        round(return_on_equity, 2) if return_on_equity is not None else None
    )

    # Cash conversion ratio (operating cash flow / net income).
    cash_conversion_ratio = _safe_divide(
        operating_cash_flow, net_income, "cash_conversion_ratio", errors
    )
    metrics["cash_conversion_ratio"] = (
        round(cash_conversion_ratio, 2) if cash_conversion_ratio is not None else None
    )

    interest_expense = income.get("interest_expense")
    interest_coverage = _safe_divide(
        operating_income, interest_expense, "interest_coverage_ratio", errors
    )
    metrics["interest_coverage_ratio"] = (
        round(interest_coverage, 2) if interest_coverage is not None else None
    )

    logger.info("Metrics agent completed with metrics: %s", metrics)

    return {
        "metrics": metrics,
        "errors": errors,
    }

