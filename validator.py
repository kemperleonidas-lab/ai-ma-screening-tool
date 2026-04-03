"""Validation layer between parser (Agent 1) and metrics (Agent 2)."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from state import FinancialState

logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = ("balance_sheet", "income_statement", "cash_flow")

# Must be non-null and non-zero (parser keys).
CRITICAL_NONZERO_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("revenues", "income_statement", "revenues"),
    ("gross_profit", "income_statement", "gross_profit"),
    ("net_income_loss", "income_statement", "net_income_loss"),
    ("assets", "balance_sheet", "assets"),
    ("net_cash_from_operating_activities", "cash_flow", "net_cash_from_operating_activities"),
)

EARNINGS_QUALITY_CRITICAL = (
    "CRITICAL: Earnings quality concern — positive net income not supported by "
    "operating cash flow, potential accounting quality issue"
)


def _is_non_null(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    return True


def _to_float(val: Any) -> Optional[float]:
    if not _is_non_null(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _section_nonempty(financial_data: Dict[str, Any], name: str) -> bool:
    sec = financial_data.get(name)
    if not isinstance(sec, dict):
        return False
    return len(sec) > 0


def run_validator_agent(state: FinancialState) -> Dict[str, Any]:
    """Validate parsed financial_data; set data_quality_score and possibly documents_parsed."""

    errors: List[str] = list(state.get("errors", []))  # type: ignore[arg-type]
    financial_data: Dict[str, Any] = dict(state.get("financial_data") or {})  # type: ignore[arg-type]
    documents_parsed: bool = bool(state.get("documents_parsed", False))
    parsing_confidence = float(state.get("parsing_confidence", 0.0) or 0.0)

    logger.info(
        "Validator: documents_parsed=%s parsing_confidence=%.3f",
        documents_parsed,
        parsing_confidence,
    )

    checks_passed = 0
    checks_total = 0

    income = financial_data.get("income_statement") or {}
    balance = financial_data.get("balance_sheet") or {}
    cash_flow = financial_data.get("cash_flow") or {}
    if not isinstance(income, dict):
        income = {}
    if not isinstance(balance, dict):
        balance = {}
    if not isinstance(cash_flow, dict):
        cash_flow = {}

    def _record(
        name: str,
        passed: bool,
        values_desc: str,
        fail_msg: Optional[str],
    ) -> None:
        nonlocal checks_total, checks_passed
        checks_total += 1
        if passed:
            checks_passed += 1
            logger.info("Validator PASS: %s | %s", name, values_desc)
        else:
            logger.warning("Validator FAIL: %s | %s", name, values_desc)
            if fail_msg:
                errors.append(fail_msg)

    # --- Sections (critical) ---
    for section in REQUIRED_SECTIONS:
        ok = _section_nonempty(financial_data, section)
        if ok:
            _record(
                f"section_{section}_nonempty",
                True,
                f"section={section!r} has data",
                None,
            )
        else:
            _record(
                f"section_{section}_nonempty",
                False,
                f"section={section!r} missing or empty",
                (
                    f"CRITICAL: financial_data section '{section}' is missing or empty — "
                    "downstream analysis is blocked."
                ),
            )

    # --- Critical fields: non-null and non-zero ---
    for logical, stmt_key, field_key in CRITICAL_NONZERO_FIELDS:
        stmt = financial_data.get(stmt_key) or {}
        if not isinstance(stmt, dict):
            stmt = {}
        val = stmt.get(field_key)
        v = _to_float(val)
        if v is None:
            _record(
                f"critical_field_{logical}",
                False,
                f"{stmt_key}.{field_key}=None or non-numeric",
                (
                    f"WARNING: critical field '{logical}' ({stmt_key}.{field_key}) "
                    "must be non-null and non-zero; value is missing or invalid."
                ),
            )
        elif v == 0.0:
            _record(
                f"critical_field_{logical}",
                False,
                f"{stmt_key}.{field_key}=0",
                (
                    f"WARNING: critical field '{logical}' ({stmt_key}.{field_key}) "
                    "must be non-zero; actual value is 0."
                ),
            )
        else:
            _record(
                f"critical_field_{logical}",
                True,
                f"{stmt_key}.{field_key}={v}",
                None,
            )

    # --- Mathematical cross-checks ---

    def _math_check(
        name: str,
        predicate: Callable[[], bool],
        pass_desc: str,
        fail_msg_fn: Callable[[], str],
        prereq: Callable[[], bool],
        skip_desc: str,
    ) -> None:
        if not prereq():
            checks_total += 1
            logger.warning(
                "Validator SKIP: %s | %s (counts as failed check)",
                name,
                skip_desc,
            )
            errors.append(
                f"DATA QUALITY: {name} could not be evaluated — {skip_desc}"
            )
            return
        passed = predicate()
        fail_msg = fail_msg_fn() if not passed else ""
        _record(name, passed, pass_desc if passed else pass_desc, fail_msg or None)

    rev = _to_float(income.get("revenues"))
    gp = _to_float(income.get("gross_profit"))
    ni = _to_float(income.get("net_income_loss"))
    oi = _to_float(income.get("operating_income_loss"))
    ta = _to_float(balance.get("assets"))
    ca = _to_float(balance.get("assets_current"))
    cash_eq = _to_float(balance.get("cash_and_cash_equivalents"))
    ltd = _to_float(balance.get("long_term_debt"))
    tl = _to_float(balance.get("total_liabilities"))
    cl = _to_float(balance.get("liabilities_current"))
    ocf = _to_float(cash_flow.get("net_cash_from_operating_activities"))
    capex = _to_float(cash_flow.get("capital_expenditures"))

    # Gross profit <= revenues
    if rev is not None and gp is not None:
        passed = gp <= rev
        _record(
            "gross_profit_lte_revenues",
            passed,
            f"gross_profit={gp}, revenues={rev}",
            None
            if passed
            else (
                f"DATA QUALITY: gross profit must be less than or equal to revenues "
                f"(gross_profit={gp}, revenues={rev})."
            ),
        )
    else:
        checks_total += 1
        logger.warning(
            "Validator SKIP: gross_profit_lte_revenues | missing revenues or gross_profit"
        )
        errors.append(
            "DATA QUALITY: gross_profit <= revenues could not be evaluated — "
            "revenues or gross_profit missing."
        )

    # Total assets > current assets
    if ta is not None and ca is not None:
        passed = ta > ca
        _record(
            "total_assets_gt_current_assets",
            passed,
            f"assets={ta}, assets_current={ca}",
            None
            if passed
            else (
                f"DATA QUALITY: total assets must be greater than current assets "
                f"(assets={ta}, assets_current={ca})."
            ),
        )
    else:
        checks_total += 1
        logger.warning(
            "Validator SKIP: total_assets_gt_current_assets | missing assets or assets_current"
        )
        errors.append(
            "DATA QUALITY: total assets > current assets could not be evaluated — "
            "assets or assets_current missing."
        )

    # Net income < revenues
    if rev is not None and ni is not None:
        passed = ni < rev
        _record(
            "net_income_lt_revenues",
            passed,
            f"net_income_loss={ni}, revenues={rev}",
            None
            if passed
            else (
                f"DATA QUALITY: net income must be less than revenues "
                f"(net_income_loss={ni}, revenues={rev})."
            ),
        )
    else:
        checks_total += 1
        logger.warning(
            "Validator SKIP: net_income_lt_revenues | missing revenues or net_income_loss"
        )
        errors.append(
            "DATA QUALITY: net income < revenues could not be evaluated — "
            "revenues or net_income_loss missing."
        )

    # Operating income <= gross profit
    if oi is not None and gp is not None:
        passed = oi <= gp
        _record(
            "operating_income_lte_gross_profit",
            passed,
            f"operating_income_loss={oi}, gross_profit={gp}",
            None
            if passed
            else (
                f"DATA QUALITY: operating income must be less than or equal to gross profit "
                f"(operating_income_loss={oi}, gross_profit={gp})."
            ),
        )
    else:
        checks_total += 1
        logger.warning(
            "Validator SKIP: operating_income_lte_gross_profit | "
            "missing operating_income_loss or gross_profit"
        )
        errors.append(
            "DATA QUALITY: operating income <= gross profit could not be evaluated — "
            "operating_income_loss or gross_profit missing."
        )

    # Cash and equivalents <= current assets
    if cash_eq is not None and ca is not None:
        passed = cash_eq <= ca
        _record(
            "cash_lte_current_assets",
            passed,
            f"cash_and_cash_equivalents={cash_eq}, assets_current={ca}",
            None
            if passed
            else (
                f"DATA QUALITY: cash and equivalents must be less than or equal to "
                f"current assets (cash_and_cash_equivalents={cash_eq}, assets_current={ca})."
            ),
        )
    else:
        checks_total += 1
        logger.warning(
            "Validator SKIP: cash_lte_current_assets | "
            "missing cash_and_cash_equivalents or assets_current"
        )
        errors.append(
            "DATA QUALITY: cash <= current assets could not be evaluated — "
            "cash_and_cash_equivalents or assets_current missing."
        )

    # Long term debt <= total liabilities
    if ltd is not None and tl is not None:
        passed = ltd <= tl
        _record(
            "long_term_debt_lte_total_liabilities",
            passed,
            f"long_term_debt={ltd}, total_liabilities={tl}",
            None
            if passed
            else (
                f"DATA QUALITY: long term debt must be less than or equal to total liabilities "
                f"(long_term_debt={ltd}, total_liabilities={tl})."
            ),
        )
    else:
        checks_total += 1
        logger.warning(
            "Validator SKIP: long_term_debt_lte_total_liabilities | "
            "missing long_term_debt or total_liabilities"
        )
        errors.append(
            "DATA QUALITY: long term debt <= total liabilities could not be evaluated — "
            "long_term_debt or total_liabilities missing."
        )

    # Current liabilities <= total liabilities
    if cl is not None and tl is not None:
        passed = cl <= tl
        _record(
            "current_liabilities_lte_total_liabilities",
            passed,
            f"liabilities_current={cl}, total_liabilities={tl}",
            None
            if passed
            else (
                f"DATA QUALITY: current liabilities must be less than or equal to "
                f"total liabilities (liabilities_current={cl}, total_liabilities={tl})."
            ),
        )
    else:
        checks_total += 1
        logger.warning(
            "Validator SKIP: current_liabilities_lte_total_liabilities | "
            "missing liabilities_current or total_liabilities"
        )
        errors.append(
            "DATA QUALITY: current liabilities <= total liabilities could not be evaluated — "
            "liabilities_current or total_liabilities missing."
        )

    # Capital expenditures < total assets
    if capex is not None and ta is not None:
        passed = capex < ta
        _record(
            "capex_lt_total_assets",
            passed,
            f"capital_expenditures={capex}, assets={ta}",
            None
            if passed
            else (
                f"DATA QUALITY: capital expenditures must be less than total assets "
                f"(capital_expenditures={capex}, assets={ta})."
            ),
        )
    else:
        checks_total += 1
        logger.warning(
            "Validator SKIP: capex_lt_total_assets | missing capital_expenditures or assets"
        )
        errors.append(
            "DATA QUALITY: capital expenditures < total assets could not be evaluated — "
            "capital_expenditures or assets missing."
        )

    # If net income > 0, operating cash flow must be > 0
    checks_total += 1
    if ni is not None and ocf is not None:
        if ni <= 0:
            checks_passed += 1
            logger.info(
                "Validator PASS: earnings_quality_ocf_when_ni_positive | "
                "net_income_loss=%s (not positive; rule N/A), ocf=%s",
                ni,
                ocf,
            )
        elif ocf > 0:
            checks_passed += 1
            logger.info(
                "Validator PASS: earnings_quality_ocf_when_ni_positive | "
                "net_income_loss=%s, net_cash_from_operating_activities=%s",
                ni,
                ocf,
            )
        else:
            logger.warning(
                "Validator FAIL: earnings_quality_ocf_when_ni_positive | "
                "net_income_loss=%s, net_cash_from_operating_activities=%s",
                ni,
                ocf,
            )
            msg = (
                f"DATA QUALITY: net income is positive ({ni}) but operating cash flow "
                f"is not positive (net_cash_from_operating_activities={ocf})."
            )
            errors.append(msg)
            if ocf < 0:
                errors.append(EARNINGS_QUALITY_CRITICAL)
                logger.error("%s (net_income_loss=%s, ocf=%s)", EARNINGS_QUALITY_CRITICAL, ni, ocf)
    else:
        logger.warning(
            "Validator SKIP: earnings_quality_ocf_when_ni_positive | "
            "missing net_income_loss or net_cash_from_operating_activities"
        )
        errors.append(
            "DATA QUALITY: earnings quality (NI vs OCF) could not be evaluated — "
            "net_income_loss or net_cash_from_operating_activities missing."
        )

    data_quality_score = checks_passed / checks_total if checks_total else 0.0
    data_quality_score = max(0.0, min(1.0, float(data_quality_score)))

    logger.info(
        "Validator: data_quality_score=%.3f (%d/%d checks passed).",
        data_quality_score,
        checks_passed,
        checks_total,
    )

    if data_quality_score < 0.7:
        documents_parsed = False
        crit = (
            "CRITICAL: data_quality_score is below 0.7 — downstream analysis is unreliable."
        )
        errors.append(crit)
        logger.error(crit)
    elif data_quality_score < 0.85:
        mod = (
            "MODERATE: data_quality_score is between 0.7 and 0.85 — "
            "some metrics may be unreliable."
        )
        errors.append(mod)
        logger.warning(mod)

    return {
        "errors": errors,
        "documents_parsed": documents_parsed,
        "data_quality_score": data_quality_score,
    }
