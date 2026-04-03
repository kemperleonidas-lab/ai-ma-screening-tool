"""Agent 5: final M&A target screening report generator."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from state import FinancialState

logger = logging.getLogger(__name__)

VERDICTS = frozenset({"PURSUE", "PASS", "MORE DILIGENCE REQUIRED"})

# Expected top-level keys from the LLM (exact JSON contract)
_REPORT_JSON_KEYS = (
    "company_snapshot",
    "investment_considerations",
    "risk_factors",
    "screening_verdict",
    "verdict_rationale",
    "benchmark_basis_line",
    "key_metrics_table",
)

_JSON_FENCE_PATTERN = re.compile(
    r"```\s*json\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_GENERIC_FENCE_PATTERN = re.compile(
    r"```\s*\n?(.*?)```",
    re.DOTALL,
)

def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric_value(metric_key: str, value: Any) -> str:
    """
    Canonical metric formatting for key_metrics_table (deterministic; not LLM-dependent).

    Conventions:
    - margins: "12.34%"
    - ROE: plain ratio "0.30" (NOT percent)
    - coverage: "7.89x"
    - USD amounts assumed to be in millions when coming from metrics/free_cash_flow: "$123,456M"
    """
    k = str(metric_key)

    if value is None:
        return ""

    # Monetary (millions) conventions used in this codebase
    if k in {"free_cash_flow", "revenues", "total_assets"}:
        n = _to_float(value)
        if n is None:
            return str(value)
        return f"${n:,.0f}M"

    n = _to_float(value)
    if n is None:
        return str(value)

    if k == "return_on_equity":
        return f"{n:.2f}"
    if k == "interest_coverage_ratio":
        return f"{n:.2f}x"
    if k in {"gross_margin", "operating_margin", "net_margin"} or "margin" in k:
        return f"{n:.2f}%"
    return f"{n:.2f}"


def _build_canonical_key_metrics_table(
    metrics: Dict[str, Any],
    financial_data: Dict[str, Any],
) -> Dict[str, str]:
    meta = (financial_data or {}).get("meta") if isinstance(financial_data, dict) else {}
    fiscal_year = None
    if isinstance(meta, dict):
        fiscal_year = meta.get("fiscal_year")

    # Deterministic, banker-usable set: ratios + absolute scale + fiscal-year context.
    canonical_order = [
        "fiscal_year",
        "revenues",
        "total_assets",
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

    out: Dict[str, str] = {}
    out["fiscal_year"] = "" if fiscal_year is None else str(fiscal_year)
    for k in canonical_order:
        if k == "fiscal_year":
            continue
        out[k] = _format_metric_value(k, metrics.get(k))
    return out


def _extract_json_from_response(content: str) -> str:
    """Pull JSON text from a ```json ... ``` fence, else first ``` ... ``` block, else stripped body."""

    if not content or not str(content).strip():
        raise ValueError("Empty LLM response")

    text = str(content)
    m = _JSON_FENCE_PATTERN.search(text)
    if m:
        return m.group(1).strip()

    m = _GENERIC_FENCE_PATTERN.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith("{") or candidate.startswith("["):
            return candidate

    return text.strip()


def _parse_json_loose(text: str) -> Dict[str, Any]:
    """Parse JSON with light cleanup for common LLM issues (trailing commas)."""

    t = text.strip()
    t = re.sub(r",\s*(?=[}\]])", "", t)
    try:
        data = json.loads(t)
    except json.JSONDecodeError as e1:
        logger.debug("json.loads failed on extracted block: %s", e1)
        raise ValueError(f"Invalid JSON: {e1}") from e1

    if not isinstance(data, dict):
        raise ValueError("Parsed JSON root must be an object")
    return data


def _parse_report_payload(content: str) -> Dict[str, Any]:
    """Extract fenced JSON and parse to dict."""

    extracted = _extract_json_from_response(content)
    return _parse_json_loose(extracted)


def _invalid_or_missing_fields(data: Dict[str, Any]) -> List[str]:
    """Return human-readable reasons for any missing or invalid required fields."""
    problems: List[str] = []
    for key in _REPORT_JSON_KEYS:
        if key not in data:
            problems.append(f"missing key: {key}")
            continue
        val = data[key]
        if key == "investment_considerations":
            if not isinstance(val, list):
                problems.append("investment_considerations (must be a list of 3 strings)")
            elif len(val) != 3:
                problems.append(
                    f"investment_considerations (expected 3 items, got {len(val)})"
                )
            else:
                for i, item in enumerate(val):
                    if not isinstance(item, str):
                        problems.append(
                            f"investment_considerations[{i}] (must be string, got {type(item).__name__})"
                        )
        elif key == "risk_factors":
            if not isinstance(val, list):
                problems.append("risk_factors (must be a list of 3 strings)")
            elif len(val) != 3:
                problems.append(f"risk_factors (expected 3 items, got {len(val)})")
            else:
                for i, item in enumerate(val):
                    if not isinstance(item, str):
                        problems.append(
                            f"risk_factors[{i}] (must be string, got {type(item).__name__})"
                        )
        elif key == "key_metrics_table":
            if not isinstance(val, dict):
                problems.append("key_metrics_table (must be a dict)")
            else:
                for mk, mv in val.items():
                    if not isinstance(mv, str):
                        problems.append(
                            f"key_metrics_table[{mk!r}] (value must be string, got {type(mv).__name__})"
                        )
        elif key == "company_snapshot":
            if not isinstance(val, str):
                problems.append(f"company_snapshot (must be string, got {type(val).__name__})")
        elif key == "verdict_rationale":
            if not isinstance(val, str):
                problems.append(f"verdict_rationale (must be string, got {type(val).__name__})")
        elif key == "benchmark_basis_line":
            if not isinstance(val, str):
                problems.append(f"benchmark_basis_line (must be string, got {type(val).__name__})")
        elif key == "screening_verdict":
            if not isinstance(val, str):
                problems.append(f"screening_verdict (must be string, got {type(val).__name__})")
            elif str(val).upper().strip() not in VERDICTS:
                problems.append(
                    f"screening_verdict (invalid value: {val!r}, expected one of {sorted(VERDICTS)})"
                )
    return problems


def _ensure_three(items: Any, placeholders: List[str]) -> List[str]:
    out: List[str] = []
    if isinstance(items, list):
        for x in items:
            if len(out) >= 3:
                break
            s = str(x).strip()
            if s:
                out.append(s)
    while len(out) < 3:
        out.append(placeholders[len(out)])
    return out[:3]


def _derive_risk_factors(red_flags: Any, industry: str) -> List[str]:
    """
    Deterministic risk factors for the report.

    Policy:
    - If red_flags has items: use up to the first 3, cleaned.
    - If red_flags is empty: use non-metric-specific diligence risks only (no claims like
      \"weak margin vs peers\" unless present in red_flags).
    """
    out: List[str] = []
    if isinstance(red_flags, list):
        for x in red_flags:
            if len(out) >= 3:
                break
            s = str(x).strip()
            if not s:
                continue
            # Strip scoring suffix if present (keeps bullet professional).
            s = re.sub(r"\s*\(score contribution:\s*[\d.]+\)\s*$", "", s, flags=re.IGNORECASE)
            out.append(s)
    if out:
        return _ensure_three(out, ["—", "—", "—"])

    ind = str(industry or "").strip()
    # Three diligence risks that are always relevant in M&A and do not depend on metric comparisons.
    return [
        f"Integration execution risk: validate operating model fit, change management, and synergy capture assumptions for {ind or 'the sector'}.",
        "Commercial diligence risk: confirm customer concentration, retention/cohort behavior (if applicable), and competitive differentiation under realistic go-to-market scenarios.",
        "Transaction diligence risk: validate legal/regulatory constraints, key contract terms, and technology/operational dependencies that could affect closing certainty or integration.",
    ][:3]


_METRIC_KEYWORDS: Dict[str, List[str]] = {
    "gross_margin": ["gross margin"],
    "operating_margin": ["operating margin"],
    "net_margin": ["net margin"],
    "current_ratio": ["current ratio"],
    "debt_to_equity": ["debt-to-equity", "debt to equity"],
    "return_on_equity": ["return on equity", "roe"],
    "free_cash_flow": ["free cash flow", "fcf"],
    "cash_conversion_ratio": ["cash conversion ratio"],
    "interest_coverage_ratio": ["interest coverage", "interest coverage ratio"],
}


def _extract_score_contribution(text: str) -> Optional[float]:
    m = re.search(r"score contribution:\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_metric_key_from_red_flag(text: str) -> str | None:
    t = (text or "").lower()
    # Benchmark-derived red flags: "Benchmark weakness on <metric_name>: ..."
    m = re.search(r"benchmark weakness on\s+([a-z0-9_]+)\s*:", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    if "operating margin" in t:
        return "operating_margin"
    if "gross margin" in t:
        return "gross_margin"
    if "net margin" in t:
        return "net_margin"
    if "current ratio" in t:
        return "current_ratio"
    if "debt-to-equity" in t or "debt to equity" in t:
        return "debt_to_equity"
    if "return on equity" in t or re.search(r"\broe\b", t):
        return "return_on_equity"
    if "cash conversion ratio" in t:
        return "cash_conversion_ratio"
    if "free cash flow" in t or re.search(r"\bfcf\b", t):
        return "free_cash_flow"
    if "interest coverage" in t:
        return "interest_coverage_ratio"
    return None


def _build_risk_input(
    *,
    state: FinancialState,
    red_flags: List[str],
    benchmark_analysis: Dict[str, Any],
    industry_benchmark: Dict[str, Any],
    peer_comparison: Any,
) -> Dict[str, Any]:
    """
    Build pre-identified risk observations so the LLM acts as a formatter.
    """
    metrics = state.get("metrics") or {}
    financial_data = state.get("financial_data") or {}
    market_data = (financial_data or {}).get("market_data") or {}
    size_tier = None
    if isinstance(market_data, dict):
        size_tier = market_data.get("size_tier")

    # Structured red flags: include metric key, value (from `metrics`), and score contribution.
    structured_red_flags: List[Dict[str, Any]] = []
    for rf in red_flags:
        text = str(rf)
        metric_key = _extract_metric_key_from_red_flag(text)
        value = None
        if metric_key and isinstance(metrics, dict):
            value = metrics.get(metric_key)
        structured_red_flags.append(
            {
                "text": text,
                "metric": metric_key,
                "value": value,
                "score_contribution": _extract_score_contribution(text),
            }
        )

    # Benchmark weaknesses: metric + gap size, derived from benchmark_analysis / industry_benchmark.
    #
    # Strict rule: when `red_flags` is empty, we do not include benchmark weaknesses at all.
    # This prevents metric-based risk factors from being generated when there are no
    # pre-identified observations to rephrase.
    benchmark_weaknesses: List[Dict[str, Any]] = []
    if structured_red_flags:
        if isinstance(benchmark_analysis, dict):
            for metric, details in benchmark_analysis.items():
                if not isinstance(details, dict):
                    continue
                assessment = str(details.get("assessment") or "").upper().strip()
                if assessment != "WEAK":
                    continue
                tv = details.get("target_value")
                pa = details.get("peer_average")
                if tv is None or pa is None:
                    continue
                try:
                    gap = abs(float(tv) - float(pa))
                except (TypeError, ValueError):
                    continue
                benchmark_weaknesses.append(
                    {
                        "basis": "peer",
                        "metric": str(metric),
                        "gap_size": gap,
                        "target_value": tv,
                        "peer_average": pa,
                    }
                )

        if isinstance(industry_benchmark, dict):
            for metric, details in industry_benchmark.items():
                if not isinstance(details, dict):
                    continue
                assessment = str(details.get("assessment") or "").upper().strip()
                if assessment != "BELOW":
                    continue
                tv = details.get("target_value")
                ia = details.get("industry_avg")
                if tv is None or ia is None:
                    continue
                try:
                    gap = abs(float(tv) - float(ia))
                except (TypeError, ValueError):
                    continue
                benchmark_weaknesses.append(
                    {
                        "basis": "industry",
                        "metric": str(metric),
                        "gap_size": gap,
                        "target_value": tv,
                        "industry_avg": ia,
                    }
                )

    peer_count = len(peer_comparison) if isinstance(peer_comparison, list) else 0
    is_thin = peer_count < 5

    return {
        "red_flags": structured_red_flags,
        "benchmark_weaknesses": benchmark_weaknesses,
        "peer_info": {"peer_count": peer_count, "is_thin": is_thin},
        "company_context": {"sector": str(state.get("industry") or ""), "size_tier": size_tier},
    }


def _enforce_risk_factors_grounded(
    risk_factors: List[str],
    risk_input: Dict[str, Any],
    *,
    industry: str,
) -> List[str]:
    """
    Guardrail: if LLM mentions metric-level concerns that are not present in
    `risk_input`, replace them with structural M&A risks.
    """
    allowed_metric_keys: set[str] = set()
    for rf in risk_input.get("red_flags", []) or []:
        if isinstance(rf, dict) and rf.get("metric"):
            allowed_metric_keys.add(str(rf["metric"]))

    def _risk_has_metric_keyword(t: str) -> bool:
        low = t.lower()
        for phrases in _METRIC_KEYWORDS.values():
            for ph in phrases:
                if ph in low:
                    return True
        return False

    def _risk_mentions_allowed_metric(t: str) -> bool:
        low = t.lower()
        for mk in allowed_metric_keys:
            for ph in _METRIC_KEYWORDS.get(mk, []):
                if ph in low:
                    return True
        return False

    structural = [
        f"Integration complexity risk: validate operating model fit, change management, and synergy execution for deals in the {industry} sector.",
        "Regulatory/closing risk: confirm antitrust/regulatory requirements and material contract conditions that could affect timing and certainty of closing.",
        "Market concentration risk: assess competitive dynamics and customer/vendor dependency that could pressure growth post-close.",
    ]

    out: List[str] = []
    for rf in risk_factors:
        s = str(rf).strip()
        if not s:
            continue
        if allowed_metric_keys and _risk_mentions_allowed_metric(s):
            out.append(s)
            continue
        if (not allowed_metric_keys) and _risk_has_metric_keyword(s):
            # No metric-level observations were provided; reject metric-level phrasing.
            out.append(structural[len(out) % len(structural)])
            continue
        # If metric keywords appear but don't map to allowed metrics, replace.
        if _risk_has_metric_keyword(s) and not _risk_mentions_allowed_metric(s):
            out.append(structural[len(out) % len(structural)])
            continue
        out.append(s)

    return _ensure_three(out, ["—", "—", "—"])


def _fallback_report_from_state(
    company_name: str,
    ticker: str,
    industry: str,
    metrics: Dict[str, Any],
    red_flags: List[str],
    risk_score: float,
    benchmark_basis_line: str,
    confidence_score: float,
) -> Dict[str, Any]:
    """Construct report_json when LLM parsing fails entirely."""

    km = {k: str(v) if v is not None else "" for k, v in metrics.items()}
    rf_source = [str(x) for x in red_flags if str(x).strip()]
    risks = _ensure_three(
        rf_source,
        [
            "Automated risk signals could not be fully enumerated.",
            "Further diligence is required on financial and operational risks.",
            "Integration and market risks should be validated in confirmatory work.",
        ],
    )
    snap = (
        f"{company_name} ({ticker}) operates in the {industry} space. "
        "Narrative generation was unavailable; use computed metrics and source filings for decisions."
    )
    return {
        "company_snapshot": snap,
        "key_metrics_table": km,
        "investment_considerations": [
            f"Review strategic fit and synergies for {company_name} using the metrics table and benchmarks.",
            "Assess peer-relative positioning and growth assumptions with additional commercial diligence.",
            "Validate valuation and integration assumptions before proceeding.",
        ],
        "risk_factors": risks,
        "screening_verdict": "MORE DILIGENCE REQUIRED",
        "verdict_rationale": (
            "Automated screening could not produce a full LLM report. "
            f"Financial Health score is {round(10.0 - risk_score, 1):.1f}/10; "
            "use red-flag items and metrics as the basis for next steps."
        ),
        "benchmark_basis_line": benchmark_basis_line,
        "confidence_score": confidence_score,
    }


def _compute_confidence_from_state(
    *,
    state: FinancialState,
    financial_data: Dict[str, Any],
    peer_comparison: Any,
) -> float:
    """
    Deterministic confidence score computed in code (not LLM).

    confidence = 0.4*data_completeness + 0.4*peer_coverage_ratio + 0.2*data_quality_score
    """

    expected_fields: List[Tuple[str, str]] = [
        ("income_statement", "revenues"),
        ("income_statement", "gross_profit"),
        ("income_statement", "operating_income_loss"),
        ("income_statement", "net_income_loss"),
        ("balance_sheet", "assets"),
        ("balance_sheet", "assets_current"),
        ("balance_sheet", "liabilities_current"),
        ("balance_sheet", "total_liabilities"),
        ("balance_sheet", "stockholders_equity"),
        ("cash_flow", "net_cash_from_operating_activities"),
        ("cash_flow", "capital_expenditures"),
        ("income_statement", "interest_expense"),
    ]

    def _safe_get(section: str, key: str) -> Any:
        sec = (financial_data or {}).get(section)
        if not isinstance(sec, dict):
            return None
        return sec.get(key)

    non_null = 0
    total = len(expected_fields)
    for section, key in expected_fields:
        v = _safe_get(section, key)
        if v is not None:
            non_null += 1

    data_completeness = (non_null / total) if total else 0.0

    peer_count = len(peer_comparison) if isinstance(peer_comparison, list) else 0
    peer_coverage_ratio = min(1.0, peer_count / 6.0)

    data_quality_score = state.get("data_quality_score", 0.0)  # type: ignore[assignment]
    try:
        dq = float(data_quality_score or 0.0)
    except (TypeError, ValueError):
        dq = 0.0
    dq = max(0.0, min(1.0, dq))

    confidence = (0.4 * data_completeness) + (0.4 * peer_coverage_ratio) + (0.2 * dq)
    confidence = max(0.0, min(1.0, confidence))
    return float(confidence)


def _format_text_report(company_name: str, ticker: str, industry: str, data: Dict[str, Any], risk_score: float) -> str:
    """Format parsed JSON report into readable text."""
    lines: List[str] = []
    lines.append(f"M&A Target Screening Report: {company_name} ({ticker})")
    lines.append(f"Industry: {industry}")
    health_score = round(10.0 - float(risk_score or 0.0), 1)
    lines.append(f"Financial Health Score: {health_score:.1f}/10")
    lines.append("")

    lines.append("Company Snapshot")
    lines.append(str(data.get("company_snapshot", "")))
    lines.append("")

    lines.append("Key Metrics")
    metrics_table = data.get("key_metrics_table", {}) or {}
    if isinstance(metrics_table, dict):
        for key, value in metrics_table.items():
            lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("Investment Considerations")
    for item in data.get("investment_considerations", []) or []:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("Risk Factors")
    for item in data.get("risk_factors", []) or []:
        lines.append(f"- {item}")
    lines.append("")

    lines.append(f"Screening Verdict: {data.get('screening_verdict', '')}")
    lines.append(str(data.get("verdict_rationale", "")))
    lines.append(str(data.get("benchmark_basis_line", "")))
    lines.append("")
    lines.append(f"Confidence Score: {data.get('confidence_score', 0.0)}")

    return "\n".join(lines).strip()


def _invoke_llm(llm: ChatOpenAI, system: str, user: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content if hasattr(response, "content") else str(response)


def run_reporter_agent(state: FinancialState) -> Dict[str, Any]:
    """Generate final report JSON + text from all prior analysis outputs."""
    errors: List[str] = list(state.get("errors", []))  # type: ignore[arg-type]

    company_name = str(state.get("company_name", "Unknown Company"))
    industry = str(state.get("industry", "Unknown Industry"))
    ticker = str(state.get("ticker", "N/A"))
    metrics = state.get("metrics", {}) or {}
    financial_data = state.get("financial_data", {}) or {}
    benchmark_analysis = state.get("benchmark_analysis", {}) or {}
    industry_benchmark = state.get("industry_benchmark", {}) or {}
    peer_comparison = state.get("peer_comparison", []) or []
    benchmark_peer_retrieval_mode = str(state.get("benchmark_peer_retrieval_mode", "") or "sector_only")
    benchmark_size_tier = str(state.get("benchmark_size_tier", "") or "").strip()
    red_flags = state.get("red_flags", []) or []
    risk_score = float(state.get("risk_score", 0.0) or 0.0)
    peer_count = len(peer_comparison) if isinstance(peer_comparison, list) else 0

    # Confidence is computed deterministically in code (pre-LLM) and will override any LLM attempts.
    precomputed_confidence = _compute_confidence_from_state(
        state=state,
        financial_data=financial_data if isinstance(financial_data, dict) else {},
        peer_comparison=peer_comparison,
    )

    if benchmark_peer_retrieval_mode == "size_matched" and benchmark_size_tier:
        benchmark_basis_line_default = (
            f"Benchmarked against {peer_count} size-matched peers ({benchmark_size_tier}, {industry})."
        )
    elif benchmark_peer_retrieval_mode == "sector_fallback":
        benchmark_basis_line_default = (
            f"Benchmarked against {peer_count} sector peers (size cohort too thin for matched comparison)."
        )
    else:
        benchmark_basis_line_default = f"Benchmarked against {peer_count} sector peers."

    payload = {
        "company_name": company_name,
        "industry": industry,
        "ticker": ticker,
        "metrics": metrics,
        "financial_data_meta": (financial_data or {}).get("meta") if isinstance(financial_data, dict) else {},
        "confidence_score_precomputed": precomputed_confidence,
        "benchmark_analysis": benchmark_analysis,
        "industry_benchmark": industry_benchmark,
        "peer_comparison": peer_comparison,
        "benchmark_peer_retrieval_mode": benchmark_peer_retrieval_mode,
        "benchmark_size_tier": benchmark_size_tier,
        "benchmark_basis_line_default": benchmark_basis_line_default,
        "red_flags": red_flags,
        "risk_score": risk_score,
    }

    risk_input = _build_risk_input(
        state=state,
        red_flags=red_flags if isinstance(red_flags, list) else [],
        benchmark_analysis=benchmark_analysis if isinstance(benchmark_analysis, dict) else {},
        industry_benchmark=industry_benchmark if isinstance(industry_benchmark, dict) else {},
        peer_comparison=peer_comparison,
    )
    payload["risk_input"] = risk_input

    canonical_key_metrics_table = _build_canonical_key_metrics_table(
        metrics if isinstance(metrics, dict) else {},
        financial_data if isinstance(financial_data, dict) else {},
    )

    system_prompt_full = """You are an M&A target screening analyst.

Return your response as valid JSON wrapped in a markdown code block using ```json and ``` tags (opening fence ```json on its own line, closing fence ``` on its own line). Do not put any text outside the code block except optional brief whitespace.

The JSON object must use exactly these keys:

{
  "company_snapshot": "<string: one sentence on what the company does and its market position>",
  "investment_considerations": ["<string>", "<string>", "<string>"],
  "risk_factors": ["<string>", "<string>", "<string>"],
  "screening_verdict": "PURSUE" | "PASS" | "MORE DILIGENCE REQUIRED",
  "verdict_rationale": "<string: exactly two sentences explaining the verdict>",
  "benchmark_basis_line": "<string: exactly one sentence stating peer count and whether benchmark used size-matched cohort or full-sector fallback>",
  "key_metrics_table": { "<snake_case_metric_name>": "<formatted value string>", ... }
}

Rules:
- investment_considerations: exactly 3 strings; frame positives as M&A opportunities (strategy, synergies, deal thesis), not generic praise.
- risk_factors: exactly 3 strings; each with clear M&A implications (execution, regulatory, integration, valuation, etc.).
- Convert the following pre-identified risk observations into professional M&A risk factor language.
- Do NOT identify additional financial risks beyond what is listed in `risk_input`.
- red_flags is empty — do not reference any financial metrics as risks under any circumstances.
- If the pre-identified observation list has fewer than 3 items, fill remaining slots with structural M&A risks appropriate for a company of this size and sector (integration complexity, regulatory, market concentration) — never invent metric-level financial concerns.
- The payload includes `confidence_score_precomputed`. Do NOT output `confidence_score` in your JSON response; code will inject it.
- key_metrics_table: every value MUST be a formatted string (not a raw number). Example: margins like "40.50%", free cash flow like "$105,539M", ratios with two decimals, return_on_equity like "0.30" (plain ratio, do not convert to percent).
- Use only the keys above; values must satisfy the types described.
- verdict_rationale: Write the rationale in plain professional business language as if writing to a senior banker. Never mention risk scores, overlap counts, thresholds, or any internal system mechanics. Focus only on the company's financial characteristics and what they mean for an acquirer.
- benchmark_basis_line: must be one sentence and explicitly state how many peers were used plus whether they were size-matched or full-sector fallback.

Screening verdict decision rules (compute using the provided JSON data):
- Inputs you may use:
  - `risk_score` (number)
  - `benchmark_analysis` (peers): metric -> { peer_average, target_value, assessment } where `assessment` is "WEAK", "AVERAGE", or "STRONG" (or null)
  - `industry_benchmark` (industry averages): metric -> { industry_avg, target_value, assessment } where `assessment` is "BELOW", "IN LINE", or "ABOVE" (or null)
  - `red_flags` (list of strings)
- Define `weak_overlap_count` as the number of metrics where BOTH are true:
  - peer `assessment` is "WEAK"
  - industry `assessment` is "BELOW"
- Cohort-relative assessment:
  - If `benchmark_peer_retrieval_mode` is "size_matched", evaluate peer performance within that size cohort (same `size_tier`).
  - Do not penalize companies in smaller tiers (e.g. small_cap or micro_cap) for trailing much larger peers when size-matched benchmarking is used; assessment is relative to the cohort.
- Define `critical_red_flag_count` as the number of strings in `red_flags` that mention any of: "Debt-to-equity", "Free cash flow", "Operating margin", "Current ratio", "Return on equity", "Cash conversion ratio".
- Define `conflicting_signals` if there exists at least one metric where:
  - peer `assessment` is "WEAK" but industry `assessment` is not "BELOW" (null, "IN LINE", or "ABOVE"), OR
  - industry `assessment` is "BELOW" but peer `assessment` is not "WEAK" (null, "AVERAGE", or "STRONG").

Verdict selection (must output one of: PURSUE | PASS | MORE DILIGENCE REQUIRED):
- If `risk_score` is exactly 0 AND `weak_overlap_count` is fewer than 2, verdict MUST be "PURSUE" (not "MORE DILIGENCE REQUIRED").
- "PASS" must be selected if either:
  - `risk_score` exceeds 6, OR
  - `critical_red_flag_count` is at least 2.
- "MORE DILIGENCE REQUIRED" must be selected only if either:
  - `risk_score` is between 2 and 5 inclusive, OR
  - `conflicting_signals` is true.
- Otherwise select "PURSUE"."""

    user_prompt = (
        "Using the following analysis data, generate a structured M&A target screening report JSON.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )

    system_prompt_retry = """You are an M&A target screening analyst.

Return your response as valid JSON wrapped in a markdown code block using ```json and ``` tags only.

Return a JSON object with exactly these keys (no key_metrics_table):

{
  "company_snapshot": "<one sentence string>",
  "investment_considerations": ["<string>", "<string>", "<string>"],
  "risk_factors": ["<string>", "<string>", "<string>"],
  "screening_verdict": "PURSUE" | "PASS" | "MORE DILIGENCE REQUIRED",
  "verdict_rationale": "<two sentences>",
  "benchmark_basis_line": "<one sentence: peer count + size-matched or full-sector context>",
}

Keep strings concise and valid JSON (escape quotes inside strings).
Do NOT include `confidence_score` in your JSON response; confidence will be injected by code from `confidence_score_precomputed`.
Write the rationale in plain professional business language as if writing to a senior banker. Never mention risk scores, overlap counts, thresholds, or any internal system mechanics. Focus only on the company's financial characteristics and what they mean for an acquirer.
Always include benchmark_basis_line and explicitly state peer count plus whether benchmarking was size-matched or full-sector fallback.

Risk factor generation rules (same as full prompt):
- Convert the following pre-identified risk observations into professional M&A risk factor language.
- Do NOT identify additional financial risks beyond what is listed in `risk_input`.
- red_flags is empty — do not reference any financial metrics as risks under any circumstances.
- If the pre-identified observation list has fewer than 3 items, fill remaining slots with structural M&A risks appropriate for a company of this size and sector (integration complexity, regulatory, market concentration) — never invent metric-level financial concerns.

Apply the same screening verdict decision rules as in the full-schema prompt (risk_score + weak_overlap_count + critical_red_flag_count + conflicting_signals).
If benchmark_peer_retrieval_mode is size_matched, make verdict judgments relative to that size cohort; do not penalize smaller-tier targets for trailing peers outside their tier."""

    user_prompt_retry = (
        "Summarize the following analysis into the JSON object described in your instructions.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    report_data: Dict[str, Any] = {}
    content = ""

    # --- First attempt: full schema ---
    try:
        content = _invoke_llm(llm, system_prompt_full, user_prompt)
        logger.info("Reporter raw LLM response (attempt 1, before parse):\n%s", content)
        report_data = _parse_report_payload(content)
        logger.info("Reporter: JSON parsed successfully on first attempt.")
    except Exception as exc:
        logger.warning("Reporter: first JSON parse failed: %s", exc)
        errors.append(f"Reporter primary JSON parse failed: {exc}")

        # --- Second attempt: essential fields only ---
        try:
            content = _invoke_llm(llm, system_prompt_retry, user_prompt_retry)
            logger.info("Reporter raw LLM response (attempt 2, before parse):\n%s", content)
            partial = _parse_report_payload(content)
            report_data = dict(partial)
            report_data["key_metrics_table"] = dict(canonical_key_metrics_table)
            logger.info("Reporter: JSON parsed on retry; key_metrics_table filled from state metrics.")
        except Exception as exc2:
            logger.exception("Reporter: retry JSON parse failed: %s", exc2)
            errors.append(f"Reporter retry JSON parse failed: {exc2}")
            report_data = _fallback_report_from_state(
                company_name=company_name,
                ticker=ticker,
                industry=industry,
                metrics=metrics if isinstance(metrics, dict) else {},
                red_flags=list(red_flags) if isinstance(red_flags, list) else [],
                risk_score=risk_score,
                benchmark_basis_line=benchmark_basis_line_default,
                confidence_score=precomputed_confidence,
            )
            errors.append(
                "Reporter: using fallback report built from state (metrics + red_flags); "
                "LLM output could not be parsed."
            )

    field_issues = _invalid_or_missing_fields(report_data)
    if field_issues:
        logger.warning(
            "Reporter JSON missing or invalid fields (using fallbacks for these): %s",
            "; ".join(field_issues),
        )

    if "company_snapshot" not in report_data:
        report_data["company_snapshot"] = ""
    elif not isinstance(report_data["company_snapshot"], str):
        report_data["company_snapshot"] = str(report_data["company_snapshot"])

    if "verdict_rationale" not in report_data:
        report_data["verdict_rationale"] = ""
    elif not isinstance(report_data["verdict_rationale"], str):
        report_data["verdict_rationale"] = str(report_data["verdict_rationale"])

    if "benchmark_basis_line" not in report_data:
        report_data["benchmark_basis_line"] = benchmark_basis_line_default
    elif not isinstance(report_data["benchmark_basis_line"], str):
        report_data["benchmark_basis_line"] = str(report_data["benchmark_basis_line"])

    if "key_metrics_table" not in report_data or not isinstance(
        report_data.get("key_metrics_table"), dict
    ):
        report_data["key_metrics_table"] = dict(canonical_key_metrics_table)

    # Enforce deterministic metric formatting (LLM cannot regress formatting).
    report_data["key_metrics_table"] = dict(canonical_key_metrics_table)

    km = report_data.get("key_metrics_table", {})
    if isinstance(km, dict):
        coerced: Dict[str, str] = {}
        for k, v in km.items():
            if not isinstance(v, str):
                logger.warning(
                    "key_metrics_table[%r]: coercing value to string (was %s)",
                    k,
                    type(v).__name__,
                )
            coerced[str(k)] = "" if v is None else str(v)
        report_data["key_metrics_table"] = coerced

    inv = _ensure_three(
        report_data.get("investment_considerations"),
        [
            "Strategic fit and synergy case require further validation.",
            "Peer-relative performance supports additional confirmatory diligence.",
            "Market position may offer option value pending deeper commercial review.",
        ],
    )
    risks = _ensure_three(
        report_data.get("risk_factors"),
        [
            "Key risks could not be fully enumerated from available data.",
            "Integration and execution risks remain to be assessed.",
            "Regulatory and valuation risks require targeted diligence.",
        ],
    )
    # Guardrail: ensure metric-level phrasing only appears when it is present in risk_input.
    if isinstance(risk_input, dict):
        risks = _enforce_risk_factors_grounded(risks, risk_input, industry=industry)

    verdict = str(report_data.get("screening_verdict", "MORE DILIGENCE REQUIRED")).upper().strip()
    if verdict not in VERDICTS:
        verdict = "MORE DILIGENCE REQUIRED"

    conf = precomputed_confidence

    parsed_json: Dict[str, Any] = {
        "company_snapshot": str(report_data.get("company_snapshot", "")),
        "key_metrics_table": report_data.get("key_metrics_table", {}) or {},
        "investment_considerations": inv,
        "risk_factors": risks,
        "screening_verdict": verdict,
        "verdict_rationale": str(report_data.get("verdict_rationale", "")),
        "benchmark_basis_line": str(report_data.get("benchmark_basis_line", benchmark_basis_line_default)),
        "confidence_score": conf,
    }
    parsed_json["confidence_score"] = max(0.0, min(1.0, parsed_json["confidence_score"]))

    # Enrich report_json with context needed for PDF header (deterministic; no LLM dependency).
    try:
        parsed_json["industry"] = str(state.get("industry") or "")
    except Exception:
        parsed_json["industry"] = ""
    try:
        md = (financial_data or {}).get("market_data") if isinstance(financial_data, dict) else {}
        parsed_json["size_tier"] = str((md or {}).get("size_tier") or "")
    except Exception:
        parsed_json["size_tier"] = ""

    final_report = _format_text_report(company_name, ticker, industry, parsed_json, risk_score)
    logger.info("Reporter returning report_json keys: %s", list(parsed_json.keys()))

    return {
        "final_report": final_report,
        "confidence": parsed_json["confidence_score"],
        "report_json": parsed_json,
        "company_snapshot": parsed_json.get("company_snapshot", ""),
        "investment_considerations": parsed_json.get("investment_considerations", []),
        "risk_factors": parsed_json.get("risk_factors", []),
        "screening_verdict": parsed_json.get("screening_verdict", "MORE DILIGENCE REQUIRED"),
        "errors": errors,
    }
