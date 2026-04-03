"""Entry point for running the end-to-end financial analysis pipeline over a given filing."""

from __future__ import annotations

import json
import pprint
import subprocess
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from state import FinancialState, get_initial_state
from agents.parser import run_parser_agent as parser_agent
from agents.validator import run_validator_agent as validator_agent
from agents.metrics import run_metrics_agent as metrics_agent
from agents.benchmarker import run_benchmarker_agent as benchmarker_agent
from agents.red_flags import run_red_flags_agent as red_flags_agent
from agents.reporter import run_reporter_agent as reporter_agent
from utils.constants import CANONICAL_SECTORS
from utils.pdf_generator import generate_pdf_report


def build_graph():
    """Build a LangGraph with parser, metrics, benchmarker, red-flags, and reporter agents."""

    builder = StateGraph(FinancialState)
    builder.add_node("parser_agent", parser_agent)
    builder.add_node("validator_agent", validator_agent)
    builder.add_node("metrics_agent", metrics_agent)
    builder.add_node("benchmarker_agent", benchmarker_agent)
    builder.add_node("red_flags_agent", red_flags_agent)
    builder.add_node("reporter_agent", reporter_agent)
    builder.set_entry_point("parser_agent")
    builder.add_edge("parser_agent", "validator_agent")
    builder.add_edge("validator_agent", "metrics_agent")
    builder.add_edge("metrics_agent", "benchmarker_agent")
    builder.add_edge("benchmarker_agent", "red_flags_agent")
    builder.add_edge("red_flags_agent", "reporter_agent")
    builder.add_edge("reporter_agent", END)
    return builder.compile()


APP = build_graph()


def build_post_parser_graph():
    """Build a LangGraph that starts after parser confirmation (validator -> reporter)."""

    builder = StateGraph(FinancialState)
    builder.add_node("validator_agent", validator_agent)
    builder.add_node("metrics_agent", metrics_agent)
    builder.add_node("benchmarker_agent", benchmarker_agent)
    builder.add_node("red_flags_agent", red_flags_agent)
    builder.add_node("reporter_agent", reporter_agent)
    builder.set_entry_point("validator_agent")
    builder.add_edge("validator_agent", "metrics_agent")
    builder.add_edge("metrics_agent", "benchmarker_agent")
    builder.add_edge("benchmarker_agent", "red_flags_agent")
    builder.add_edge("red_flags_agent", "reporter_agent")
    builder.add_edge("reporter_agent", END)
    return builder.compile()


POST_PARSER_APP = build_post_parser_graph()

RAG_DIR = Path("rag")
CLASSIFICATION_SCHEDULE_PATH = RAG_DIR / "classification_schedule.json"


def _write_classification_schedule(next_rebuild_due: date) -> None:
    CLASSIFICATION_SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"next_rebuild_due": next_rebuild_due.isoformat()}
    with CLASSIFICATION_SCHEDULE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _maybe_run_scheduled_rebuild() -> None:
    """Run periodic peer DB refresh if the schedule is due."""

    today = date.today()
    next_due_default = today + timedelta(days=90)

    if not CLASSIFICATION_SCHEDULE_PATH.exists():
        _write_classification_schedule(next_due_default)
        return

    try:
        with CLASSIFICATION_SCHEDULE_PATH.open("r", encoding="utf-8") as f:
            schedule = json.load(f)
        next_due_raw = (schedule or {}).get("next_rebuild_due", "")
        next_due = date.fromisoformat(str(next_due_raw))
    except Exception:
        next_due = today

    if next_due <= today:
        print("Scheduled peer database refresh running — this may take a few minutes.")
        subprocess.run([sys.executable, str(RAG_DIR / "build_database.py")], check=True)
        subprocess.run([sys.executable, str(RAG_DIR / "build_damodaran.py")], check=True)
        _write_classification_schedule(today + timedelta(days=90))


def run_analysis(
    industry: str,
    ticker: str,
    company_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize state, run the graph, and print results.

    ``company_name`` is optional. If omitted or blank, the parser (Agent 1) sets it from the
    SEC EDGAR company facts API ``entityName`` after fetching XBRL data.

    Raises
    ------
    ValueError
        If ``ticker`` is empty or whitespace-only.
    """
    _maybe_run_scheduled_rebuild()

    ticker_clean = str(ticker).strip()
    if not ticker_clean:
        raise ValueError("ticker must not be empty")

    cn = (company_name or "").strip()

    state: FinancialState = get_initial_state(
        industry=industry,
        ticker=ticker_clean,
        company_name=cn,
    )

    # 1) Run parser first so we can confirm auto-detected sector with user.
    parser_update = parser_agent(state)
    state.update(parser_update)

    detected_company = str(state.get("company_name", cn) or cn or "Unknown Company")
    detected_sic_code = str(state.get("sic_code", "") or "N/A")
    detected_sic_desc = str(state.get("sic_description", "") or "N/A")
    proposed_sector = str(state.get("industry", "") or industry or "Industrials")

    print(
        f"Detected: {detected_company} | SIC {detected_sic_code} ({detected_sic_desc}) | "
        f"Proposed sector: {proposed_sector}."
    )

    # 2) Confirm or override sector with canonical validation.
    confirmed_sector = proposed_sector
    while True:
        user_sector = input("Confirm sector or enter a different one (press Enter to accept): ").strip()
        if not user_sector:
            break
        if user_sector in CANONICAL_SECTORS:
            confirmed_sector = user_sector
            break
        print("Invalid sector. Choose one of:")
        for s in CANONICAL_SECTORS:
            print(f"- {s}")

    state["industry"] = confirmed_sector

    # 3) Continue remaining pipeline (validator through reporter).
    result = POST_PARSER_APP.invoke(state)
    report_json = result.get("report_json", {}) or {}
    if isinstance(report_json, dict) and report_json:
        pdf_ticker = str(result.get("ticker", ticker_clean))
        from datetime import date

        target_pdf_path = f"outputs/{pdf_ticker}_{date.today().isoformat()}_report.pdf"
        print(f"Attempting to save PDF to: {target_pdf_path}")
        try:
            pdf_path = generate_pdf_report(
                report_json=report_json,
                company_name=str(result.get("company_name", cn)),
                ticker=pdf_ticker,
                risk_score=float(result.get("risk_score", 0.0) or 0.0),
                peer_comparison=list(result.get("peer_comparison", []) or []),
                benchmark_analysis=dict(result.get("benchmark_analysis", {}) or {}),
                industry_benchmark=dict(result.get("industry_benchmark", {}) or {}),
            )
            print(f"PDF report saved to: {pdf_path}")
        except Exception as exc:
            print("PDF generation failed with error:")
            print(exc)
            print("Full traceback:")
            print(traceback.format_exc())
    pprint.pprint(result)
    return result


if __name__ == "__main__":
    print("M&A financial analysis — enter run parameters (ticker is required).\n")
    ticker_in = input("Ticker symbol: ").strip()
    if not ticker_in:
        print("Error: ticker cannot be empty.")
        raise SystemExit(1)
    industry_in = input("Industry (e.g. Technology Software): ").strip()
    name_in = input("Company name (optional — press Enter to auto-detect from SEC EDGAR): ").strip()
    final_state = run_analysis(
        industry=industry_in,
        ticker=ticker_in,
        company_name=name_in or None,
    )

