from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class FinancialState(TypedDict):
    """Container for all shared state across financial analysis agents."""

    #: Human-readable company name as provided by the user, or resolved from SEC EDGAR.
    #: May start empty; the parser can set this from the company facts ``entityName`` when missing.
    #: Used by all downstream agents for logging, display, and report generation.
    company_name: str

    #: High-level industry label for the company (e.g. "Software", "Banking").
    #: Populated by the Orchestration/CLI layer or a ClassificationAgent.
    #: Used by analysis agents to select relevant benchmarks, templates, and heuristics.
    industry: str

    #: SEC Standard Industrial Classification code (digits as string), from submissions API.
    sic_code: str

    #: SEC SIC description from submissions API.
    sic_description: str

    #: Stock ticker symbol for the company (e.g. "AAPL", "MSFT").
    #: Populated by the Orchestration/CLI layer when a new analysis is started.
    #: Used by the ParsingAgent to query SEC EDGAR APIs for structured data.
    ticker: str

    #: Flag indicating whether all required filing documents have been successfully parsed.
    #: Populated by the ParsingAgent after attempting to extract text and structured data.
    #: Used by downstream agents to decide whether they can safely rely on parsed content.
    documents_parsed: bool

    #: Structured financial data extracted from filings or external data sources.
    #: Expected structure:
    #: {
    #:     "balance_sheet": {...},
    #:     "income_statement": {...},
    #:     "cash_flow": {...},
    #: }
    #: Populated by the FinancialExtractionAgent.
    #: Used by metric, trend, benchmark, and risk-analysis agents.
    financial_data: Dict[str, Dict[str, Any]]

    #: Confidence score \[0, 1\] for the quality of parsing and extraction.
    #: Populated by the ParsingAgent and/or FinancialExtractionAgent.
    #: Used by the OrchestrationAgent to decide whether to re-parse or request user review.
    parsing_confidence: float

    #: Post-parse validation score \[0, 1\] from required fields and sanity checks on ``financial_data``.
    #: Populated by the Validator agent between parser and metrics.
    data_quality_score: float

    #: Derived point-in-time financial metrics (margins, leverage ratios, growth rates, etc.).
    #: Populated by the MetricsAgent based on `financial_data`.
    #: Used by TrendAnalysisAgent, BenchmarkAgent, and ReportingAgent.
    metrics: Dict[str, Any]

    #: Time-series and multi-period trends computed from metrics and financial statements.
    #: Populated by the TrendAnalysisAgent.
    #: Used by BenchmarkAgent, RiskAnalysisAgent, and ReportingAgent.
    trends: Dict[str, Any]

    #: Benchmark results comparing the company to sector/industry/peer group baselines.
    #: Populated by the BenchmarkAgent using `metrics`, `trends`, and `industry`.
    #: Used by ReportingAgent to highlight relative over- and under-performance.
    benchmark_analysis: Dict[str, Any]

    #: Benchmark results comparing the company to sector/industry-wide averages (Damodaran industry averages).
    #: Populated by the BenchmarkAgent using `metrics` and `industry`.
    #: Used by RedFlagDetectionAgent and PDF reporting to avoid false flags.
    industry_benchmark: Dict[str, Any]

    #: Peer comparison results versus specific competitor companies.
    #: Populated by the PeerComparisonAgent.
    #: Used by ReportingAgent to add competitive context to the final narrative.
    peer_comparison: Dict[str, Any]

    #: List of textual descriptions of potential issues, anomalies, and red flags.
    #: Populated by the RedFlagDetectionAgent using financial signals and narrative context.
    #: Used by RiskAnalysisAgent and surfaced prominently in the final report.
    red_flags: List[str]

    #: Aggregate numeric risk score on a \[0, 10\] scale (0 = lowest risk, 10 = highest).
    #: Populated by the RiskAnalysisAgent based on red flags, leverage, volatility, etc.
    #: Used by ReportingAgent and UI to quickly communicate overall riskiness.
    risk_score: float

    #: Fully assembled human-readable report (markdown or rich text).
    #: Populated by the ReportingAgent once all upstream analysis is complete.
    #: Used as the final deliverable presented to the user or exported to external systems.
    final_report: str

    #: Structured JSON payload produced by the ReportingAgent before text/PDF rendering.
    #: Populated by the ReportingAgent from the parsed LLM response.
    #: Used by downstream renderers (e.g. PDF generator) and API consumers.
    report_json: Dict[str, Any]

    #: One-sentence business / market snapshot from the screening report.
    company_snapshot: str

    #: Strategic M&A opportunity bullets (typically three) from the screening report.
    investment_considerations: List[str]

    #: M&A-relevant risk bullets (typically three) from the screening report.
    risk_factors: List[str]

    #: Screening outcome: PURSUE, PASS, or MORE DILIGENCE REQUIRED.
    screening_verdict: str

    #: Overall confidence score \[0, 1\] in the final analysis and report.
    #: Populated by the OrchestrationAgent by aggregating individual agent confidences.
    #: Used by the UI and any automated consumers to judge reliability of the output.
    confidence: float

    #: List of error messages and warnings encountered during the pipeline.
    #: Populated incrementally by all agents when recoverable or fatal issues occur.
    #: Used by the OrchestrationAgent and UI for debugging and user guidance.
    errors: List[str]


def get_initial_state(
    industry: str,
    ticker: str,
    company_name: str = "",
) -> FinancialState:
    """Return a freshly initialized `FinancialState` for a new analysis run.

    ``company_name`` may be omitted or empty; the parser can fill it from SEC EDGAR company facts.
    """

    return FinancialState(
        company_name=company_name,
        industry=industry,
        sic_code="",
        sic_description="",
        ticker=ticker,
        documents_parsed=False,
        financial_data={
            "balance_sheet": {},
            "income_statement": {},
            "cash_flow": {},
        },
        parsing_confidence=0.0,
        data_quality_score=0.0,
        metrics={},
        trends={},
        benchmark_analysis={},
        industry_benchmark={},
        peer_comparison={},
        red_flags=[],
        risk_score=0.0,
        final_report="",
        report_json={},
        company_snapshot="",
        investment_considerations=[],
        risk_factors=[],
        screening_verdict="",
        confidence=0.0,
        errors=[],
    )

