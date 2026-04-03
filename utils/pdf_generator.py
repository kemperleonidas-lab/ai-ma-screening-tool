"""M&A Target Screening Report — single-page Letter PDF (ReportLab)."""

from __future__ import annotations

import re
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --- Palette ---
NAVY = colors.HexColor("#1B2A4A")
GREY_8 = colors.HexColor("#4A4A4A")  # MORE DILIGENCE
NEAR_BLACK = colors.HexColor("#2A2A2A")  # PASS
ROW_ALT = colors.HexColor("#F5F7FA")
BODY = colors.HexColor("#1A1A1A")
GREY_TEXT = colors.HexColor("#666666")
WHITE = colors.white

# Financial Health Score palette (display-only; derived from risk_score)
HEALTH_HIGH = colors.HexColor("#1B2A4A")  # navy
HEALTH_MODERATE = colors.HexColor("#666666")  # medium grey
HEALTH_LOW = colors.HexColor("#8B1A1A")  # dark red

MARGIN_CM = 1.8
HEADER_BAR_H = 1.2 * cm

# Base font sizes (pt); reduced globally by font_delta to fit one page
_BASE_SIZES = {
    "header_company": 13.0,
    "header_conf": 8.0,
    "report_title": 10.0,
    "date": 8.0,
    "label_upper": 8.0,
    "snapshot": 9.0,
    "verdict_main": 11.0,
    "verdict_sub": 8.0,
    "metrics_label": 8.0,
    "metrics_header": 8.0,
    "metrics_cell": 8.5,
    "risk_big": 28.0,
    "risk_label": 10.0,
    "inv_label": 8.0,
    "inv_item": 8.5,
    "risk_bar": 8.0,
    "risk_num": 8.5,
    "footer": 7.0,
}

MAX_FONT_REDUCTION = 4.0  # pt
FONT_STEP = 0.5


def _sanitize_for_reportlab(text: str) -> str:
    """String replacement before any ReportLab Paragraph (XML-safe)."""
    s = str(text)
    s = s.replace("M&A;", "M&A")
    s = re.sub(
        r"&(?!#(?:[0-9]{1,5}|x[0-9a-fA-F]+);|(?:amp|lt|gt|quot|apos);)",
        "&amp;",
        s,
    )
    return s


def _to_title_case(metric_key: str) -> str:
    return str(metric_key).replace("_", " ").title()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = (
            str(value)
            .replace("%", "")
            .replace("$", "")
            .replace(",", "")
            .replace("B", "")
            .replace("M", "")
            .strip()
        )
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _is_margin_key(key: str) -> bool:
    k = key.lower()
    return k in {"gross_margin", "operating_margin", "net_margin"} or "margin" in k


def _format_dollars_millions(num_millions: float) -> str:
    """Format a USD amount provided in millions as $XM or $Y.YB."""
    n = float(num_millions)
    if abs(n) >= 1000.0:
        return f"${n / 1000.0:.1f}B"
    return f"${n:,.0f}M"


def _format_metric_value(metric_key: str, value: Any) -> str:
    key = str(metric_key)
    if key == "free_cash_flow":
        if isinstance(value, (int, float)):
            return _format_dollars_millions(float(value))
        num = _to_float(value)
        if num is not None:
            return _format_dollars_millions(num)
        return _sanitize_for_reportlab(str(value))

    num = _to_float(value)
    if num is None:
        return _sanitize_for_reportlab(str(value))
    if key in {"revenues", "total_assets"}:
        return _format_dollars_millions(num)
    if key == "return_on_equity":
        return f"{num:.2f}"
    if key == "interest_coverage_ratio":
        return f"{num:.2f}x"
    if _is_margin_key(key):
        return f"{num:.2f}%"
    return f"{num:.2f}"


def _normalize_metric_key(key: str) -> str:
    # Normalize both "cash_conversion_ratio" and "Cash Conversion Ratio" style keys.
    s = str(key).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s, flags=re.UNICODE)
    s = s.strip("_")
    return s


def _peer_markup(assessment: str) -> str:
    """Render peer/industry assessment labels consistently in the PDF."""
    if assessment is None:
        return "—"
    t = str(assessment).upper().strip()
    if t in ("", "NONE", "NULL"):
        return "—"

    # Peer assessments
    if t == "STRONG":
        return "<b>ABOVE</b>"
    if t == "WEAK":
        return "<i>BELOW</i>"
    if t == "AVERAGE":
        return "IN LINE"

    # Industry-average assessments
    if t == "ABOVE":
        return "<b>ABOVE</b>"
    if t == "BELOW":
        return "<i>BELOW</i>"
    if t == "IN LINE":
        return "IN LINE"

    return "IN LINE"


def _verdict_colors(verdict: str) -> Any:
    v = verdict.upper().strip()
    if v == "PURSUE":
        return NAVY
    if v == "PASS":
        return NEAR_BLACK
    return GREY_8  # MORE DILIGENCE REQUIRED and default


def _health_level_label(health_score: float) -> str:
    if health_score >= 7.0:
        return "HIGH"
    if health_score >= 4.0:
        return "MODERATE"
    return "LOW"


def _health_level_color(level: str) -> Any:
    t = str(level).upper().strip()
    if t == "HIGH":
        return HEALTH_HIGH
    if t == "MODERATE":
        return HEALTH_MODERATE
    return HEALTH_LOW


def _pdf_page_count(data: bytes) -> int | None:
    """Return page count, or None if pypdf/PyPDF2 unavailable (skip auto-shrink loop)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            return None
    return len(PdfReader(BytesIO(data)).pages)


def _style(
    name: str,
    parent: ParagraphStyle,
    *,
    font: str = "Helvetica",
    size: float = 10,
    leading: float | None = None,
    text_color: Any = BODY,
    bold: bool = False,
    italic: bool = False,
    align: int | None = None,
) -> ParagraphStyle:
    fn = "Helvetica-Bold" if bold else ("Helvetica-Oblique" if italic else "Helvetica")
    ld = leading if leading is not None else size * 1.15
    ps = ParagraphStyle(
        name,
        parent=parent,
        fontName=fn,
        fontSize=size,
        leading=ld,
        textColor=text_color,
    )
    if align is not None:
        ps = ParagraphStyle(name, parent=ps, alignment=align)
    return ps


def _build_story(
    report_json: Dict[str, Any],
    company_name: str,
    ticker: str,
    risk_score: float,
    peer_comparison: List[Dict[str, Any]] | None,
    benchmark_analysis: Dict[str, Any] | None,
    industry_benchmark: Dict[str, Any] | None,
    content_width: float,
    sizes: Dict[str, float],
    base_normal: ParagraphStyle,
) -> List[Any]:
    story: List[Any] = []
    today = date.today().strftime("%Y-%m-%d")

    s_company = _style("hc", base_normal, size=sizes["header_company"], bold=True, text_color=WHITE)
    s_conf = _style("cf", base_normal, size=sizes["header_conf"], bold=True, text_color=WHITE)
    s_rep = _style("rt", base_normal, size=sizes["report_title"], bold=False, text_color=NAVY)
    s_date = _style("dt", base_normal, size=sizes["date"], text_color=GREY_TEXT)
    s_label = _style("lb", base_normal, size=sizes["label_upper"], bold=True, text_color=NAVY)
    s_snap = _style("sn", base_normal, size=sizes["snapshot"], italic=True, text_color=BODY)
    s_vm = _style("vm", base_normal, size=sizes["verdict_main"], bold=True, text_color=WHITE, align=1)
    s_vs = _style("vs", base_normal, size=sizes["verdict_sub"], text_color=WHITE, align=1)
    s_mlab = _style("ml", base_normal, size=sizes["metrics_label"], bold=True, text_color=NAVY)
    s_mh = _style("mh", base_normal, size=sizes["metrics_header"], bold=True, text_color=WHITE, leading=sizes["metrics_header"] * 1.1)
    s_mcell = _style("mc", base_normal, size=sizes["metrics_cell"], text_color=BODY, leading=sizes["metrics_cell"] * 1.15)
    # Health score is display-only and derived from risk_score.
    health_score = round(10.0 - float(risk_score or 0.0), 1)
    health_level = _health_level_label(health_score)
    health_color = _health_level_color(health_level)

    s_rbig = _style("rb", base_normal, size=sizes["risk_big"], bold=True, text_color=health_color, align=1)
    s_rlv = _style("rl", base_normal, size=sizes["risk_label"], text_color=health_color, align=1)
    s_ilab = _style("il", base_normal, size=sizes["inv_label"], bold=True, text_color=NAVY)
    s_iit = _style("ii", base_normal, size=sizes["inv_item"], text_color=BODY, leading=sizes["inv_item"] * 1.2)
    s_rbar = _style("rba", base_normal, size=sizes["risk_bar"], bold=True, text_color=WHITE)
    s_rnum = _style("rnu", base_normal, size=sizes["risk_num"], text_color=BODY)
    s_ft = _style("ft", base_normal, size=sizes["footer"], text_color=GREY_TEXT, leading=sizes["footer"] * 1.2)

    # --- Section 1: Header bar ---
    left_h = _sanitize_for_reportlab(f"{company_name} ({ticker})")
    hdr = Table(
        [
            [
                Paragraph(left_h, s_company),
                Paragraph(_sanitize_for_reportlab("CONFIDENTIAL"), s_conf),
            ]
        ],
        colWidths=[content_width * 0.72, content_width * 0.28],
        rowHeights=[HEADER_BAR_H],
    )
    hdr.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ]
        )
    )
    story.append(hdr)

    sub_tbl = Table(
        [
            [
                Paragraph(_sanitize_for_reportlab("M&A Target Screening Report"), s_rep),
                Paragraph(_sanitize_for_reportlab(today), s_date),
            ]
        ],
        colWidths=[content_width * 0.62, content_width * 0.38],
    )
    sub_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(sub_tbl)
    # Subtitle: size tier + industry context (user-confirmed) with thin cohort marker.
    def _title_case_tier(tier: str) -> str:
        t = str(tier or "").strip()
        if not t:
            return ""
        return " ".join([p.capitalize() for p in t.replace("_", " ").split()])

    tier_title = _title_case_tier(str(report_json.get("size_tier") or ""))
    sector = str(report_json.get("industry") or "").strip()
    peer_n = len(peer_comparison or [])
    if tier_title and sector:
        subtitle = f"{tier_title} | {sector} (user-confirmed)"
        if peer_n < 3:
            subtitle += " (thin cohort)"
        story.append(Paragraph(_sanitize_for_reportlab(subtitle), s_date))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=NAVY))

    # --- Section 2: Company snapshot ---
    story.append(Spacer(1, 4))
    story.append(Paragraph(_sanitize_for_reportlab("COMPANY OVERVIEW"), s_label))
    story.append(Spacer(1, 4))

    # Scale & context line (fiscal year + absolute size) for banker usability.
    km = report_json.get("key_metrics_table", {}) or {}
    fy = ""
    rev = ""
    ta = ""
    if isinstance(km, dict):
        fy = str(km.get("fiscal_year") or "").strip()
        rev = str(km.get("revenues") or "").strip()
        ta = str(km.get("total_assets") or "").strip()

    # Reformat $-denominated scale values consistently (switch to B at >= 1,000M).
    rev_num = _to_float(rev) if rev else None
    if rev_num is not None:
        rev = _format_dollars_millions(rev_num)
    ta_num = _to_float(ta) if ta else None
    if ta_num is not None:
        ta = _format_dollars_millions(ta_num)
    scale_parts: List[str] = []
    if fy:
        scale_parts.append(f"FY {fy}")
    if rev:
        scale_parts.append(f"Revenue: {rev}")
    if ta:
        scale_parts.append(f"Total assets: {ta}")
    if scale_parts:
        story.append(Paragraph(_sanitize_for_reportlab(" | ".join(scale_parts)), s_date))
        story.append(Spacer(1, 2))

    snap = str(report_json.get("company_snapshot", ""))
    story.append(Paragraph(_sanitize_for_reportlab(snap), s_snap))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=NAVY))

    # --- Section 3: Verdict banner ---
    verdict = str(report_json.get("screening_verdict", "MORE DILIGENCE REQUIRED")).upper().strip()
    vcol = _verdict_colors(verdict)
    rationale = str(report_json.get("verdict_rationale", ""))
    vblock = Table(
        [
            [
                Paragraph(
                    _sanitize_for_reportlab(f"<b>{verdict}</b>"),
                    s_vm,
                )
            ],
            [Paragraph(_sanitize_for_reportlab(rationale), s_vs)],
        ],
        colWidths=[content_width],
    )
    vblock.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), vcol),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(Spacer(1, 6))
    story.append(vblock)
    story.append(Spacer(1, 6))

    # --- Section 4: Benchmarks (two tables) ---
    metrics = report_json.get("key_metrics_table", {}) or {}
    normalized_metrics: Dict[str, Any] = {}
    if isinstance(metrics, dict):
        for mk, mv in metrics.items():
            normalized_metrics[_normalize_metric_key(str(mk))] = mv

    peer_benchmark = benchmark_analysis or {}
    industry_benchmark = industry_benchmark or {}

    # Keys normalized here — if cash_conversion_ratio still missing, check benchmarker.py GPT key naming.
    peer_lookup: Dict[str, Dict[str, Any]] = {}
    if isinstance(peer_benchmark, dict):
        for bk, bv in peer_benchmark.items():
            if isinstance(bv, dict):
                peer_lookup[_normalize_metric_key(str(bk))] = bv

    industry_lookup: Dict[str, Dict[str, Any]] = {}
    if isinstance(industry_benchmark, dict):
        for bk, bv in industry_benchmark.items():
            if isinstance(bv, dict):
                industry_lookup[_normalize_metric_key(str(bk))] = bv

    row_h = 12  # pt; keep compact because we render two tables
    left_w = content_width * 0.48
    gap_w = content_width * 0.04
    right_w = content_width * 0.48
    m_colw = [left_w * 0.44, left_w * 0.28, left_w * 0.28]

    def _table_rows(lookup: Dict[str, Dict[str, Any]], col_title: str, *, direct_peer: bool) -> List[List[Any]]:
        rows: List[List[Any]] = [
            [
                Paragraph(_sanitize_for_reportlab("Metric"), s_mh),
                Paragraph(_sanitize_for_reportlab("Value"), s_mh),
                Paragraph(_sanitize_for_reportlab(col_title), s_mh),
            ]
        ]
        # Render metrics present in lookup with non-null assessments.
        for nk, details in lookup.items():
            if not isinstance(details, dict):
                continue
            assess = details.get("assessment")
            if assess is None:
                continue

            raw_val = normalized_metrics.get(nk)
            if raw_val is None:
                raw_val = details.get("target_value")

            if direct_peer:
                raw_assess = str(assess).upper().strip()
                if raw_assess == "STRONG":
                    assess_display = "ABOVE"
                elif raw_assess == "AVERAGE":
                    assess_display = "IN LINE"
                elif raw_assess == "WEAK":
                    assess_display = "BELOW"
                else:
                    assess_display = str(assess)
            else:
                assess_display = _peer_markup(assess)
            rows.append(
                [
                    Paragraph(_sanitize_for_reportlab(_to_title_case(str(nk))), s_mcell),
                    Paragraph(_sanitize_for_reportlab(_format_metric_value(str(nk), raw_val)), s_mcell),
                    Paragraph(_sanitize_for_reportlab(assess_display), s_mcell),
                ]
            )
        return rows

    # Shorten column headers to avoid truncation; section labels above remain verbose.
    peers_rows = _table_rows(peer_lookup, "vs Peers", direct_peer=True)
    industry_rows = _table_rows(industry_lookup, "vs Ind. Avg", direct_peer=False)

    def _build_table(rows: List[List[Any]]) -> Table:
        rh = [row_h] * len(rows)
        t = Table(rows, colWidths=m_colw, rowHeights=rh, repeatRows=1)
        ts_cmd = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), sizes["metrics_cell"]),
            ("FONTSIZE", (0, 0), (-1, 0), sizes["metrics_header"]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CFD8E3")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]
        for i in range(1, len(rows)):
            bg = ROW_ALT if i % 2 == 0 else colors.white
            ts_cmd.append(("BACKGROUND", (0, i), (-1, i), bg))
        t.setStyle(TableStyle(ts_cmd))
        return t

    # Render table sections only when they have data rows.
    italic_style = _style("it", base_normal, size=sizes["metrics_cell"], text_color=GREY_TEXT, italic=True)

    left_stack: List[Any] = [
        Paragraph(_sanitize_for_reportlab("KEY FINANCIAL METRICS"), s_mlab),
        Spacer(1, 4),
    ]

    if len(industry_rows) <= 1:
        left_stack.extend(
            [
                Paragraph(
                    _sanitize_for_reportlab("Industry average data not available for this sector"),
                    italic_style,
                ),
                Spacer(1, 8),
            ]
        )
    else:
        mt_industry = _build_table(industry_rows)
        left_stack.extend(
            [
                Paragraph(_sanitize_for_reportlab("vs Industry Average"), s_mlab),
                Spacer(1, 2),
                mt_industry,
                Spacer(1, 6),
            ]
        )

    if len(peers_rows) <= 1:
        left_stack.append(
            Paragraph(
                _sanitize_for_reportlab("No peer comparison data available for this sector"),
                italic_style,
            )
        )
    else:
        mt_peers = _build_table(peers_rows)
        left_stack.extend(
            [
                Paragraph(_sanitize_for_reportlab("vs Direct Peers"), s_mlab),
                Spacer(1, 2),
                mt_peers,
            ]
        )

    inv_items = list(report_json.get("investment_considerations", []) or [])[:3]
    while len(inv_items) < 3:
        inv_items.append("—")
    bullet_char = '<font color="#1B2A4A">&#9632;</font>'
    inv_parts: List[Any] = [
        Paragraph(_sanitize_for_reportlab("SCREENING ASSESSMENT"), s_mlab),
        Spacer(1, 4),
        Paragraph(
            _sanitize_for_reportlab(f"{health_score:.1f} / 10"),
            s_rbig,
        ),
        Paragraph(_sanitize_for_reportlab(health_level), s_rlv),
        Spacer(1, 12),
        Paragraph(_sanitize_for_reportlab("Investment Considerations"), s_ilab),
        Spacer(1, 4),
    ]
    for i, line in enumerate(inv_items):
        inv_parts.append(
            Paragraph(
                _sanitize_for_reportlab(f"{bullet_char} {line}"),
                s_iit,
            )
        )
        if i < len(inv_items) - 1:
            inv_parts.append(Spacer(1, 4))

    right_stack = inv_parts

    left_cell = Table([[f] for f in left_stack], colWidths=[left_w])
    left_cell.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    right_cell = Table([[f] for f in right_stack], colWidths=[right_w])
    right_cell.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    gap_cell = Table([[""]], colWidths=[gap_w])

    two = Table([[left_cell, gap_cell, right_cell]], colWidths=[left_w, gap_w, right_w])
    two.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(two)
    story.append(Spacer(1, 8))

    # --- Section 5: Risk factors ---
    rf = list(report_json.get("risk_factors", []) or [])[:3]
    while len(rf) < 3:
        rf.append("—")

    rf_header = Table(
        [[Paragraph(_sanitize_for_reportlab("KEY RISK FACTORS"), s_rbar)]],
        colWidths=[content_width],
    )
    rf_header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(rf_header)
    story.append(Spacer(1, 4))
    for i, rtxt in enumerate(rf):
        story.append(
            Paragraph(
                _sanitize_for_reportlab(f"{i + 1}. {rtxt}"),
                s_rnum,
            )
        )
        if i < len(rf) - 1:
            story.append(Spacer(1, 5))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))

    # --- Section 6: Footer ---
    conf = _to_float(report_json.get("confidence_score"))
    conf = 0.0 if conf is None else max(0.0, min(1.0, conf))
    disc = (
        "This report is generated by an automated system and is not investment advice. "
        "Do not use as the sole basis for any transaction decision."
        " Peer benchmarking compares the target against companies of equivalent size tier within the same sector. "
        "Thin cohort: fewer than 3 peers available."
    )
    foot_right = f"Data source: SEC EDGAR | Confidence: {conf * 100:.0f}%"
    ft = Table(
        [
            [
                Paragraph(_sanitize_for_reportlab(disc), s_ft),
                Paragraph(_sanitize_for_reportlab(foot_right), s_ft),
            ]
        ],
        colWidths=[content_width * 0.58, content_width * 0.42],
    )
    ft.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ]
        )
    )
    story.append(Spacer(1, 4))
    story.append(ft)

    return story


def generate_pdf_report(
    report_json: Dict[str, Any],
    company_name: str,
    ticker: str,
    risk_score: float,
    peer_comparison: List[Dict[str, Any]] | None = None,
    benchmark_analysis: Dict[str, Any] | None = None,
    industry_benchmark: Dict[str, Any] | None = None,
) -> str:
    """Generate a one-page M&A target screening PDF; shrinks fonts if needed to fit."""
    print("generate_pdf_report() started.")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    output_path = output_dir / f"{ticker}_{today}_report.pdf"

    margin = MARGIN_CM * cm
    content_width = letter[0] - 2 * margin

    base = getSampleStyleSheet()
    base_normal = base["Normal"]

    delta = 0.0
    last_path = str(output_path)
    while delta <= MAX_FONT_REDUCTION + 1e-6:
        sizes = {k: max(5.0, v - delta) for k, v in _BASE_SIZES.items()}
        story = _build_story(
            report_json,
            company_name,
            ticker,
            risk_score,
            peer_comparison,
            benchmark_analysis,
            industry_benchmark,
            content_width,
            sizes,
            base_normal,
        )

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
        )
        doc.build(story)
        data = buf.getvalue()
        n_pages = _pdf_page_count(data)
        if n_pages is None or n_pages <= 1:
            output_path.write_bytes(data)
            last_path = str(output_path)
            if delta > 0:
                print(f"PDF: reduced fonts by {delta:.1f} pt to fit one page.")
            break
        delta += FONT_STEP
    else:
        # Fallback: write last attempt even if >1 page
        buf = BytesIO()
        sizes = {k: max(5.0, v - MAX_FONT_REDUCTION) for k, v in _BASE_SIZES.items()}
        story = _build_story(
            report_json,
            company_name,
            ticker,
            risk_score,
            peer_comparison,
            benchmark_analysis,
            industry_benchmark,
            content_width,
            sizes,
            base_normal,
        )
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
        )
        doc.build(story)
        output_path.write_bytes(buf.getvalue())
        print("Warning: PDF may exceed one page; max font reduction applied.")

    return last_path
