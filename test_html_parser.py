from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
SEC_USER_AGENT = "financial-analyzer contact@example.com"
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "application/json",
}

# CLI-controlled cache toggle (set in main()).
NO_CACHE = False


def _fetch_text(url: str) -> str:
    global NO_CACHE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_cache_dir = os.path.join(script_dir, "cache", "html")

    def _html_cache_path(u: str) -> str:
        safe = u.replace("://", "_").replace("/", "_").replace(".", "_")
        safe = safe[:200]
        return os.path.join(html_cache_dir, safe + ".html")

    cache_path = _html_cache_path(url) if not NO_CACHE else None
    if cache_path is not None and os.path.exists(cache_path):
        filename = os.path.basename(cache_path)
        print(f"[CACHE HIT] html: {filename}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Request failed for {url} (HTTP {resp.status_code}): {resp.text[:500]}")

    text = resp.text
    if cache_path is not None:
        os.makedirs(html_cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def _fetch_json(url: str) -> Dict[str, Any]:
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Request failed for {url} (HTTP {resp.status_code}): {resp.text[:500]}")
    try:
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Failed to decode JSON for {url}: {exc}") from exc


def _fetch_submissions_metadata(cik_padded: str) -> Dict[str, Any]:
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    return _fetch_json(url)


def _extract_sic_code(submissions: Dict[str, Any]) -> Optional[int]:
    sic_val = submissions.get("sic")
    if sic_val is None:
        return None
    try:
        return int(str(sic_val).strip())
    except Exception:
        return None


def _extract_cik_from_atom(atom_xml: str) -> Tuple[str, str]:
    """
    Extract the <cik> value from browse-edgar Atom XML response.

    Returns:
      (cik_numeric, cik_padded_10)
    """
    root = ET.fromstring(atom_xml)
    # Atom responses use namespaces; search by suffix.
    cik_val: Optional[str] = None
    for elem in root.iter():
        if isinstance(elem.tag, str) and elem.tag.endswith("cik") and elem.text:
            cik_val = elem.text.strip()
            break
    if not cik_val:
        raise RuntimeError("Could not find <cik> in Atom XML response.")

    cik_padded = cik_val
    # Keep exactly what the Atom response gives us as "padded" CIK.
    # Compute numeric by stripping leading zeros (and fall back to padded if all zeros).
    cik_numeric = cik_padded.lstrip("0")
    if cik_numeric == "":
        cik_numeric = cik_padded
    return cik_numeric, cik_padded


def step1_find_cik(ticker: str) -> Tuple[str, str]:
    url = (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&CIK={ticker}"
        "&type=10-K&dateb=&owner=include&count=1&search_text=&output=atom"
    )
    atom_xml = _fetch_text(url)
    cik_numeric, cik_padded = _extract_cik_from_atom(atom_xml)
    print(f"CIK found: numeric={cik_numeric}, padded={cik_padded}")
    return cik_numeric, cik_padded


def step2_find_latest_10k_accession(cik_padded: str) -> str:
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    data = _fetch_json(url)
    filings = data.get("filings") or {}

    # SEC submissions JSON uses parallel arrays under filings.recent, not a list of objects.
    recent = filings.get("recent") or {}
    if not isinstance(recent, dict):
        raise RuntimeError("Unexpected shape from submissions JSON: filings.recent is not an object.")

    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    if not isinstance(forms, list) or not isinstance(accessions, list):
        raise RuntimeError("Unexpected shape from submissions JSON: filings.recent.form/accessionNumber not lists.")

    n = min(len(forms), len(accessions))
    for i in range(n):
        if forms[i] == "10-K":
            accession = accessions[i]
            if not accession:
                raise RuntimeError("Found a 10-K entry but accessionNumber is missing/empty.")
            print(f"Most recent 10-K accession: {accession}")
            return str(accession)

    raise RuntimeError("Could not find any '10-K' entry in filings.recent.")


def step3_find_primary_document_url(
    cik_numeric: str,
    cik_padded: str,
    accession_with_dashes: str,
) -> str:
    folder = accession_with_dashes.replace("-", "")
    index_url_plan = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_numeric}/{folder}/{accession_with_dashes}-index.json"
    )
    index_data: Dict[str, Any] | None = None
    try:
        index_data = _fetch_json(index_url_plan)
    except RuntimeError as exc:
        # Some filings expose the filing index at {folder}/index.json rather than
        # {accession}-index.json. Fall back to that URL if the plan URL 404s.
        msg = str(exc)
        if "HTTP 404" not in msg:
            raise
    if index_data is None:
        index_url_fallback = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{cik_numeric}/{folder}/index.json"
        )
        index_data = _fetch_json(index_url_fallback)

    primary_filename: Optional[str] = None

    # Prefer the plan-shaped schema if present.
    documents = index_data.get("documents")
    if isinstance(documents, list) and documents:
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            if doc.get("type") != "10-K":
                continue
            filename = doc.get("document")
            if not isinstance(filename, str):
                continue
            if filename.lower().endswith(".htm") or filename.lower().endswith(".html"):
                primary_filename = filename
                break

    # Fallback for the actual SEC shape observed in practice:
    # index.json can contain only `directory.item` entries (no `documents` array).
    if primary_filename is None:
        directory = index_data.get("directory") or {}
        items = directory.get("item") or []
        if not isinstance(items, list) or not items:
            raise RuntimeError("Unexpected shape from index JSON: no documents and no directory.item list.")

        candidates: list[tuple[float, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str):
                continue
            if not (name.lower().endswith(".htm") or name.lower().endswith(".html")):
                continue
            size_raw = item.get("size")
            size_num = -1.0
            if isinstance(size_raw, str) and size_raw.strip() != "":
                try:
                    size_num = float(size_raw.replace(",", ""))
                except ValueError:
                    size_num = -1.0
            candidates.append((size_num, name))

        if not candidates:
            raise RuntimeError("Could not locate any .htm/.html files in directory.item listing.")

        # Heuristic: primary filing doc tends to be the largest HTML document in the folder.
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        primary_filename = candidates[0][1]

    if not primary_filename:
        raise RuntimeError("Could not locate primary 10-K .htm/.html document in index JSON.")

    primary_url = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_numeric}/{folder}/{primary_filename}"
    )
    print(f"Primary document URL: {primary_url}")
    return primary_url


def _clean_cell_value(raw: str) -> str:
    """
    Clean a cell value:
    - strip all whitespace, dollar signs, and parentheses
    - convert (1,234) -> -1234
    """
    s = raw or ""
    s = s.replace("$", "")
    s = s.strip()

    # Detect negative numbers presented as parentheses: (1,234) -> -1234
    m = re.fullmatch(r"\(([-+]?[0-9,\.]+)\)", s)
    if m:
        inner = m.group(1).replace(",", "").strip()
        # Normalize cases where parentheses already include a sign inside.
        if inner.startswith("-"):
            inner = inner[1:]
        return f"-{inner}"

    # For non-negative values, remove parentheses as requested.
    s = s.replace("(", "").replace(")", "")
    # Strip whitespace (keep digits/commas/dots etc.).
    s = re.sub(r"\s+", "", s)
    return s


def find_financial_tables(
    soup: BeautifulSoup,
    ticker: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Find the three financial statement tables:
      - income statement
      - balance sheet
      - cash flow

    Detection strategy:
    - find text nodes containing any of the phrases (case-insensitive)
    - from each matched text node, walk forward until the next <table>
    - for each statement type, pick the first candidate table with > 5 rows
    """

    # Numeric density gate (shared by U3 and the outer numeric gate).
    # U3: require at least 4 cells with values matching \d{3,}.
    U3_MIN_NUMERIC_DENSITY_CELLS = 4
    MIN_NUMERIC_CELLS = U3_MIN_NUMERIC_DENSITY_CELLS

    # I2: segment table rejection.
    SEGMENT_KEYWORD_REJECTION_THRESHOLD = 4

    # U1: TOC/navigational numeric table rejection.
    # U1_PAGE_NUM_MAX_DIGITS is intentionally conservative: many legitimate
    # financial line items in SEC statements can be 3-digit values (e.g.,
    # $500m) and we don't want those to be misclassified as TOC/page numbers.
    U1_PAGE_NUM_MAX_DIGITS = 2
    U1_FINANCIAL_FIGURE_MIN_DIGITS = 4

    # B2/C2: multi-column supplemental rejection.
    MAX_DATA_COLUMNS = 6

    # Candidate-generation-only phrases, split into tiers.
    # Tiers exist to reduce false positives: lower-tier phrases are used only
    # when higher-tier phrases generate *zero* candidates.
    #
    # Phrase matching is not the final selection step because statement
    # section names can appear in TOC, segment schedules, or supplemental
    # disclosures.
    STATEMENT_PHRASE_TIERS: Dict[str, Dict[str, list[str]]] = {
        "income_statement": {
            "tier1": [
                "consolidated statements of operations",
                "consolidated statements of income",
                "statements of operations",
                "statements of income",
            ],
            "tier2": [
                "consolidated statements of earnings",
                "statement of operations",
                "statement of income",
                "statement of earnings",
                # CAT / alternative subtitles (explicitly Tier 2).
                "results of operations",
                "consolidated results of operations",
                "statements of earnings",
                "consolidated statements of comprehensive income",
            ],
            "tier3": [
                "income from operations",
                "income statement",
                "income statements",
            ],
        },
        "balance_sheet": {
            "tier1": [
                "consolidated balance sheets",
                "consolidated statements of financial position",
            ],
            "tier2": [
                "balance sheet",
                "statement of financial position",
                "financial position",
            ],
            "tier3": [
                "consolidated balance sheet",
                "balance sheets",
            ],
        },
        "cash_flow": {
            "tier1": [
                "consolidated statements of cash flows",
            ],
            "tier2": [
                "statements of cash flows",
                "consolidated statement of cash flows",
                "statement of cash flow",
                "cash flow statement",
            ],
            "tier3": [
                "cash flows",
                "cash flows from operating",
                "cash flows statements",
                "cash flow statements",
            ],
        },
    }

    results: Dict[str, Any] = {}
    balance_sheet_b1_rejects: list[dict[str, Any]] = []
    seen_b1_reject_ids: set[int] = set()

    # Optional diagnostics: purely additive visibility into the candidate
    # generation + winner-selection process.
    debug_report: Dict[str, Any] = {} if debug else {}

    def _leftmost_nonempty_row_labels(table: Any, limit: int = 8) -> list[str]:
        labels: list[str] = []
        if table is None:
            return labels
        try:
            for tr in table.find_all("tr"):
                if len(labels) >= limit:
                    break
                cells = tr.find_all(["td", "th"])
                if not cells:
                    continue
                # Take the first non-empty cell in this row.
                for c in cells:
                    raw = c.get_text(" ", strip=True)
                    norm = re.sub(r"\s+", " ", raw).strip()
                    if norm:
                        labels.append(norm)
                        break
        except Exception:
            # Best-effort only; diagnostics must never break extraction.
            return labels
        return labels

    def _finalize_debug_for_statement(
        statement_type: str,
        phrase_candidates_debug: list[dict[str, Any]],
        phrase_candidates_before_content_check: int,
        rejected_phrase_candidates_debug: list[dict[str, Any]],
        winner_table: Any | None,
        winner_reason: str | None,
        no_table_reason: str | None,
    ) -> None:
        debug_report[statement_type] = {
            "phrase_candidate_count_before_content_check": phrase_candidates_before_content_check,
            "phrase_candidate_count": len(phrase_candidates_debug),
            "phrase_candidates": phrase_candidates_debug,
            "rejected_phrase_candidates": rejected_phrase_candidates_debug,
            "winner_table": winner_table,
            "winner_reason": winner_reason,
            "no_table_reason": no_table_reason,
        }

    statement_keywords: Dict[str, list[str]] = {
        "income_statement": [
            "revenue",
            "sales",
            "net sales",
            "total revenues",
            "total sales",
            "sales and revenues",
            "gross profit",
            "gross margin",
            "operating income",
            "cost of revenue",
            "cost of products",
            "cost of goods",
            "net income",
            "net earnings",
            "earnings",
            "profit",
            "profit before",
            "earnings before",
            "provision for income",
            "benefit from income",
            "statements of operations",
            "consolidated statements of operations",
            "results of operations",
            "income from operations",
        ],
        "balance_sheet": [
            "total assets",
            "total liabilities",
            "total current assets",
            "total current liabilities",
            # Avoid generic terms like "equity"/"stockholders" because many
            # index/summary tables include them.
        ],
        "cash_flow": [
            "net cash",
            "operating activities",
            "investing activities",
            "financing activities",
            "cash flows",
        ],
    }

    statement_exclude_keywords: Dict[str, list[str]] = {
        # Prevent selecting income statement tables when looking for balance
        # sheet tables.
        "balance_sheet": [
            "totalrevenue",
            "costofrevenue",
            "grossmargin",
            "operatingincome",
            "netincome",
            "earningspershare",
        ],
        # Cash flow statements typically include cash keywords, but we'll keep
        # this minimal.
        "cash_flow": [
            "earningspershare",
        ],
    }

    statement_required_keywords: Dict[str, list[str]] = {
        # For the main balance sheet, we want the table that actually has
        # total assets/liabilities, not lease/disclosure sub-tables.
        "balance_sheet": [
            "total assets",
            "total liabilities",
        ]
    }

    statement_validation_terms: Dict[str, list[str]] = {
        "income_statement": [
            "revenue",
            "sales",
            "net sales",
            "total revenues",
            "total sales",
            "sales and revenues",
            "net income",
            "net earnings",
            "earnings",
            "profit before",
            "earnings before",
            "provision for income",
            "benefit from income",
            "operations",
        ],
        "balance_sheet": ["total assets", "current assets", "stockholders", "liabilities"],
        # Some filings (notably MSFT iXBRL) label cash-flow rows without the
        # exact "operating activities / investing activities / financing activities"
        # phrases, so we include broader net-cash markers.
        "cash_flow": [
            "operating activities",
            "investing activities",
            "financing activities",
            "netcash",
            "operations",
            "investing",
            "financing",
        ],
    }

    def _count_numeric_cells(table: Any) -> int:
        """
        Count cells that are standalone numbers with >=3 digits.
        Used to reject TOC/navigation/summary tables.
        """
        if table is None:
            return 0
        try:
            cells = table.find_all(["td", "th"])
        except Exception:
            return 0
        count = 0
        for c in cells:
            try:
                txt = c.get_text(" ", strip=True)
            except Exception:
                continue
            s = (txt or "").strip()
            if not s:
                continue
            # Skip common non-numeric placeholders.
            if s in {"—", "-", "–", "N/A", "NA"}:
                continue

            # Accept typical SEC number formatting:
            # - commas: "101,832"
            # - parentheses for negatives: "(7,056)"
            # - optional leading '-'
            s_norm = s.replace(",", "").replace("$", "").replace("(", "").replace(")", "")
            if re.fullmatch(r"-?\d{3,}(\.\d+)?", s_norm):
                count += 1
        return count

    def _table_passes_numeric_gate(table: Any) -> bool:
        return _count_numeric_cells(table) >= MIN_NUMERIC_CELLS

    def _data_column_count(table: Any) -> int:
        """
        Approximate data-column count by counting header-row <th> entries (if
        present); otherwise fall back to the first non-empty row's <td> cells.
        """
        if table is None:
            return 0
        try:
            # Look at a limited window near the top of the statement for the
            # most "header-like" row.
            first_rows = list(table.find_all("tr"))[:12]
        except Exception:
            return 0

        best_year_cell_count = -1
        best_header_cell_count = 0

        for tr in first_rows:
            ths = tr.find_all("th")
            tds = tr.find_all("td")
            if not ths and not tds:
                continue

            header_cells = ths if ths else tds
            header_cell_count = len(header_cells)
            year_cell_count = 0
            for cell in header_cells:
                cell_text = cell.get_text(" ", strip=True).lower()
                if re.search(r"\b20\d{2}\b", cell_text):
                    year_cell_count += 1

            # Prefer a header row with the most distinct year-carrying header cells.
            if year_cell_count > best_year_cell_count:
                best_year_cell_count = year_cell_count
                best_header_cell_count = header_cell_count

        # If we detected year-carrying header cells, treat those as "data columns"
        # for the B2/C2 multi-column rejection gates.
        if best_year_cell_count > 0:
            return best_year_cell_count

        # Absolute fallback: count columns in the first row that has cells.
        for tr in first_rows:
            ths = tr.find_all("th")
            if ths:
                return len(ths)
            tds = tr.find_all("td")
            if tds:
                return len(tds)
        return 0

    def _normalize_cell_digits(cell_text: str) -> str:
        # Remove common formatting artifacts but keep digits only.
        s = (cell_text or "").strip()
        s = s.replace(",", "").replace("$", "")
        s = s.replace("(", "").replace(")", "")
        # Remove whitespace and non-digits; keep sign out of the way.
        digits = re.sub(r"\D+", "", s)
        return digits

    def _table_rejection_reason(table: Any, statement_type: str) -> str | None:
        """
        Deterministic rejection gate for statement tables (no LLM calls).
        Returns a string rule id when rejected, otherwise None.
        """
        if table is None:
            return "NULL_TABLE"

        try:
            text_raw = table.get_text(" ", strip=True).lower()
        except Exception:
            return "TABLE_TEXT_EXTRACT_FAIL"

        # Whitespace-stripped lowercase text used by all rules.
        t_ns = re.sub(r"\s+", "", text_raw)

        # Universal rules
        # U1 — TOC/navigational numeric table rejection.
        page_num_cells = 0
        financial_fig_cells = 0
        try:
            cells = table.find_all(["td", "th"])
        except Exception:
            cells = []
        for c in cells:
            try:
                cell_txt = c.get_text(" ", strip=True)
            except Exception:
                continue
            digits = _normalize_cell_digits(cell_txt)
            if not digits:
                continue
            if len(digits) <= U1_PAGE_NUM_MAX_DIGITS:
                page_num_cells += 1
            if len(digits) >= U1_FINANCIAL_FIGURE_MIN_DIGITS:
                financial_fig_cells += 1
        if page_num_cells > financial_fig_cells:
            return "U1_TOC_NUMERIC_REJECT"

        # U2 — Supplemental/consolidating data rejection.
        supplemental_tokens = (
            "supplementaldata",
            "supplementalconsolidating",
            "consolidatingadjustments",
            "supplementaldatafor",
        )
        nonempty_rows_checked = 0
        for tr in table.find_all("tr"):
            if nonempty_rows_checked >= 3:
                break
            try:
                row_txt = tr.get_text(" ", strip=True).lower()
            except Exception:
                continue
            row_ns = re.sub(r"\s+", "", row_txt)
            if not row_ns:
                continue
            nonempty_rows_checked += 1
            if any(tok in row_ns for tok in supplemental_tokens):
                return "U2_SUPPLEMENTAL_CONSOLIDATING_REJECT"

        # U3 — Minimum numeric density.
        numeric_cells = _count_numeric_cells(table)
        if numeric_cells < U3_MIN_NUMERIC_DENSITY_CELLS:
            return "U3_MIN_NUMERIC_DENSITY_REJECT"

        # Statement-specific rules
        if statement_type == "income_statement":
            # I1 — Cash flow language rejection.
            cashflow_marker_combined = (
                t_ns.count("netcash")
                + t_ns.count("operatingactivities")
                + t_ns.count("investingactivities")
                + t_ns.count("financingactivities")
            )
            if cashflow_marker_combined >= 3:
                return "I1_CASH_FLOW_LANGUAGE_REJECT"

            # I2 — Segment table rejection.
            if t_ns.count("segment") >= SEGMENT_KEYWORD_REJECTION_THRESHOLD:
                return "I2_SEGMENT_THRESHOLD_REJECT"
            if (
                "reconciliation" in t_ns
                or "adjustedebitda" in t_ns
                or "adjustedoperating" in t_ns
                or "totalsegment" in t_ns
            ):
                return "I2_RECONCILIATION_OR_SEGMENT_REJECT"

            # I3 — Bottom-line anchor requirement.
            bottom_line_tokens = (
                "netincome",
                "netearnings",
                "netloss",
                # Common bottom-line labels across sectors/filers.
                "profit",  # e.g. CAT "Profit"
                "earnings",
                "profitfor",
                "earningsfor",
                "profitfortheyear",
                "earningsfortheyear",
                "netprofitloss",
            )
            if not any(tok in t_ns for tok in bottom_line_tokens):
                return "I3_BOTTOM_LINE_ANCHOR_MISSING"

            # I4 — Revenue anchor requirement.
            revenue_tokens_high_conf = (
                "revenue",
                "netsales",
                "totalsales",
                "salesandrevenue",
                "totalrevenue",
                "operatingrevenue",
                # JNJ / CAT-style top-line labels.
                "salestocustomers",
                "salesandotherrevenue",
                "productsales",
                "servicesales",
            )
            has_high_conf_revenue_anchor = any(tok in t_ns for tok in revenue_tokens_high_conf)
            if not has_high_conf_revenue_anchor:
                # Lower-confidence fallback: whole-word "sales" only.
                # Use regex boundary matching to avoid accidental matches like
                # "capitalexpenditures".
                text_space_norm = re.sub(r"\s+", " ", text_raw.lower()).strip()
                if re.search(r"\bsales\b", text_space_norm):
                    return None
                return "I4_REVENUE_ANCHOR_MISSING"

            return None

        if statement_type == "balance_sheet":
            # B1 — Must contain total assets + at least one liabilities anchor.
            if "totalassets" not in t_ns:
                return "B1_TOTAL_ASSETS_MISSING"
            if not (
                "totalliabilities" in t_ns
                or "liabilitiesandstockholders" in t_ns
                or "liabilitiesandshareholders" in t_ns
            ):
                return "B1_LIABILITIES_ANCHOR_MISSING"

            # B2 — Multi-column supplemental rejection.
            if _data_column_count(table) > MAX_DATA_COLUMNS:
                return "B2_SUPPLEMENTAL_MULTI_COLUMN_REJECT"

            return None

        if statement_type == "cash_flow":
            # C1 — Primary statement completeness (two-path).
            # Path A: total-line style — "net cash" near each activity type.
            # Path B: section-header style (AAPL/MCD) — three "... activities"
            # headers plus a net income reconciliation line, without requiring
            # "net cash" on the same line as each section header.

            def _substrings_within_distance(
                s: str, a: str, b: str, max_gap: int = 30
            ) -> bool:
                """True if some occurrence of `a` is within max_gap chars of some `b`."""
                if a not in s or b not in s:
                    return False
                for ma in re.finditer(re.escape(a), s):
                    ia, ea = ma.start(), ma.end()
                    for mb in re.finditer(re.escape(b), s):
                        ib, eb = mb.start(), mb.end()
                        if max(ia, ib) < min(ea, eb):
                            return True
                        if ea <= ib:
                            if ib - ea <= max_gap:
                                return True
                        elif eb <= ia:
                            if ia - eb <= max_gap:
                                return True
                return False

            path_a_operating = _substrings_within_distance(t_ns, "netcash", "operating")
            path_a_investing = _substrings_within_distance(t_ns, "netcash", "investing")
            path_a_financing = _substrings_within_distance(t_ns, "netcash", "financing")
            path_a = sum([path_a_operating, path_a_investing, path_a_financing]) >= 2

            path_b_headers = (
                "operatingactivities" in t_ns
                and "investingactivities" in t_ns
                and "financingactivities" in t_ns
            )
            path_b_recon = any(
                tok in t_ns
                for tok in ("netincome", "netearnings", "netloss", "netprofit")
            )
            path_b = path_b_headers and path_b_recon

            if not path_a and not path_b:
                return "C1_PRIMARY_COMPLETENESS_REJECT"

            # C2 — Multi-column supplemental rejection.
            if _data_column_count(table) > MAX_DATA_COLUMNS:
                return "C2_SUPPLEMENTAL_MULTI_COLUMN_REJECT"

            return None

        return None

    def _table_passes_content_check(table: Any, statement_type: str) -> bool:
        return _table_rejection_reason(table, statement_type) is None

    def table_contains_any_terms(table: Any, terms: list[str]) -> bool:
        if table is None:
            return False
        try:
            text_raw = table.get_text(" ", strip=True).lower()
        except Exception:
            return False

        # Two matching modes:
        # - For multiword terms: compare on whitespace-stripped text
        #   (SEC tables often concatenate words).
        # - For single-word terms: require whole-word match on space-normalized
        #   text to avoid substring false positives (e.g. "deferred revenues"
        #   matching "revenue").
        text_ws_removed = re.sub(r"\s+", "", text_raw)
        text_space_norm = re.sub(r"\s+", " ", text_raw)

        for term in terms:
            term_l = (term or "").lower().strip()
            if not term_l:
                continue

            if " " in term_l:
                term_removed = re.sub(r"\s+", "", term_l)
                if term_removed and term_removed in text_ws_removed:
                    return True
            else:
                pat = r"\b" + re.escape(term_l) + r"\b"
                if re.search(pat, text_space_norm):
                    return True
        return False

    def next_table_after(text_node: Any) -> Optional[Any]:
        # Walk forward in DOM order to find the next <table>.
        # Primary search: only scan the next-elements window for speed.
        # Fallback: if no table is found in that window, scan the same parent
        # container descendants after the start node.
        parent = getattr(text_node, "parent", None)

        # Primary: search within a limited window.
        for i, el in enumerate(getattr(text_node, "next_elements", []) or []):
            if getattr(el, "name", None) == "table":
                return el
            if i >= 20:
                break

        # Fallback: search in the same parent container after this node.
        if parent is not None:
            try:
                started = False
                for el in parent.descendants:
                    if not started:
                        if el is text_node:
                            started = True
                        continue
                    if getattr(el, "name", None) == "table":
                        return el
            except Exception:
                return None

        return None

    def phrase_to_pattern(phrase: str) -> re.Pattern[str]:
        # Treat spaces inside the phrase as "any whitespace" so we can match
        # phrases broken across lines/extra spaces in SEC HTML.
        tokens = str(phrase).split()
        if not tokens:
            return re.compile(r"$^")
        escaped = [re.escape(t) for t in tokens]
        # Use \s+ between tokens; do not anchor with word boundaries because
        # SEC HTML frequently concatenates words in the raw text nodes.
        return re.compile(r"(?i)" + r"\s+".join(escaped))

    def table_has_keywords(table: Any, keywords: list[str]) -> bool:
        try:
            t = table.get_text(" ", strip=True).lower()
            t = re.sub(r"\s+", "", t)
        except Exception:
            return False
        for k in keywords:
            k_norm = re.sub(r"\s+", "", str(k).lower())
            if k_norm and k_norm in t:
                return True
        return False

    def table_keyword_hit_count(table: Any, keywords: list[str]) -> int:
        try:
            t = table.get_text(" ", strip=True).lower()
            t = re.sub(r"\s+", "", t)
        except Exception:
            return 0
        hits = 0
        for k in keywords:
            k_norm = re.sub(r"\s+", "", str(k).lower())
            if k_norm and k_norm in t:
                hits += 1
        return hits

    def table_has_excluded_keywords(table: Any, excluded: list[str]) -> bool:
        return table_keyword_hit_count(table, excluded) > 0

    # iXBRL-friendly heading discovery: statement headings are often embedded in
    # block-level elements (div/p/span/h1-h6) rather than exposed as bare text
    # nodes. We therefore scan "heading-like" elements, then walk forward to
    # the next <table>.
    #
    # Pass 1 (heading scan) is candidate generation only:
    # - it identifies which headings likely correspond to statement sections
    # Pass 2 (next-table walk) is unchanged DOM traversal:
    # - from each matched heading element, find the following <table>.
    heading_like_tags = {"p", "div", "span", "h1", "h2", "h3", "h4", "h5", "h6"}
    heading_like_elements: list[Any] = []
    seen_heading_ids: set[int] = set()

    def _maybe_add_heading(el: Any) -> None:
        el_id = id(el)
        if el_id in seen_heading_ids:
            return
        seen_heading_ids.add(el_id)
        heading_like_elements.append(el)

    for el in soup.find_all(list(heading_like_tags)):
        _maybe_add_heading(el)

    # Also include elements whose id/class suggest headings/titles.
    heading_attr_keywords = ("header", "heading", "title", "statement")
    for el in soup.find_all(True):
        try:
            id_attr = str(el.get("id") or "").lower()
            class_attr = " ".join([str(x) for x in (el.get("class") or [])]).lower()
        except Exception:
            continue
        if any(k in id_attr or k in class_attr for k in heading_attr_keywords):
            _maybe_add_heading(el)

    # Precompute normalized full text for each candidate heading element.
    # For iXBRL, words are sometimes split across spans with whitespace inserted
    # mid-word; we therefore keep both a whitespace-normalized version and a
    # no-whitespace version for matching.
    heading_scan: list[tuple[Any, str, str]] = []
    for el in heading_like_elements:
        try:
            raw = el.get_text(" ", strip=True)
            norm = re.sub(r"\s+", " ", raw).strip().lower()
            no_ws = re.sub(r"\s+", "", norm)
        except Exception:
            continue
        if norm:
            heading_scan.append((el, norm, no_ws))

    for statement_type in STATEMENT_PHRASE_TIERS.keys():
        keywords = statement_keywords.get(statement_type, [])
        # Phrase-matched candidates (passing content gate), deduped by underlying table identity.
        #
        # Candidate tuple:
        # (table, row_count, has_kw, tier_rank, numeric_density_count, discovery_index)
        candidates: list[tuple[Any, int, bool, int, int, int]] = []
        seen_phrase_table_ids: set[int] = set()
        discovery_index = 0
        best: Optional[tuple[int, int, int, Any]] = None  # (hits, tier_rank, row_count, table) for full scan

        # Debug-only visibility into phrase-matching candidates and winner choice.
        debug_phrase_candidates: list[dict[str, Any]] = [] if debug else []
        debug_rejected_phrase_candidates: list[dict[str, Any]] = [] if debug else []
        phrase_candidates_before_content_check = 0
        winner_table: Any | None = None
        winner_reason: str | None = None
        no_table_reason: str | None = None
        full_scan_reason_hint: str | None = None
        seeded_best_table: Any | None = None

        for tier_name in ("tier1", "tier2", "tier3"):
            tier_phrases = STATEMENT_PHRASE_TIERS[statement_type].get(tier_name) or []
            tier_rank = {"tier1": 1, "tier2": 2, "tier3": 3}[tier_name]

            for phrase in tier_phrases:
                pattern = phrase_to_pattern(phrase)
                phrase_no_ws = re.sub(r"\s+", "", str(phrase).strip().lower())

                for heading_el, heading_text, heading_text_no_ws in heading_scan:
                    # Primary match uses phrase_to_pattern on normalized text.
                    # Secondary match removes ALL whitespace to handle iXBRL
                    # mid-word splitting (e.g. "STATEM" + "ENTS").
                    if not pattern.search(heading_text) and phrase_no_ws not in heading_text_no_ws:
                        continue

                    # If the phrase matched on an inner fragment (e.g. a <span>
                    # inside the real section heading), walk from the nearest
                    # block-level heading container to reduce false/missed
                    # table matches.
                    heading_walk_tags = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6"}
                    start_node = heading_el
                    if getattr(heading_el, "name", None) not in heading_walk_tags:
                        start_node = heading_el.find_parent(list(heading_walk_tags)) or heading_el

                    candidate = next_table_after(start_node)
                    if candidate is None:
                        continue

                    row_count = len(candidate.find_all("tr"))
                    if row_count <= 5:
                        continue
                    if not _table_passes_numeric_gate(candidate):
                        continue
                    validation_terms = statement_validation_terms.get(statement_type) or []
                    if validation_terms and not table_contains_any_terms(candidate, validation_terms):
                        continue
                    phrase_candidates_before_content_check += 1
                    rejection_rule = _table_rejection_reason(candidate, statement_type)
                    if rejection_rule is not None:
                        if (
                            statement_type == "balance_sheet"
                            and rejection_rule
                            in (
                                "B1_TOTAL_ASSETS_MISSING",
                                "B1_LIABILITIES_ANCHOR_MISSING",
                            )
                        ):
                            _tid = id(candidate)
                            if _tid not in seen_b1_reject_ids:
                                seen_b1_reject_ids.add(_tid)
                                balance_sheet_b1_rejects.append(
                                    {
                                        "table": candidate,
                                        "numeric_density_count": _count_numeric_cells(
                                            candidate
                                        ),
                                        "rejection_rule": rejection_rule,
                                    }
                                )
                        if debug:
                            debug_rejected_phrase_candidates.append(
                                {
                                    "phrase": phrase,
                                    "phrase_tier": tier_name,
                                    "row_count": row_count,
                                    "numeric_density_count": _count_numeric_cells(candidate),
                                    "keyword_hit_count": table_keyword_hit_count(candidate, keywords),
                                    "content_check_pass": False,
                                    "rejection_rule": rejection_rule,
                                    "data_column_count": _data_column_count(candidate),
                                    "leftmost_row_labels": _leftmost_nonempty_row_labels(candidate, 8),
                                    "has_kw": table_has_keywords(candidate, keywords),
                                    "table": candidate,
                                }
                            )
                        continue
                    has_kw = table_has_keywords(candidate, keywords)
                    excluded = statement_exclude_keywords.get(statement_type) or []
                    if has_kw and excluded and table_has_excluded_keywords(candidate, excluded):
                        has_kw = False
                    tid = id(candidate)
                    if tid in seen_phrase_table_ids:
                        continue
                    seen_phrase_table_ids.add(tid)

                    numeric_density_count = _count_numeric_cells(candidate)
                    candidates.append(
                        (candidate, row_count, has_kw, tier_rank, numeric_density_count, discovery_index)
                    )
                    discovery_index += 1

                    if debug:
                        debug_phrase_candidates.append(
                            {
                                "phrase": phrase,
                                "phrase_tier": tier_name,
                                "row_count": row_count,
                                "numeric_density_count": numeric_density_count,
                                "keyword_hit_count": table_keyword_hit_count(candidate, keywords),
                                "content_check_pass": True,
                                "rejection_rule": None,
                                "leftmost_row_labels": _leftmost_nonempty_row_labels(candidate, 8),
                                "has_kw": has_kw,
                                "table": candidate,
                            }
                        )

                    # Avoid scanning the entire doc once we've found some plausible matches.
                    if len(candidates) >= 25:
                        break

                if len(candidates) >= 25:
                    break

            # Lower tiers are fallback-only: only try them if this tier generated
            # zero candidates.
            if candidates:
                break

        if not candidates:
            # Keyword-only fallback (only when phrase matching produced no candidates).
            print(f"WARNING: Phrase match failed for {statement_type} — using keyword-only fallback")

            fallback_table: Any | None = None
            # (hits, tier_rank, row_count, table) where tier_rank is unknown in fallback.
            tier_rank_fallback = 99
            best_fallback: Optional[tuple[int, int, int, Any]] = None
            for table in soup.find_all("table"):
                row_count = len(table.find_all("tr"))
                if row_count <= 5:
                    continue
                if not _table_passes_numeric_gate(table):
                    continue

                validation_terms = statement_validation_terms.get(statement_type) or []
                if validation_terms and not table_contains_any_terms(table, validation_terms):
                    continue
                if not _table_passes_content_check(table, statement_type):
                    continue

                if statement_type == "cash_flow":
                    # Strict cash flow requirement: must contain all three sections.
                    required_all = ["operating activities", "investing activities", "financing activities"]
                    if not all(table_contains_any_terms(table, [t]) for t in required_all):
                        continue
                hits = table_keyword_hit_count(table, keywords)
                excluded = statement_exclude_keywords.get(statement_type) or []
                if hits > 0 and excluded and table_has_excluded_keywords(table, excluded):
                    hits = 0
                if hits <= 0:
                    continue

                if best_fallback is None:
                    best_fallback = (hits, tier_rank_fallback, row_count, table)
                else:
                    best_hits, _best_tier_rank, best_rows, _best_table = best_fallback
                    # Tier rank is constant (99) in fallback; tie-break reduces to row_count.
                    if hits > best_hits or (hits == best_hits and row_count > best_rows):
                        best_fallback = (hits, tier_rank_fallback, row_count, table)

            if best_fallback is not None:
                fallback_table = best_fallback[3]

            if fallback_table is None:
                print(
                    f"WARNING: Could not identify {statement_type} table for {ticker} — manual inspection required"
                )
                if debug:
                    no_table_reason = "keyword-only fallback found no passing tables"
                results[statement_type] = None
                if debug:
                    _finalize_debug_for_statement(
                        statement_type,
                        debug_phrase_candidates,
                        phrase_candidates_before_content_check,
                        debug_rejected_phrase_candidates,
                        None,
                        winner_reason,
                        no_table_reason,
                    )
            else:
                if debug:
                    winner_table = fallback_table
                    winner_reason = (
                        "keyword-only fallback (highest keyword hit count; then tier rank=99; then row count)"
                    )
                results[statement_type] = fallback_table
                if debug:
                    _finalize_debug_for_statement(
                        statement_type,
                        debug_phrase_candidates,
                        phrase_candidates_before_content_check,
                        debug_rejected_phrase_candidates,
                        winner_table,
                        winner_reason,
                        no_table_reason,
                    )
            continue

        # Phrase candidates always win: if any phrase-matched candidates pass the content gate,
        # select from that pool and do NOT fall through to full-scan.
        if candidates:
            required = statement_required_keywords.get(statement_type) or []
            phrase_pool = candidates
            if required:
                satisfying = []
                for (t, rc, hkw, trk, ndc, di) in phrase_pool:
                    if table_keyword_hit_count(t, required) >= len(required):
                        satisfying.append((t, rc, hkw, trk, ndc, di))
                if satisfying:
                    phrase_pool = satisfying

            # Tie-break among phrase candidates:
            # 1) highest keyword hit count
            # 2) lowest phrase tier rank (tier1=1 highest confidence)
            # 3) highest numeric density
            # 4) highest row count
            # 5) earliest discovery order
            # Use explicit key to avoid confusion: build full score tuple.
            def _phrase_score(c: tuple[Any, int, bool, int, int, int]) -> tuple[int, int, int, int, int]:
                t, row_count, _has_kw, tier_rank, numeric_density_count, disc_idx = c
                return (
                    table_keyword_hit_count(t, keywords),
                    -tier_rank,  # lower rank is better
                    numeric_density_count,
                    row_count,
                    -disc_idx,  # earlier is better
                )

            phrase_pool_sorted = sorted(phrase_pool, key=_phrase_score, reverse=True)
            best_table = phrase_pool_sorted[0][0]
            if debug:
                winner_table = best_table
                winner_reason = (
                    "phrase pool winner (keyword hits; then tier; then numeric density; then row count; then doc order)"
                )
                _finalize_debug_for_statement(
                    statement_type,
                    debug_phrase_candidates,
                    phrase_candidates_before_content_check,
                    debug_rejected_phrase_candidates,
                    winner_table,
                    winner_reason,
                    no_table_reason,
                )
            results[statement_type] = best_table
            continue

        # If phrase-walk doesn't produce a keyword-bearing table (common with some
        # SEC HTML where headings are split across multiple nodes), fall back to
        # scanning all tables for the first plausible keyword-bearing one.
        if debug and full_scan_reason_hint is None:
            full_scan_reason_hint = "no passing phrase candidates"
        for table in soup.find_all("table"):
            row_count = len(table.find_all("tr"))
            if row_count <= 5:
                continue
            if not _table_passes_numeric_gate(table):
                continue
            validation_terms = statement_validation_terms.get(statement_type) or []
            if validation_terms and not table_contains_any_terms(table, validation_terms):
                continue
            if not _table_passes_content_check(table, statement_type):
                continue
            hits = table_keyword_hit_count(table, keywords)
            excluded = statement_exclude_keywords.get(statement_type) or []
            if hits > 0 and excluded and table_has_excluded_keywords(table, excluded):
                hits = 0
            if hits <= 0:
                continue

            # Prefer tables that match more keywords; tie-break on tier rank then row count.
            tier_rank_full_scan = 99
            if best is None:
                best = (hits, tier_rank_full_scan, row_count, table)
            else:
                best_hits, best_tier_rank, best_rows, _best_table = best
                if (
                    hits > best_hits
                    or (hits == best_hits and tier_rank_full_scan < best_tier_rank)
                    or (
                        hits == best_hits
                        and tier_rank_full_scan == best_tier_rank
                        and row_count > best_rows
                    )
                ):
                    best = (hits, tier_rank_full_scan, row_count, table)

        if best is not None:
            if debug:
                winner_table = best[3]
                if seeded_best_table is not None and best[3] is seeded_best_table:
                    winner_reason = (
                        "income_statement: seeded from phrase-matched candidate; full scan did not override"
                    )
                else:
                    hint = full_scan_reason_hint or "phrase candidates did not yield a deterministic winner"
                    winner_reason = (
                        "full scan winner (highest keyword hit count; then tier rank; then row count); "
                        + hint
                    )
            results[statement_type] = best[3]
        else:
            print(
                f"WARNING: Could not identify {statement_type} table for {ticker} — manual inspection required"
            )
            if debug:
                no_table_reason = full_scan_reason_hint or "no passing candidate found via phrase matching or full scan"
            results[statement_type] = None

        if debug:
            _finalize_debug_for_statement(
                statement_type,
                debug_phrase_candidates,
                phrase_candidates_before_content_check,
                debug_rejected_phrase_candidates,
                winner_table,
                winner_reason,
                no_table_reason,
            )

    if results.get("balance_sheet") is None and balance_sheet_b1_rejects:
        merged_bs = _try_merge_split_balance_sheet(
            balance_sheet_b1_rejects, _count_numeric_cells
        )
        if merged_bs is not None:
            results["split_balance_sheet_merged"] = merged_bs
            print(
                f"WARNING: Balance sheet appears split across two tables for {ticker} — merging assets and liabilities halves"
            )
            if debug:
                bs_dbg = debug_report.setdefault("balance_sheet", {})
                bs_dbg["split_balance_sheet_merged"] = merged_bs
                bs_dbg["winner_reason"] = "split_balance_sheet_merged (B1 pair merge)"

    if debug:
        print("\n" + "=" * 80)
        print("FINANCIAL TABLE IDENTIFICATION DIAGNOSTICS")
        print("=" * 80)
        for st in ("income_statement", "balance_sheet", "cash_flow"):
            rec = debug_report.get(st) or {}
            print(f"\n--- {st} ---")
            before_cnt = rec.get("phrase_candidate_count_before_content_check", 0)
            after_cnt = rec.get("phrase_candidate_count", 0)
            rejected_sample = rec.get("rejected_phrase_candidates") or []
            print(f"Phrase candidates before content gate: {before_cnt}")
            print(f"Phrase candidates after content gate:  {after_cnt}")
            print(f"Rejected phrase candidates sample (up to 5): {min(len(rejected_sample), 5)}")

            if rejected_sample:
                for i, cand in enumerate(rejected_sample[:5], start=1):
                    print(f"\nRejected Candidate {i}:")
                    print(f"  phrase_matched: {cand.get('phrase')!r}")
                    print(f"  phrase_tier: {cand.get('phrase_tier')!r}")
                    print(f"  row_count: {cand.get('row_count')}")
                    print(f"  numeric_density_count: {cand.get('numeric_density_count')}")
                    print(f"  keyword_hit_count: {cand.get('keyword_hit_count')}")
                    print(f"  data_column_count: {cand.get('data_column_count')}")
                    print(f"  rejection_rule: {cand.get('rejection_rule')}")
                    labels = cand.get("leftmost_row_labels") or []
                    print(f"  first_8_row_labels: {labels}")

            phrase_candidates: list[dict[str, Any]] = rec.get("phrase_candidates") or []
            if phrase_candidates:
                for i, cand in enumerate(phrase_candidates[:5], start=1):
                    print(f"\nCandidate {i}:")
                    print(f"  phrase_matched: {cand.get('phrase')!r}")
                    print(f"  row_count: {cand.get('row_count')}")
                    print(f"  numeric_density_count: {cand.get('numeric_density_count')}")
                    print(f"  keyword_hit_count: {cand.get('keyword_hit_count')}")
                    print(f"  content_check_pass: {cand.get('content_check_pass')}")
                    print(f"  rejection_rule: {cand.get('rejection_rule')}")
                    labels = cand.get("leftmost_row_labels") or []
                    print(f"  first_8_row_labels: {labels}")

            winner_table = rec.get("winner_table")
            if winner_table is None:
                if rec.get("split_balance_sheet_merged"):
                    print(
                        f"\nWinner: split_balance_sheet_merged — {rec.get('winner_reason')}"
                    )
                else:
                    reason = rec.get("no_table_reason") or "NO TABLE SELECTED"
                    print(f"\nWinner: NO TABLE SELECTED — {reason}")
            else:
                keywords = statement_keywords.get(st, []) or []
                winner_row_count = len(winner_table.find_all("tr"))
                winner_numeric_density = _count_numeric_cells(winner_table)
                winner_kw_hits = table_keyword_hit_count(winner_table, keywords)

                phrase_matched = None
                for cand in phrase_candidates:
                    if cand.get("table") is winner_table:
                        phrase_matched = cand.get("phrase")
                        break
                if phrase_matched is None:
                    phrase_matched = "(from full scan or keyword-only fallback)"

                print("\nWinner:")
                print(f"  phrase_matched: {phrase_matched!r}")
                print(f"  row_count: {winner_row_count}")
                print(f"  numeric_density_count: {winner_numeric_density}")
                print(f"  keyword_hit_count: {winner_kw_hits}")
                print(f"  reason: {rec.get('winner_reason')}")

    return results


def table_to_text(table: Any) -> str:
    """
    Convert an HTML <table> element to clean text:
    - each <tr> becomes a single line
    - join cell values with ' | '
    - clean each cell value by stripping whitespace, '$', and parentheses
    - convert negative parentheses numbers: (1,234) -> -1234
    """
    if table is None:
        return ""
    lines: list[str] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        cell_vals: list[str] = []
        for c in cells:
            txt = c.get_text(" ", strip=True)
            cell_vals.append(_clean_cell_value(txt))
        lines.append(" | ".join(cell_vals))
    return "\n".join(lines).strip()


def _try_merge_split_balance_sheet(
    rejects: list[dict[str, Any]],
    count_numeric_cells: Any,
) -> dict[str, Any] | None:
    """
    Pair phrase-matched balance-sheet tables rejected by B1 into one merged text
    (assets half + liabilities half) for LLM extraction.
    """
    if not rejects:
        return None

    def _t_ns(table: Any) -> str:
        try:
            text_raw = table.get_text(" ", strip=True).lower()
        except Exception:
            return ""
        return re.sub(r"\s+", "", text_raw)

    def is_assets_half(table: Any) -> bool:
        t = _t_ns(table)
        if "totalliabilities" in t:
            return False
        return "totalassets" in t or "totalcurrentassets" in t

    def is_liabilities_half(table: Any) -> bool:
        t = _t_ns(table)
        if "totalassets" in t:
            return False
        return "totalliabilities" in t or "totalcurrentliabilities" in t

    assets_tables: list[tuple[Any, int]] = []
    liab_tables: list[tuple[Any, int]] = []
    seen_a: set[int] = set()
    seen_l: set[int] = set()

    for r in rejects:
        tbl = r.get("table")
        if tbl is None:
            continue
        dens = int(count_numeric_cells(tbl))
        tid = id(tbl)
        if is_assets_half(tbl) and tid not in seen_a:
            seen_a.add(tid)
            assets_tables.append((tbl, dens))
        elif is_liabilities_half(tbl) and tid not in seen_l:
            seen_l.add(tid)
            liab_tables.append((tbl, dens))

    pairs: list[tuple[int, int, int, Any, Any]] = []
    for a_tbl, a_d in assets_tables:
        for l_tbl, l_d in liab_tables:
            if a_tbl is l_tbl:
                continue
            comb = a_d + l_d
            pairs.append((comb, a_d, l_d, a_tbl, l_tbl))

    if not pairs:
        return None

    best = max(
        pairs,
        key=lambda p: (p[0], p[1], p[2], id(p[3]), id(p[4])),
    )
    comb, _a_d, _l_d, assets_table, liabilities_table = best
    merged_text = "\n".join(
        [table_to_text(assets_table), table_to_text(liabilities_table)]
    ).strip()
    return {
        "kind": "split_balance_sheet_merged",
        "merged_text": merged_text,
        "assets_table": assets_table,
        "liabilities_table": liabilities_table,
        "combined_numeric_density": comb,
    }


def _parse_json_from_text(raw_text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from the model output.

    The prompt requests JSON-only, but this parser is defensive: it extracts the
    first JSON object substring if the model includes surrounding whitespace.
    """
    raw_text = raw_text.strip()
    # Fast path: direct JSON.
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Defensive fallback: locate the first {...} blob.
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    obj = json.loads(raw_text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON was not an object.")
    return obj


def _coerce_numeric(value: Any) -> Optional[float]:
    """
    Coerce numeric-ish strings into float.
    Returns None for null/empty.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in {"null", ""}:
            return None
        # Remove common formatting characters.
        s = s.replace("$", "").replace(",", "")
        # Handle parentheses negatives: (123) -> -123
        if re.fullmatch(r"\(\s*[-+]?[0-9]*\.?[0-9]+\s*\)", s):
            inner = s.strip()[1:-1].strip()
            if inner.startswith("-"):
                inner = inner[1:]
            return -float(inner)
        try:
            return float(s)
        except Exception:
            return None
    return None


def normalize_financial_data(financial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize all spec'd numeric fields to millions based on reporting_unit.

    The LLM response may be nested by statement (e.g. `income_statement`,
    `balance_sheet`, `cash_flow_statement`), so we scale recursively.
    """
    reporting_unit = str(financial_data.get("reporting_unit") or "").strip().lower()

    if reporting_unit == "millions" or reporting_unit == "millions_inferred":
        scale = 1.0
    elif reporting_unit == "thousands" or reporting_unit == "thousands_inferred":
        scale = 1.0 / 1000.0
    elif reporting_unit == "billions":
        scale = 1000.0
    elif reporting_unit == "units":
        scale = 1.0 / 1_000_000.0
    else:
        # Unknown unit: keep as-is; downstream will likely treat as missing.
        return financial_data

    numeric_fields: List[str] = [
        "revenues",
        "gross_profit",
        "operating_income_loss",
        "net_income_loss",
        "interest_expense",
        "assets",
        "assets_current",
        "liabilities_current",
        "total_liabilities",
        "long_term_debt",
        "stockholders_equity",
        "cash_and_cash_equivalents",
        "net_cash_from_operating_activities",
        "net_cash_from_investing_activities",
        "net_cash_from_financing_activities",
        "capital_expenditures",
    ]

    def _scale_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in numeric_fields:
                    num = _coerce_numeric(v)
                    out[k] = None if num is None else (num * scale)
                else:
                    out[k] = _scale_obj(v)
            return out
        if isinstance(obj, list):
            return [_scale_obj(x) for x in obj]
        return obj

    return _scale_obj(financial_data)


def validate_extraction(financial_data: Dict[str, Any], ticker: str) -> None:
    """
    Run sanity checks on extracted financial data and print a PASS/FAIL report.
    """
    income = financial_data.get("income_statement") or {}
    balance = financial_data.get("balance_sheet") or {}
    cash = financial_data.get("cash_flow_statement") or {}

    revenues = income.get("revenues")
    gross_profit = income.get("gross_profit")
    operating_income_loss = income.get("operating_income_loss")
    net_income_loss = income.get("net_income_loss")
    assets = balance.get("assets")
    assets_current = balance.get("assets_current")
    liabilities_current = balance.get("liabilities_current")
    total_liabilities = balance.get("total_liabilities")
    stockholders_equity = balance.get("stockholders_equity")
    net_cash_from_operating_activities = cash.get("net_cash_from_operating_activities")
    capital_expenditures = cash.get("capital_expenditures")
    fiscal_year = income.get("fiscal_year")

    checks: List[tuple[bool, str]] = []

    # 1) revenues is not null and greater than 0
    checks.append((revenues is not None and revenues > 0, "revenues is null or <= 0"))

    # 2) gross_profit is not null and greater than 0
    checks.append(
        (gross_profit is not None and gross_profit > 0, "gross_profit is null or <= 0")
    )

    # 3) gross_margin between 5% and 98%
    gross_margin_pct: Optional[float] = None
    if revenues is not None and gross_profit is not None and revenues != 0:
        gross_margin_pct = (gross_profit / revenues) * 100.0
    if gross_margin_pct is None:
        checks.append((False, "gross_margin cannot be computed"))
    else:
        passed = 5.0 <= gross_margin_pct <= 98.0
        checks.append((passed, f"gross_margin out of range: {gross_margin_pct:.1f}%"))

    # 4) operating_income_loss is not null
    checks.append((operating_income_loss is not None, "operating_income_loss is null"))

    # 5) net_income_loss is not null
    checks.append((net_income_loss is not None, "net_income_loss is null"))

    # 6) assets not null and greater than assets_current (if both present)
    if assets is None:
        checks.append((False, "assets is null"))
    elif assets_current is not None:
        passed = assets > assets_current
        checks.append(
            (
                passed,
                f"assets ({assets}) is not greater than assets_current ({assets_current})",
            )
        )
    else:
        checks.append((True, "assets_current missing (skipped assets > assets_current)"))

    # 7) total_liabilities not null
    checks.append((total_liabilities is not None, "total_liabilities is null"))

    # 8) stockholders_equity not null
    checks.append(
        (stockholders_equity is not None, "stockholders_equity is null")
    )

    # 9) net_cash_from_operating_activities not null
    checks.append(
        (net_cash_from_operating_activities is not None, "net_cash_from_operating_activities is null")
    )

    # 10) capital_expenditures not null and >= 0
    checks.append(
        (
            capital_expenditures is not None and capital_expenditures >= 0,
            "capital_expenditures is null or < 0",
        )
    )

    # 11) fiscal_year not null and between 2020 and 2026
    passed_year = (
        fiscal_year is not None
        and isinstance(fiscal_year, (int, float))
        and 2020 <= int(fiscal_year) <= 2026
    )
    checks.append((passed_year, "fiscal_year is null or out of range (2020-2026)"))

    passed_count = 0
    total = len(checks)
    for passed, reason in checks:
        if passed:
            print("PASS")
            passed_count += 1
        else:
            print(f"FAIL — {reason}")

    print(f"{passed_count}/{total} checks passed")


def extract_financials_with_llm(
    income_text: str,
    balance_text: str,
    cashflow_text: str,
    ticker: str,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Use OpenAI to extract structured financial fields from the statement tables.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment/.env.")

    client = OpenAI(api_key=api_key)

    def _detect_reporting_unit(table_text: str) -> Optional[str]:
        """
        Detect reporting unit from SEC table text (whitespace-insensitive).
        Returns: 'millions' | 'thousands' | 'billions' | None
        """
        t = (table_text or "").lower()
        t_ns = re.sub(r"\s+", "", t)

        # Precedence: explicit billions > millions > thousands.
        if (
            "inbillions" in t_ns
            or "amountsinbillions" in t_ns
            or "dollarsinbillions" in t_ns
        ):
            return "billions"
        if (
            "inmillions" in t_ns
            or "amountsinmillions" in t_ns
            or "dollarsinmillions" in t_ns
        ):
            return "millions"
        if "inthousands" in t_ns or "amountsinthousands" in t_ns:
            return "thousands"
        return None

    combined_text = "\n".join([income_text or "", balance_text or "", cashflow_text or ""])
    detected_unit = _detect_reporting_unit(combined_text)

    if detected_unit:
        reporting_unit_line = (
            f"The tables state values are reported in {detected_unit}. "
            "Use this as the reporting_unit value — do not infer it yourself."
        )
    else:
        reporting_unit_line = (
            "The tables do not explicitly state the unit. Infer it, but "
            "flag uncertainty by returning reporting_unit as 'millions_inferred' "
            "or 'thousands_inferred'."
        )

    prompt = f"""
You are extracting financial data from the most recent fiscal year only.
Tables show multiple years side by side — always use the leftmost data column which is the most recent year.

You MUST follow the instructions exactly:
1) All values must be in the original reported currency units as they appear in the filing.
   Do not convert or scale.
   If values are in millions, return them in millions.
   If in thousands, return in thousands.
   If values are in billions, return them in billions.
   State the reporting unit in a field called reporting_unit with value 'millions', 'thousands', 'billions', 'millions_inferred', or 'thousands_inferred'.
   {reporting_unit_line}
2) Return null for any field you cannot find with confidence.
   Do not guess or derive values unless the field is genuinely calculable from other extracted fields
   (e.g. gross_profit = revenues - cost_of_revenues if gross profit subtotal is not explicitly shown).
3) Response must be a JSON object only, no markdown fences, no explanation text.

Extract these exact fields grouped by statement:

From income statement:
- revenues: total net revenue, top line before any deductions
- gross_profit: revenue minus cost of revenues/goods sold, the subtotal explicitly labelled gross profit or gross margin
- operating_income_loss: income from operations after all operating expenses
- net_income_loss: bottom line net income attributable to the company
- interest_expense: interest expense on debt, positive number
- fiscal_year: the calendar year of the most recent period (integer, e.g. 2025)

From balance sheet:
- assets: total assets, the final total on the asset side
- assets_current: total current assets subtotal
- liabilities_current: total current liabilities subtotal
- total_liabilities: the subtotal of ALL liabilities before shareholders equity — do NOT use the final reconciling total that equals total assets. On a balance sheet, total liabilities is always less than total assets. If the value you find equals total assets, you have the wrong line — look for the liabilities subtotal above the equity section.
- If there is no explicit 'Total liabilities' subtotal line, compute total_liabilities as the sum of ALL liability line items shown before the shareholders equity section (including both current and non-current liabilities, such as long-term debt, operating/finance lease obligations, deferred income taxes and other liabilities, and redeemable noncontrolling interest if presented).
 - Do NOT return total_liabilities equal to liabilities_current; total_liabilities must include non-current liabilities too.
 - If total_liabilities cannot be found directly, derive it as assets - stockholders_equity (this should be less than total assets and represents all liabilities before equity).
- long_term_debt: long-term debt or notes payable due after one year, excluding current portion
- stockholders_equity: total shareholders equity or stockholders equity, the final equity total
- cash_and_cash_equivalents: cash and cash equivalents line, first line of current assets

From cash flow statement:
- net_cash_from_operating_activities: net cash provided by operating activities, the operating section subtotal
- net_cash_from_investing_activities: net cash used in investing activities, the investing section subtotal
- net_cash_from_financing_activities: net cash used in financing activities, the financing section subtotal
- capital_expenditures: purchases of property plant and equipment or capital expenditures, always return as a positive number regardless of how it appears in the filing

Sign conventions for cash flow fields — apply these strictly:
- net_cash_from_operating_activities: always positive when 
the company generated cash from operations, negative if cash 
was consumed
- net_cash_from_investing_activities: almost always negative 
(cash spent on investments/capex) — return as negative unless 
the company was a net seller of assets
- net_cash_from_financing_activities: return as negative when 
net cash was used (e.g. debt repayments, buybacks exceed 
proceeds), positive when net cash was received
- capital_expenditures: always return as a positive number 
regardless of how it appears in the filing

If the table shows these as positive numbers under a section 
labelled 'used in' or 'investing' or 'financing', apply the 
correct sign based on the economic meaning, not the raw 
presentation.

If you cannot confidently locate a value, set it to null.

TICKER: {ticker}

INCOME_STATEMENT_TEXT:
{income_text}

BALANCE_SHEET_TEXT:
{balance_text}

CASH_FLOW_TEXT:
{cashflow_text}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        # Encourage strict JSON; still parse defensively.
        response_format={"type": "json_object"},
    )

    raw_response_text = resp.choices[0].message.content or ""
    extracted = _parse_json_from_text(raw_response_text)

    normalized = normalize_financial_data(extracted)

    # Post-extraction self-correction loop.
    income_norm = normalized.get("income_statement") or {}
    balance_norm = normalized.get("balance_sheet") or {}
    cash_norm = normalized.get("cash_flow_statement") or {}

    assets = balance_norm.get("assets")
    total_liabilities = balance_norm.get("total_liabilities")
    revenues = income_norm.get("revenues")
    gross_profit = income_norm.get("gross_profit")
    assets_current = balance_norm.get("assets_current")
    capital_expenditures = cash_norm.get("capital_expenditures")

    corrections_needed: list[str] = []

    # (a) total_liabilities must be less than assets.
    if (
        assets is not None
        and total_liabilities is not None
        and isinstance(assets, (int, float))
        and isinstance(total_liabilities, (int, float))
        and total_liabilities >= assets
    ):
        corrections_needed.append(
            "total_liabilities equals or exceeds total_assets — you grabbed the wrong line. "
            "Total liabilities is the subtotal of liabilities only, before equity, and is always less than total assets."
        )

    # (b) gross_margin plausibility.
    if (
        revenues is not None
        and gross_profit is not None
        and isinstance(revenues, (int, float))
        and isinstance(gross_profit, (int, float))
        and revenues > 0
    ):
        gm_pct = (gross_profit / revenues) * 100.0
        if not (5.0 <= gm_pct <= 98.0):
            corrections_needed.append(
                f"gross_profit appears incorrect — computed gross margin is {gm_pct:.1f}% which is outside plausible range. "
                "Re-read the income statement and find the gross profit or gross margin subtotal line explicitly."
            )

    # (c) assets_current missing but assets present.
    if assets is not None and assets_current is None:
        corrections_needed.append(
            "assets_current is missing — find the total current assets subtotal on the balance sheet."
        )

    # (d) capital_expenditures missing.
    if capital_expenditures is None:
        corrections_needed.append(
            "capital_expenditures is missing — look for purchases of property plant and equipment or capital expenditures in the investing activities section of the cash flow statement."
        )

    if not corrections_needed:
        return raw_response_text, extracted, normalized

    # Second-pass correction request.
    corrections_prompt = f"""
You previously extracted financial data from SEC 10-K tables for ticker {ticker}. Some fields appear inconsistent.

Here is your current extracted data (normalized to millions using reporting_unit):
CURRENT_DATA_JSON:
{json.dumps(normalized, indent=2)}

The following fields from your previous extraction need correction. For each issue described, re-read the relevant table carefully and return only the corrected fields as a JSON object. Do not return fields that do not need correction.

Issues:
{chr(10).join(f"- {item}" for item in corrections_needed)}

Return a JSON object that only includes the fields that need correction, using the same nested schema:
- income_statement: corrected fields only
- balance_sheet: corrected fields only
- cash_flow_statement: corrected fields only
Do not include fields that do not change.

INCOME_STATEMENT_TEXT:
{income_text}

BALANCE_SHEET_TEXT:
{balance_text}

CASH_FLOW_TEXT:
{cashflow_text}
""".strip()

    resp2 = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": corrections_prompt}],
        response_format={"type": "json_object"},
    )
    raw_correction_text = resp2.choices[0].message.content or ""

    try:
        corrections_raw = _parse_json_from_text(raw_correction_text)
    except ValueError:
        # If correction response is malformed, fall back to the original normalized data.
        return raw_response_text, extracted, normalized

    # Normalize corrections using the same reporting_unit.
    if isinstance(corrections_raw, dict):
        # Ensure corrections carry a reporting_unit for normalization scaling.
        if "reporting_unit" not in corrections_raw:
            corrections_raw["reporting_unit"] = normalized.get("reporting_unit")
        corrections_norm = normalize_financial_data(corrections_raw)
    else:
        corrections_norm = {}

    corrected_fields: list[str] = []

    for section in ("income_statement", "balance_sheet", "cash_flow_statement"):
        corr_section = corrections_norm.get(section)
        if isinstance(corr_section, dict):
            target_section = normalized.setdefault(section, {})
            for key, value in corr_section.items():
                target_section[key] = value
                corrected_fields.append(f"{section}.{key}")

    if corrected_fields:
        print("Correction pass triggered for: " + ", ".join(sorted(set(corrected_fields))))

    return raw_response_text, extracted, normalized


def find_gross_profit_fallback(
    soup: BeautifulSoup,
    revenues: float,
    fiscal_year: Any,
) -> Dict[str, Any]:
    """
    Oracle-specific gross profit fallback:

    Stage 1: cost-of-revenues breakdown tables (derive gross_profit = revenues - cost_of_revenues).
    Stage 2: gross figure near amount heuristic + focused gross-profit extraction.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment/.env.")

    client = OpenAI(api_key=api_key)

    # --- Stage 1: cost of revenues / services / license tables ---
    def table_text_norm(table: Any) -> str:
        try:
            return table.get_text(" ", strip=True).lower()
        except Exception:
            return ""

    cost_candidates: list[Any] = []
    for table in soup.find_all("table"):
        row_count = len(table.find_all("tr"))
        if row_count <= 5:
            continue
        t = table_text_norm(table)
        t_ns = re.sub(r"\s+", "", t)
        # iXBRL labels often arrive concatenated (e.g. "Costofrevenues").
        if "costof" not in t_ns:
            continue
        if any(x in t_ns for x in ["revenue", "services", "license"]):
            cost_candidates.append(table)

    # In some Oracle filings (including ORCL in our current extraction),
    # the primary statements may not literally contain the string "cost of"
    # inside <table> text, even though the income statement's operating-expense
    # components effectively represent cost of revenues. If our strict match
    # yields no candidates, fall back to tables that look like the income
    # statement (total revenues + operating expenses).
    if not cost_candidates:
        for table in soup.find_all("table"):
            row_count = len(table.find_all("tr"))
            if row_count <= 5:
                continue
            t = table_text_norm(table)
            t_ns = re.sub(r"\s+", "", t)
            if ("totalrevenues" in t_ns) and ("operatingexpenses" in t_ns):
                cost_candidates.append(table)
            if len(cost_candidates) >= 3:
                break

    for cand in cost_candidates:
        cand_text = table_to_text(cand)
        prompt = f"""
This is a table from a company's 10-K filing. Find the total cost of revenues or total cost of services for fiscal year {fiscal_year}.
Return only a JSON object:
{{"cost_of_revenues": <number or null>}}.
The value should be in millions.
If there is no explicit subtotal/total line for cost of revenues/services, derive the total by summing the cost-of-revenues/service line items that together represent the full cost section shown in the table.
Return null only if you truly cannot identify the relevant cost components.

TABLE_TEXT:
{cand_text}
""".strip()

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw_response_text = resp.choices[0].message.content or ""
        try:
            extracted = _parse_json_from_text(raw_response_text)
            cost_of_revenues = _coerce_numeric(extracted.get("cost_of_revenues"))
        except ValueError:
            # Model occasionally returns malformed/non-JSON content even with JSON mode.
            # Treat this candidate as unusable and continue searching.
            continue

            if cost_of_revenues is not None:
                # Lightweight diagnostics to understand fallback failures.
                try:
                    gross_profit_candidate = revenues - cost_of_revenues
                    gross_margin_candidate = (
                        (gross_profit_candidate / revenues) * 100.0 if revenues else 0.0
                    )
                    print(
                        f"[gross_profit_fallback:stage1] cost_of_revenues={cost_of_revenues:.3f} gross_profit_candidate={gross_profit_candidate:.3f} gross_margin_pct={gross_margin_candidate:.1f}%"
                    )
                except Exception:
                    pass

        # Validate cost and derive gross profit.
        if cost_of_revenues is not None and cost_of_revenues > 0 and cost_of_revenues < revenues:
            gross_profit = revenues - cost_of_revenues
            # Accept gross margin in the same broad band used by the
            # validate_extraction() sanity checks (5%–98%).
            gross_margin_pct = (gross_profit / revenues) * 100.0
            if 5.0 <= gross_margin_pct <= 98.0:
                return {
                    "gross_profit": gross_profit,
                    "gross_profit_source": "derived_from_cost_fallback",
                }

    # Deterministic ORCL help:
    # In ORCL's statements, there's often no explicit "gross profit" subtotal line,
    # but cost of revenues is represented by the first components under operating
    # expenses. If the LLM-derived cost fallback didn't yield a plausible
    # gross margin, try summing those known cost components directly from the
    # same table.
    for cand in cost_candidates:
        cand_text = table_to_text(cand)
        # Labels as produced by `table_to_text()` cleaning.
        targets = {
            "cloud": "cloudservicesandlicensesupport1",
            "hardware": "hardware1",
            "services": "services1",
            "total_revenues": "totalrevenues",
        }

        def first_number_from_lines_containing(substr: str) -> float | None:
            for line in cand_text.splitlines():
                if substr in line.lower():
                    # `table_to_text()` lines are structured like:
                    # "<label> | <cell> | <cell> | <value> | ..."
                    # Avoid accidentally capturing digits embedded in the label
                    # (e.g. "services1") by only parsing numeric cells after the first '|'.
                    parts = [p.strip() for p in line.split("|")]
                    for part in parts[1:]:
                        m = re.search(r"-?[0-9][0-9,]*\.?[0-9]*", part)
                        if m:
                            return float(m.group(0).replace(",", ""))
                    return None
            return None

        cloud_cost = first_number_from_lines_containing(targets["cloud"])
        hardware_cost = first_number_from_lines_containing(targets["hardware"])
        services_cost = first_number_from_lines_containing(targets["services"])
        total_revenues = first_number_from_lines_containing(targets["total_revenues"])

        if (
            cloud_cost is not None
            and hardware_cost is not None
            and services_cost is not None
            and total_revenues is not None
        ):
            cost_of_revenues = cloud_cost + hardware_cost + services_cost
            if 0 < cost_of_revenues < revenues:
                gross_profit = revenues - cost_of_revenues
                gross_margin_pct = (gross_profit / revenues) * 100.0 if revenues else 0.0
                if 5.0 <= gross_margin_pct <= 98.0:
                    return {
                        "gross_profit": gross_profit,
                        "gross_profit_source": "derived_from_cost_fallback",
                    }
        # Keep scanning other candidate tables; we only return on success.

    # --- Stage 2: gross figure table near revenues range ---
    # Search for a table containing the word "gross" near a number within [0.2*revenues, 0.9*revenues].
    low = 0.2 * revenues
    high = 0.9 * revenues

    gross_best: Any | None = None

    def parse_window_numbers(window: str) -> list[float]:
        nums: list[float] = []
        # Capture numbers with commas/decimals, optionally wrapped in parentheses.
        for m in re.finditer(r"\(?-?[0-9][0-9,]*\.?[0-9]*\)?", window):
            s = m.group(0)
            s = s.strip()
            if not s:
                continue
            neg = False
            if s.startswith("(") and s.endswith(")"):
                neg = True
                s = s[1:-1].strip()
            s = s.replace(",", "")
            try:
                v = float(s)
                if neg:
                    v = -v
                nums.append(v)
            except Exception:
                continue
        return nums

    for table in soup.find_all("table"):
        row_count = len(table.find_all("tr"))
        if row_count <= 5:
            continue
        t = table_text_norm(table)
        if "gross" not in t:
            continue

        # Find a "gross" occurrence and check surrounding windows for candidate amounts.
        for m in re.finditer(r"gross", t):
            start = max(0, m.start() - 250)
            end = min(len(t), m.end() + 250)
            window = t[start:end]
            nums = parse_window_numbers(window)
            if any((x >= low and x <= high) for x in nums):
                gross_best = table
                break
        if gross_best is not None:
            break

    if gross_best is None:
        return {"gross_profit": None, "gross_profit_source": None}

    gross_text = table_to_text(gross_best)
    prompt = f"""
This is a table from a company's 10-K filing. Find the gross profit figure for fiscal year {fiscal_year}.
Return only a JSON object:
{{"gross_profit": <number or null>}}.
The value should be in millions.
Return null if you cannot find it with confidence.

TABLE_TEXT:
{gross_text}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw_response_text = resp.choices[0].message.content or ""
    try:
        extracted = _parse_json_from_text(raw_response_text)
        gross_profit = _coerce_numeric(extracted.get("gross_profit"))
    except ValueError:
        gross_profit = None

    if gross_profit is not None and gross_profit > 0:
        return {"gross_profit": gross_profit, "gross_profit_source": "derived_from_gross_fallback"}
    return {"gross_profit": None, "gross_profit_source": None}


def main() -> None:
    start_time = time.perf_counter()
    try:
        global NO_CACHE
        # CLI flags:
        # - `--no-cache` bypasses both the SEC HTML cache and the raw LLM JSON cache.
        # - `python3 test_html_parser.py MSFT` sets ticker; keep interactive prompt when omitted.
        args = [str(a).strip() for a in sys.argv[1:] if str(a).strip()]
        NO_CACHE = "--no-cache" in args

        ticker_arg: Optional[str] = None
        for a in args:
            if a == "--no-cache":
                continue
            if a.startswith("-"):
                continue
            ticker_arg = a
            break

        if ticker_arg:
            ticker = ticker_arg.upper()
        else:
            user_ticker = input("Ticker symbol (press Enter for MSFT): ").strip()
            ticker = user_ticker.upper() if user_ticker else "MSFT"

        if not ticker or not ticker.isalnum():
            raise ValueError(
                f"Invalid ticker {ticker!r}. Expected letters/numbers only."
            )

        cik_numeric, cik_padded = step1_find_cik(ticker)

        # Financial services detection: skip standard schema for incompatible SIC ranges.
        submissions = _fetch_submissions_metadata(cik_padded)
        sic = _extract_sic_code(submissions) if submissions else None
        financial_services_mode = False
        if isinstance(sic, int) and (
            6000 <= sic <= 6099
            or 6100 <= sic <= 6199
            or 6200 <= sic <= 6299
            or 6300 <= sic <= 6399
            or 6400 <= sic <= 6411
            or 6500 <= sic <= 6552
            or 6700 <= sic <= 6799
        ):
            financial_services_mode = True

        if financial_services_mode:
            print(
                f"WARNING: {ticker} is a financial services company (SIC {sic}) — standard extraction schema does not apply. "
                "Financial services parsing not yet implemented."
            )
            # Return a normalized-shaped dict with all numeric fields null and a mode flag.
            empty_income = {
                "revenues": None,
                "gross_profit": None,
                "operating_income_loss": None,
                "net_income_loss": None,
                "interest_expense": None,
                "fiscal_year": None,
            }
            empty_balance = {
                "assets": None,
                "assets_current": None,
                "liabilities_current": None,
                "total_liabilities": None,
                "long_term_debt": None,
                "stockholders_equity": None,
                "cash_and_cash_equivalents": None,
            }
            empty_cash = {
                "net_cash_from_operating_activities": None,
                "net_cash_from_investing_activities": None,
                "net_cash_from_financing_activities": None,
                "capital_expenditures": None,
            }
            financial_data = {
                "income_statement": empty_income,
                "balance_sheet": empty_balance,
                "cash_flow_statement": empty_cash,
                "reporting_unit": None,
                "financial_services_mode": True,
            }
            print("=== Normalised financial_data dict ===")
            print(financial_data)
            return
        accession = step2_find_latest_10k_accession(cik_padded)
        primary_url = step3_find_primary_document_url(cik_numeric, cik_padded, accession)

        # Step 4 — Fetch the HTML document.
        html_text = _fetch_text(primary_url)
        soup = BeautifulSoup(html_text, "lxml")

        tables = find_financial_tables(soup, ticker, debug=True)

        income_text = table_to_text(tables["income_statement"])
        split_merged = tables.get("split_balance_sheet_merged")
        if split_merged:
            balance_text = split_merged.get("merged_text") or ""
        else:
            balance_text = table_to_text(tables["balance_sheet"])
        cash_text = table_to_text(tables["cash_flow"])

        def _print_preview(title: str, table_text: str) -> None:
            print(f"=== {title} (first 50 lines) ===")
            lines = table_text.splitlines()
            preview = "\n".join(lines[:50])
            if preview:
                print(preview)
            else:
                print("(empty)")
            print(f"Character count: {len(table_text)}")
            print()

        _print_preview("INCOME STATEMENT TABLE", income_text)
        _print_preview("BALANCE SHEET TABLE", balance_text)
        _print_preview("CASH FLOW TABLE", cash_text)

        # LLM extraction layer: structured extraction + unit normalization.
        raw_llm_json, extracted_financial_data, normalized_financial_data = extract_financials_with_llm(
            income_text=income_text,
            balance_text=balance_text,
            cashflow_text=cash_text,
            ticker=ticker,
        )

        print("=== RAW LLM JSON RESPONSE ===")
        print(raw_llm_json)
        print()

        # Oracle gross profit fallback: if the primary income statement did not
        # include a gross profit subtotal (common for ORCL), try deriving it
        # from cost-of-revenues tables.
        income = normalized_financial_data.get("income_statement") or {}
        revenues = income.get("revenues")
        gross_profit = income.get("gross_profit")
        fiscal_year = income.get("fiscal_year")
        if revenues is not None and revenues > 0 and (gross_profit is None or gross_profit == 0):
            def _derive_gross_profit_from_income_text(
                table_text: str, revenues_val: float
            ) -> float | None:
                """
                Deterministically derive gross profit by summing cost-of-revenues
                component lines from the selected income statement table text.
                """
                lines = table_text.splitlines()
                start_idx: int | None = None
                end_idx: int | None = None

                for i, line in enumerate(lines):
                    ln_ns = re.sub(r"\s+", "", (line or "").lower())
                    if start_idx is None and (
                        "costofrevenues" in ln_ns or "costoofrevenues" in ln_ns
                    ):
                        start_idx = i
                        continue
                    if start_idx is not None and (
                        "selling,generalandadministrative" in ln_ns
                        or "totaloperatingexpenses" in ln_ns
                    ):
                        end_idx = i
                        break

                if start_idx is None:
                    return None

                if end_idx is None:
                    end_idx = len(lines)

                cost_sum = 0.0
                for j in range(start_idx + 1, end_idx):
                    ln = lines[j]
                    # Sum the first numeric token from each cost-component line.
                    nums = re.findall(r"\(?-?[0-9][0-9,]*\.?[0-9]*\)?", ln or "")
                    if not nums:
                        continue
                    v = _coerce_numeric(nums[0])
                    if v is None:
                        continue
                    cost_sum += float(v)

                if not (0 < cost_sum < revenues_val):
                    return None

                gross_profit_candidate = revenues_val - cost_sum
                if revenues_val <= 0:
                    return None
                gross_margin_pct = (gross_profit_candidate / revenues_val) * 100.0
                if 5.0 <= gross_margin_pct <= 98.0 and gross_profit_candidate > 0:
                    return gross_profit_candidate
                return None

            derived_gp = _derive_gross_profit_from_income_text(income_text, float(revenues))
            if derived_gp is not None:
                income["gross_profit"] = derived_gp
                income["gross_profit_source"] = "derived_from_cost_components_in_income_text"
            else:
                print(
                    f"Gross profit not found in primary income statement — running fallback search for {ticker}"
                )
                fallback = find_gross_profit_fallback(soup, float(revenues), fiscal_year)
                fb_gp = fallback.get("gross_profit")
                if fb_gp is not None:
                    income["gross_profit"] = fb_gp
                    if fallback.get("gross_profit_source"):
                        income["gross_profit_source"] = fallback.get("gross_profit_source")

        print("=== Normalised financial_data dict ===")
        print(normalized_financial_data)
        print()

        # Manual validation check: gross_margin_pct = gross_profit / revenues * 100.
        income = normalized_financial_data.get("income_statement") or {}
        revenues = income.get("revenues")
        gross_profit = income.get("gross_profit")
        if revenues is not None and gross_profit is not None and revenues != 0:
            try:
                gm_pct = (gross_profit / revenues) * 100.0
                print(
                    "Gross margin validation (LLM): "
                    f"{gm_pct:.1f}% | expected ~68.8%"
                )
            except Exception:
                print("Gross margin validation (LLM): (could not compute)")
        else:
            print("Gross margin validation (LLM): (missing revenues or gross_profit)")

        validate_extraction(normalized_financial_data, ticker)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"Total elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

