"""Download Damodaran industry averages (margins + EV/EBITDA) and upsert into ChromaDB."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import hashlib
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from utils.constants import CANONICAL_SECTORS
from utils.sector_utils import normalize_to_canonical_sector

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("pandas is required: pip install pandas", file=sys.stderr)
    raise

# Optional engines for legacy .xls (xlrd) and .xlsx (openpyxl)
try:
    import openpyxl  # noqa: F401
except ImportError:
    pass
try:
    import xlrd  # noqa: F401
except ImportError:
    pass

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

MARGIN_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/margin.xls"
VEBITDA_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/vebitda.xls"

REQUEST_HEADERS = {
    "User-Agent": "financial-analyzer/1.0 (academic-data; contact@example.com)",
    "Accept": "*/*",
}

RAG_DIR = Path(__file__).resolve().parent
CHROMA_DIR = RAG_DIR / "chroma_db"
REGISTRY_PATH = RAG_DIR / "peer_registry.json"
DIAGNOSTIC_PATH = RAG_DIR / "damodaran_diagnostic.txt"
COLLECTION_NAME = "peer_metrics"

SOURCE_TAG = "damodaran"
DATA_TYPE = "industry_average"
TICKER_META = "INDUSTRY_AVG"

# After :func:`clean_industry_name`, exact matches are not stored (aggregate / non-industry rows).
EXCLUDED_DAMODARAN_INDUSTRIES: frozenset[str] = frozenset(
    (
        "Total Market",
        "Total Market without financials",
    )
)

# One GPT result per unique Damodaran label (fallback when heuristics do not match).
_GPT_DAMODARAN_SECTOR_CACHE: Dict[str, str] = {}


def _damodaran_heuristic_to_canonical(damodaran_label: str) -> Optional[str]:
    """Rule-based Damodaran industry name → canonical sector."""

    u = damodaran_label.upper()
    if "SOFTWARE" in u:
        return "Technology Software"
    if any(k in u for k in ("COMPUTER", "ELECTRONIC", "SEMICONDUCTOR", "HARDWARE")):
        return "Technology Hardware"
    if any(k in u for k in ("DRUG", "PHARMA", "BIOTECH")):
        return "Pharma"
    if any(k in u for k in ("BANK", "FINANCIAL", "INSURANCE", "INVESTMENT")):
        return "Financial Services"
    if any(k in u for k in ("RETAIL", "STORE", "SHOP")):
        return "Retail"
    if any(k in u for k in ("OIL", "GAS", "PETROLEUM")):
        return "Energy"
    if "MINING" in u or "METAL" in u or "STEEL" in u or "PRECIOUS METAL" in u:
        return "Metals and Mining"
    if "ENERGY" in u:
        return "Energy"
    if any(k in u for k in ("TELECOM", "WIRELESS", "CABLE")):
        return "Telecommunications"
    if any(k in u for k in ("REAL ESTATE", "REIT", "PROPERTY")):
        return "Real Estate"
    if any(k in u for k in ("FOOD", "BEVERAGE", "TOBACCO", "APPAREL")):
        return "Consumer Goods"
    if "CONSUMER" in u:
        return "Consumer Goods"
    if "HEALTH" in u:
        return "Healthcare"
    return None


def _damodaran_canonical_sector(
    damodaran_industry: str,
    errors: List[str],
) -> str:
    """Resolve Damodaran row label to a canonical sector (heuristic + one GPT call per label)."""

    hit = _damodaran_heuristic_to_canonical(damodaran_industry)
    if hit:
        return hit
    if damodaran_industry in _GPT_DAMODARAN_SECTOR_CACHE:
        return _GPT_DAMODARAN_SECTOR_CACHE[damodaran_industry]

    sector_list = ", ".join(CANONICAL_SECTORS)
    prompt = (
        f"Map this Damodaran industry label to exactly one canonical sector from: {sector_list}. "
        f"Return only the sector name, nothing else.\nIndustry label: {damodaran_industry}"
    )
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        out = normalize_to_canonical_sector(raw, default="Industrials")
        _GPT_DAMODARAN_SECTOR_CACHE[damodaran_industry] = out
        logger.info(
            "Damodaran GPT sector map raw=%r normalized=%s label=%r",
            raw,
            out,
            damodaran_industry,
        )
        return out
    except Exception as exc:
        logger.exception("Damodaran GPT sector mapping failed: %s", exc)
        errors.append(f"Damodaran GPT sector mapping failed: {exc}")
        return "Industrials"


def _read_excel_bytes(content: bytes, label: str) -> pd.DataFrame:
    """Load Damodaran Excel bytes: use sheet *Industry Averages* with header row 8."""

    buf = io.BytesIO(content)
    last_err: Optional[Exception] = None
    for engine in ("xlrd", "openpyxl"):
        try:
            buf.seek(0)
            xl = pd.ExcelFile(buf, engine=engine)
            if "Industry Averages" in xl.sheet_names:
                sheet = "Industry Averages"
            else:
                sheet = xl.sheet_names[-1]
                logger.warning(
                    "%s: no 'Industry Averages' sheet; using last sheet %r",
                    label,
                    sheet,
                )
            buf.seek(0)
            df = pd.read_excel(
                buf,
                engine=engine,
                sheet_name=sheet,
                header=8,
            )
            logger.info(
                "Read %s sheet=%r engine=%s shape=%s",
                label,
                sheet,
                engine,
                df.shape,
            )
            return df
        except Exception as exc:
            last_err = exc
            logger.debug("Read failed for %s engine=%s: %s", label, engine, exc)
    raise RuntimeError(f"Could not parse Excel for {label}: {last_err}")


def _normalize_header(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _find_column(columns: List[Any], must_include: List[str], exclude: Tuple[str, ...] = ()) -> Optional[str]:
    """Return original column name whose normalized form contains all keywords."""
    for col in columns:
        if col is None or (isinstance(col, float) and pd.isna(col)):
            continue
        h = _normalize_header(col)
        if any(ex in h for ex in exclude):
            continue
        if all(k in h for k in must_include):
            return str(col)
    return None


def _col_exact(columns: List[Any], want: str) -> Optional[str]:
    w = want.strip().lower()
    for col in columns:
        if str(col).strip().lower() == w:
            return str(col)
    return None


def _find_industry_column(columns: List[Any]) -> Optional[str]:
    """Pick the best column for industry / sector name."""
    exact = _col_exact(columns, "Industry Name")
    if exact:
        return exact
    candidates = [
        (["industry", "name"], ()),
        (["industry"], ("number", "firms")),
        (["sector"], ()),
        (["category"], ()),
    ]
    for must, exclude in candidates:
        c = _find_column(columns, must, exclude)
        if c:
            return c
    for col in columns:
        if col is None or str(col).lower().startswith("unnamed"):
            continue
        return str(col)
    return str(columns[0]) if columns else None


def _pick_margin_columns(columns: List[Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Damodaran margin.xls: Gross Margin, Net Margin; operating = pre-tax unadjusted operating margin."""
    gross = _col_exact(columns, "Gross Margin") or _find_column(columns, ["gross", "margin"])
    net = _col_exact(columns, "Net Margin") or _find_column(columns, ["net", "margin"])
    operating = (
        _col_exact(columns, "Pre-tax Unadjusted Operating Margin")
        or _col_exact(columns, "After-tax Unadjusted Operating Margin")
        or _find_column(columns, ["operating", "margin"], exclude=("pre-stock", "r&d"))
    )
    return gross, operating, net


def _pick_ev_ebitda_column(columns: List[Any]) -> Optional[str]:
    """Prefer exact *EV/EBITDA* (first occurrence if duplicated for US/Global)."""
    for col in columns:
        if str(col).strip() == "EV/EBITDA":
            return str(col)
    for must in (
        ["ev", "ebitda"],
        ["enterprise", "ebitda"],
    ):
        c = _find_column(columns, must)
        if c:
            return c
    return None


def clean_industry_name(raw: Any) -> str:
    """Strip whitespace; replace slashes with spaces; drop other punctuation; collapse spaces."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip().replace("/", " ")
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        v = val.strip()
        if v == "" or v.lower() in {"nan", "nm", "na"}:
            return None
        v = v.replace(",", "")
        try:
            return float(v)
        except ValueError:
            return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if pd.isna(x):
        return None
    return x


def _format_margin_percent(val: Any) -> Optional[float]:
    """Return margin as a human percent value (e.g. 25.4) or None."""
    x = _to_float(val)
    if x is None:
        return None
    # Ratios often stored as decimals (e.g. 0.254 = 25.4%); already-percent values stay as-is.
    if 0 < abs(x) <= 1.0:
        x = x * 100.0
    return round(x, 4)


def _format_multiple(val: Any) -> Optional[float]:
    x = _to_float(val)
    if x is None:
        return None
    return round(x, 4)


def _stable_doc_id(industry_key: str) -> str:
    h = hashlib.sha256(industry_key.encode("utf-8")).hexdigest()[:24]
    return f"damodaran_{h}"


def _build_chunk(
    industry_name: str,
    gross: Optional[float],
    operating: Optional[float],
    net_margin: Optional[float],
    ev_ebitda: Optional[float],
) -> str:
    parts: List[str] = [f"Industry Average: {industry_name}"]
    if gross is not None:
        parts.append(f"Gross Margin: {gross:.2f}%")
    if operating is not None:
        parts.append(f"Operating Margin: {operating:.2f}%")
    if net_margin is not None:
        parts.append(f"Net Margin: {net_margin:.2f}%")
    if ev_ebitda is not None:
        parts.append(f"EV/EBITDA: {ev_ebitda:.2f}x")
    parts.append("Data Type: Industry Average")
    parts.append("Source: Damodaran NYU 2024")
    return " | ".join(parts)


def download_bytes(url: str) -> bytes:
    logger.info("Downloading %s", url)
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=120)
    resp.raise_for_status()
    logger.info("Downloaded %s bytes from %s", len(resp.content), url)
    return resp.content


def write_diagnostic(
    margin_df: pd.DataFrame,
    vebitda_df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("=== margin.xls (first sheet) ===\n")
    lines.append(f"columns: {list(margin_df.columns)}\n\n")
    lines.append(margin_df.head(5).to_string())
    lines.append("\n\n=== vebitda.xls (first sheet) ===\n")
    lines.append(f"columns: {list(vebitda_df.columns)}\n\n")
    lines.append(vebitda_df.head(5).to_string())
    lines.append("\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote diagnostic to %s", path)


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.exception("Failed to load registry %s: %s", path, exc)
        return {}


def _save_registry(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    logger.info("Saved registry to %s", path)


def _build_vector_store() -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def _delete_previous_damodaran(store: Chroma) -> None:
    try:
        got = store.get(where={"source": SOURCE_TAG})
        ids = got.get("ids") or []
        if ids:
            store.delete(ids=list(ids))
            logger.info("Removed %d prior Damodaran documents from ChromaDB.", len(ids))
    except Exception as exc:
        logger.warning("Could not delete prior Damodaran entries (ok on first run): %s", exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting Damodaran industry averages import.")

    margin_raw = download_bytes(MARGIN_URL)
    vebitda_raw = download_bytes(VEBITDA_URL)

    margin_df = _read_excel_bytes(margin_raw, "margin.xls")
    vebitda_df = _read_excel_bytes(vebitda_raw, "vebitda.xls")

    write_diagnostic(margin_df, vebitda_df, DIAGNOSTIC_PATH)

    m_cols = list(margin_df.columns)
    v_cols = list(vebitda_df.columns)
    ind_m = _find_industry_column(m_cols)
    ind_v = _find_industry_column(v_cols)
    g_col, o_col, n_col = _pick_margin_columns(m_cols)
    ev_col = _pick_ev_ebitda_column(v_cols)

    logger.info(
        "Column map margin: industry=%r gross=%r operating=%r net=%r",
        ind_m,
        g_col,
        o_col,
        n_col,
    )
    logger.info("Column map vebitda: industry=%r ev_ebitda=%r", ind_v, ev_col)

    if not ind_m or not ind_v:
        logger.error("Could not resolve industry columns; see %s", DIAGNOSTIC_PATH)
        print("Summary: processed=0 stored=0 skipped=0 (column detection failed)")
        sys.exit(1)

    # Map EV/EBITDA by cleaned industry name (lowercase key for matching)
    ev_map: Dict[str, Optional[float]] = {}
    for _, row in vebitda_df.iterrows():
        raw_name = row.get(ind_v)
        key = clean_industry_name(raw_name).lower()
        if not key:
            continue
        ev_map[key] = _format_multiple(row.get(ev_col)) if ev_col else None

    store = _build_vector_store()
    _delete_previous_damodaran(store)

    processed = 0
    skipped = 0
    stored_names: List[str] = []
    batch_texts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    batch_ids: List[str] = []
    mapping_errors: List[str] = []

    for _, row in margin_df.iterrows():
        processed += 1
        raw_name = row.get(ind_m)
        industry = clean_industry_name(raw_name)
        if not industry:
            skipped += 1
            logger.debug("Skip row: empty industry after clean: %r", raw_name)
            continue

        if industry in EXCLUDED_DAMODARAN_INDUSTRIES:
            skipped += 1
            logger.debug("Skip excluded Damodaran industry: %r", industry)
            continue

        canonical_sector = _damodaran_canonical_sector(industry, mapping_errors)

        gross = _format_margin_percent(row.get(g_col)) if g_col else None
        operating = _format_margin_percent(row.get(o_col)) if o_col else None
        net_m = _format_margin_percent(row.get(n_col)) if n_col else None
        ev_e = ev_map.get(industry.lower())

        if gross is None and operating is None and net_m is None and ev_e is None:
            skipped += 1
            logger.debug("Skip industry %r: no numeric fields", industry)
            continue

        chunk = _build_chunk(industry, gross, operating, net_m, ev_e)
        meta = {
            "sector": canonical_sector,
            "data_type": DATA_TYPE,
            "source": SOURCE_TAG,
            "ticker": TICKER_META,
        }
        doc_id = _stable_doc_id(industry)
        batch_texts.append(chunk)
        batch_meta.append(meta)
        batch_ids.append(doc_id)
        stored_names.append(industry)

    stored = 0
    if batch_texts:
        try:
            store.add_texts(texts=batch_texts, metadatas=batch_meta, ids=batch_ids)
            stored = len(batch_texts)
            logger.info("Stored %d Damodaran industry documents in one batch.", stored)
        except Exception as exc:
            skipped += len(batch_texts)
            logger.exception("Batch store to ChromaDB failed: %s", exc)
            stored_names = []

    reg = _load_registry(REGISTRY_PATH)
    reg["industry_averages"] = sorted(set(stored_names))
    _save_registry(REGISTRY_PATH, reg)

    print("=== Damodaran import summary ===")
    print(f"Rows processed (margin sheet): {processed}")
    print(f"Industries stored in ChromaDB: {stored}")
    print(f"Rows/industries skipped:       {skipped}")
    print(f"Registry key industry_averages: {len(reg['industry_averages'])} names")
    print(f"Diagnostic file: {DIAGNOSTIC_PATH}")
    if mapping_errors:
        print(f"Sector mapping warnings/errors: {len(mapping_errors)}")
    logger.info(
        "Done. processed=%s stored=%s skipped=%s",
        processed,
        stored,
        skipped,
    )


if __name__ == "__main__":
    main()
