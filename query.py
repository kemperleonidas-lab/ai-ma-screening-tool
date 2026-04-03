"""RAG query interface for retrieving relevant peer company metrics from ChromaDB."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

RAG_DIR = Path(__file__).resolve().parent
CHROMA_DIR = RAG_DIR / "chroma_db"
COLLECTION_NAME = "peer_metrics"

_vector_store: Chroma | None = None


def _get_vector_store() -> Chroma:
    """Initialize and return the ChromaDB vector store (cached)."""
    global _vector_store
    if _vector_store is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    return _vector_store


def query_peers(
    sector: str,
    ticker_to_exclude: str,
    n_results: int = 5,
    size_tier: str | None = None,
) -> dict[str, object]:
    """
    Query direct peer company documents (non-industry averages) for a given sector.

    Args:
        sector: Sector to filter by (e.g. "Technology Software"); only documents with
                metadata["sector"] == sector are considered.
        ticker_to_exclude: Exclude any document where metadata["ticker"] matches this value.
        n_results: Maximum number of documents to return.

    Returns:
        Dict with:
            - docs: List[Document]
            - retrieval_mode: "size_matched" | "sector_fallback" | "sector_only"
    """
    store = _get_vector_store()
    exclude = (ticker_to_exclude or "").strip().upper()

    def _post_filter(raw_docs: list[Document]) -> list[Document]:
        out: list[Document] = []
        for d in raw_docs:
            meta = getattr(d, "metadata", None) or {}
            if meta.get("data_type") == "industry_average":
                continue
            t = str(meta.get("ticker", "")).strip().upper()
            if exclude and t == exclude:
                continue
            out.append(d)
            if len(out) >= n_results:
                break
        return out

    tier = (size_tier or "").strip()
    if tier:
        # First pass: strict sector + size_tier match.
        where_size = (
            {"$and": [{"sector": sector}, {"size_tier": tier}]}
            if sector
            else {"size_tier": tier}
        )
        docs_size = store.similarity_search(
            sector or "peers",
            k=max(10, n_results * 3),
            filter=where_size,
        )
        out_size = _post_filter(docs_size)
        if len(out_size) >= 3:
            return {"docs": out_size, "retrieval_mode": "size_matched"}

        # Fallback: sector-only so thin cohorts don't silently block benchmarking.
        docs_sector = store.similarity_search(
            sector or "peers",
            k=max(10, n_results * 3),
            filter={"sector": sector} if sector else None,
        )
        out_sector = _post_filter(docs_sector)
        return {"docs": out_sector, "retrieval_mode": "sector_fallback"}

    # No size tier requested: default sector-only retrieval.
    docs = store.similarity_search(
        sector or "peers",
        k=max(10, n_results * 3),
        filter={"sector": sector} if sector else None,
    )
    out = _post_filter(docs)
    return {"docs": out, "retrieval_mode": "sector_only"}


def query_industry_averages(
    sector: str,
    n_results: int = 3,
) -> list[Document]:
    """Query Chroma for Damodaran industry_average docs within the given sector."""
    store = _get_vector_store()
    # Chroma filter dialect in this environment expects operators like $and, not multiple equality keys.
    where = (
        {"$and": [{"sector": sector}, {"data_type": "industry_average"}]}
        if sector
        else {"data_type": "industry_average"}
    )
    docs = store.similarity_search(
        sector or "industry averages",
        k=n_results,
        filter=where,
    )
    return docs
