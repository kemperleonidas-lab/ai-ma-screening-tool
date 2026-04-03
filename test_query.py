"""Quick ChromaDB smoke test: two semantic queries against peer_metrics."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

RAG_DIR = Path(__file__).resolve().parent
CHROMA_DIR = RAG_DIR / "chroma_db"
COLLECTION_NAME = "peer_metrics"
K = 3


def _vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def _print_results(title: str, query: str, docs: list) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"Query: {query!r}")
    print(f"Top {len(docs)} results")
    print("=" * 60)
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Result #{i} ---")
        print("Text:")
        print(doc.page_content)
        print("Metadata:")
        print(json.dumps(doc.metadata, indent=2, sort_keys=True))


def main() -> None:
    store = _vector_store()

    q1 = "Technology software margins"
    docs1 = store.similarity_search(q1, k=K)
    _print_results("Query 1: Technology software margins", q1, docs1)

    q2 = "industry average software"
    docs2 = store.similarity_search(q2, k=K)
    _print_results("Query 2: industry average software", q2, docs2)

    print(f"\n{'=' * 60}")
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
