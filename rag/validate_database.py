from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import json

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def main() -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name="peer_metrics",
        embedding_function=embeddings,
        persist_directory="rag/chroma_db",
    )

    # Fetch all documents stored in the collection.
    collection = vector_store._collection  # type: ignore[attr-defined]
    records = collection.get()

    ids = records.get("ids", []) or []
    metadatas = records.get("metadatas", []) or []
    documents = records.get("documents", []) or []

    print("=== ChromaDB Peer Metrics Validation Report ===")
    if not ids:
        print("No documents found in the 'peer_metrics' collection.")
    else:
        for idx, (doc_id, metadata, doc_text) in enumerate(
            zip(ids, metadatas, documents), start=1
        ):
            ticker = metadata.get("ticker", "N/A") if isinstance(metadata, dict) else "N/A"
            company_name = (
                metadata.get("company_name", "N/A") if isinstance(metadata, dict) else "N/A"
            )

            sector = metadata.get("sector", "N/A") if isinstance(metadata, dict) else "N/A"

            print(f"\n--- Company #{idx} ---")
            print(f"Ticker:        {ticker}")
            print(f"Company Name:  {company_name}")
            print(f"Sector:        {sector}")
            print(f"Document ID:   {doc_id}")
            print("Raw Metadata:")
            print(json.dumps(metadata or {}, indent=2, sort_keys=True))
            print("Raw Text:")
            print(doc_text)

    print("\n=== End of Validation Report ===")


if __name__ == "__main__":
    main()

