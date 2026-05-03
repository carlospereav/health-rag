"""
Semantic search against the health-rag Qdrant collection.

Encodes a free-text query with the same BAAI/bge-m3 model used at
ingestion time, then retrieves the top-k most similar chunks.
"""

import os
import sys
import argparse

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_API_KEY:
    sys.exit("QDRANT_API_KEY is not set or is empty. Check your .env file.")

COLLECTION_NAME = "health_rag"
EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_TOP_K = 5


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Encode *query* and return the top-k nearest chunks from Qdrant."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = model.encode(
        query, normalize_embeddings=True
    ).tolist()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    hits: list[dict] = []
    for point in results.points:
        hits.append({
            "score": point.score,
            "text": point.payload.get("text", ""),
            "metadata": point.payload.get("metadata", {}),
        })
    return hits


def _print_results(hits: list[dict], query: str) -> None:
    """Pretty-print search results to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  Query: \"{query}\"")
    print(f"  Results: {len(hits)} hits")
    print(f"{'=' * 70}\n")

    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        print(f"--- Result {i}  (score: {hit['score']:.4f}) ---")
        print(f"  Author:   {meta.get('author', 'N/A')}")
        print(f"  Paradigm: {meta.get('paradigm', 'N/A')}")
        print(f"  Domain:   {meta.get('domain', 'N/A')}")
        print(f"  Source:   {meta.get('source_title', 'N/A')} ({meta.get('source_type', 'N/A')})")
        if meta.get("keywords"):
            print(f"  Keywords: {', '.join(meta['keywords'])}")
        print(f"\n  {hit['text']}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Semantic search over the health-rag Qdrant collection.",
    )
    p.add_argument("query", help="Free-text search query (e.g. 'benefits of sugar for metabolism')")
    p.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of results (default {DEFAULT_TOP_K})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hits = search(args.query, top_k=args.top_k)
    if not hits:
        print("No results found. Is the collection populated?")
    else:
        _print_results(hits, args.query)
