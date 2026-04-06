"""
Ingestion pipeline for the Comparative Lifestyle RAG.

Reads .txt files from data/, chunks them with LangChain,
generates dense embeddings with BAAI/bge-m3, and upserts
into Qdrant following the metadata schema from .cursorrules.
"""

import os
import uuid
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_API_KEY:
    sys.exit("QDRANT_API_KEY is not set or is empty. Check your .env file.")
COLLECTION_NAME = "health_rag"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_FILE_SIZE_MB = 50


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection if it doesn't already exist."""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"  Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"  Collection '{COLLECTION_NAME}' already exists — skipping creation.")


def load_and_chunk(file_path: Path) -> list[str]:
    """Load a .txt file via LangChain and split it into chunks."""
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        sys.exit(
            f"File too large: {size_mb:.1f} MB exceeds the "
            f"{MAX_FILE_SIZE_MB} MB limit."
        )

    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    return [chunk.page_content for chunk in chunks]


def embed_chunks(chunks: list[str], model: SentenceTransformer) -> list[list[float]]:
    """Generate normalised dense embeddings for every chunk."""
    vectors = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def build_points(
    chunks: list[str],
    embeddings: list[list[float]],
    *,
    author: str,
    paradigm: str,
    domain: str,
    source_type: str,
    source_title: str,
    keywords: list[str],
) -> list[PointStruct]:
    """
    Build Qdrant PointStructs with the exact payload schema
    defined in .cursorrules:

    {
      "chunk_id": "uuid",
      "text": "...",
      "metadata": {
        "author", "paradigm", "domain",
        "source_type", "source_title", "keywords"
      }
    }
    """
    points: list[PointStruct] = []
    for text, vector in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        payload = {
            "chunk_id": point_id,
            "text": text,
            "metadata": {
                "author": author,
                "paradigm": paradigm,
                "domain": domain,
                "source_type": source_type,
                "source_title": source_title,
                "keywords": keywords,
            },
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
    return points


def _connect() -> QdrantClient:
    """Create a QdrantClient with optional API-key auth."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ingest(
    file_path: Path,
    *,
    author: str,
    paradigm: str,
    domain: str,
    source_type: str,
    source_title: str,
    keywords: list[str],
) -> int:
    """End-to-end ingestion: load -> chunk -> embed -> upsert. Returns point count."""
    try:
        client = _connect()
        ensure_collection(client)
    except Exception as exc:
        sys.exit(f"Failed to connect to Qdrant at {QDRANT_URL}: {exc}")

    print(f"Loading and chunking '{file_path.name}' ...")
    chunks = load_and_chunk(file_path)
    print(f"  {len(chunks)} chunks produced (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    try:
        print(f"Generating embeddings with '{EMBEDDING_MODEL}' ...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        vectors = embed_chunks(chunks, model)
    except Exception as exc:
        sys.exit(f"Embedding model failed: {exc}")

    print("Building points with metadata ...")
    points = build_points(
        chunks,
        vectors,
        author=author,
        paradigm=paradigm,
        domain=domain,
        source_type=source_type,
        source_title=source_title,
        keywords=keywords,
    )

    try:
        print(f"Upserting {len(points)} points into '{COLLECTION_NAME}' ...")
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print("  Upsert complete.")
    except Exception as exc:
        sys.exit(f"Upsert failed: {exc}")

    return len(points)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest a .txt file into the health-rag Qdrant collection.",
    )
    p.add_argument("file", type=Path, help="Path to the .txt file (e.g. data/ray_peat_sugar.txt)")
    p.add_argument("--author", required=True, help="e.g. 'Ray Peat'")
    p.add_argument("--paradigm", required=True, help="e.g. 'bioenergetic'")
    p.add_argument("--domain", required=True, help="e.g. 'nutrition'")
    p.add_argument("--source-type", required=True, dest="source_type", help="e.g. 'article'")
    p.add_argument("--source-title", required=True, dest="source_title", help="e.g. 'Sugar and Metabolism'")
    p.add_argument("--keywords", nargs="+", default=[], help="e.g. sugar metabolism thyroid")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.file.exists():
        sys.exit(f"File not found: {args.file}")
    if args.file.suffix.lower() != ".txt":
        sys.exit(f"Only .txt files are supported, got: {args.file.suffix}")

    total = ingest(
        args.file,
        author=args.author,
        paradigm=args.paradigm,
        domain=args.domain,
        source_type=args.source_type,
        source_title=args.source_title,
        keywords=args.keywords,
    )
    print(f"\nIngestion complete — {total} chunks indexed in '{COLLECTION_NAME}'.")
