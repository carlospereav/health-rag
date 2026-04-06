import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_API_KEY:
    sys.exit("QDRANT_API_KEY is not set or is empty. Check your .env file.")
COLLECTION_NAME = "test_collection"

PROTECTED_COLLECTIONS = {"health_rag"}


def main():
    if COLLECTION_NAME in PROTECTED_COLLECTIONS:
        sys.exit(
            f"COLLECTION_NAME is '{COLLECTION_NAME}' which is a production collection. "
            "Change it to a safe test name before running."
        )

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print(f"Connecting to Qdrant at {QDRANT_URL} ...")
        collections = client.get_collections().collections
        print(f"  OK - Qdrant reachable. Existing collections: {len(collections)}")
    except Exception as exc:
        sys.exit(f"Failed to connect to Qdrant at {QDRANT_URL}: {exc}")

    print(f"Creating collection '{COLLECTION_NAME}' ...")
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    print("  OK - Collection created.")

    print("Verifying collection exists ...")
    info = client.get_collection(collection_name=COLLECTION_NAME)
    assert info.status.value in ("green", "yellow"), f"Unexpected status: {info.status}"
    print(f"  OK - Collection status: {info.status.value}, "
          f"vector size: {info.config.params.vectors.size}")

    print("Cleaning up - deleting test collection ...")
    client.delete_collection(collection_name=COLLECTION_NAME)
    print("  OK - Collection deleted.")

    print("\nAll checks passed. Qdrant is running correctly.")


if __name__ == "__main__":
    main()
