from typing import Any

import chromadb

from knowledge_agent2_0.config import CHROMA_DIR, COLLECTION_NAME


_client: chromadb.Client | None = None


def get_client() -> chromadb.Client:
    """
    Return a shared persistent ChromaDB client.
    """

    global _client

    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return _client


def get_collection(
    client: chromadb.Client | None = None,
    name: str = COLLECTION_NAME,
) -> Any:
    """
    Return the RAG collection, creating it if needed.

    Cosine distance is used so similarity scores behave
    like cosine similarity (score = 1 - distance).
    """

    if client is None:
        client = get_client()

    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
