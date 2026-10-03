from typing import Any, Callable

from knowledge_agent2_0.config import TOP_K
from knowledge_agent2_0.ingest.embedder import embed_query
from knowledge_agent2_0.store.vectorstore import get_collection


def retrieve(
    question: str,
    collection: Any = None,
    top_k: int = TOP_K,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the most semantically similar chunks from ChromaDB.
    """

    if collection is None:
        collection = get_collection()

    embed = embed_fn or embed_query

    count = collection.count()

    if count == 0:
        raise RuntimeError(
            "Vector store is empty. Ingest documents first."
        )

    query_embedding = embed(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []

    for chunk_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):

        retrieved_chunks.append({
            "chunk_id": chunk_id,
            "content": document,
            "score": 1.0 - distance,
            **metadata,
        })

    return retrieved_chunks
