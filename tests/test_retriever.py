import uuid

import chromadb
import pytest

from knowledge_agent2_0.retrieve.retriever import retrieve
from knowledge_agent2_0.store.vectorstore import get_collection


def unique_name(prefix: str = "test") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def make_collection() -> chromadb.Collection:
    client = chromadb.EphemeralClient()
    collection = get_collection(client=client, name=unique_name())

    collection.add(
        ids=["unrelated_c0", "relevant_c0", "partial_c0"],
        embeddings=[[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]],
        documents=["unrelated content", "relevant content", "partial content"],
        metadatas=[
            {
                "document_id": "unrelated",
                "filename": "unrelated.pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
            {
                "document_id": "relevant",
                "filename": "relevant.pdf",
                "page_number": 3,
                "chunk_index": 0,
            },
            {
                "document_id": "partial",
                "filename": "partial.pdf",
                "page_number": 5,
                "chunk_index": 0,
            },
        ],
    )

    return collection


def test_retrieve_ranks_by_similarity():
    collection = make_collection()

    results = retrieve(
        "query",
        collection=collection,
        top_k=2,
        embed_fn=lambda text: [1.0, 0.0],
    )

    assert len(results) == 2

    assert results[0]["document_id"] == "relevant"
    assert results[0]["filename"] == "relevant.pdf"
    assert results[0]["page_number"] == 3
    assert results[0]["content"] == "relevant content"
    assert results[0]["score"] == pytest.approx(1.0)

    assert results[1]["document_id"] == "partial"
    assert results[1]["score"] == pytest.approx(1.0 / (2.0 ** 0.5))


def test_retrieve_respects_top_k():
    client = chromadb.EphemeralClient()
    collection = get_collection(client=client, name=unique_name())

    collection.add(
        ids=[f"doc{i}_c0" for i in range(10)],
        embeddings=[[1.0, 0.0]] * 10,
        documents=[f"content {i}" for i in range(10)],
        metadatas=[
            {"filename": f"doc{i}.pdf", "page_number": 1, "chunk_index": 0}
            for i in range(10)
        ],
    )

    results = retrieve(
        "query",
        collection=collection,
        top_k=3,
        embed_fn=lambda text: [1.0, 0.0],
    )

    assert len(results) == 3


def test_retrieve_empty_store_raises():
    client = chromadb.EphemeralClient()
    collection = get_collection(client=client, name=unique_name())

    with pytest.raises(RuntimeError, match="Vector store is empty"):
        retrieve("query", collection=collection)
