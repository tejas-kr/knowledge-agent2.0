import uuid

import chromadb

from knowledge_agent2_0.store.vectorstore import get_collection


def unique_name(prefix: str = "test") -> str:
    """
    ChromaDB clients share one in-process store, so each test
    needs a unique collection name to stay isolated.
    """

    return f"{prefix}-{uuid.uuid4().hex}"


def test_get_collection_creates_and_reuses():
    client = chromadb.EphemeralClient()
    name = unique_name()

    collection = get_collection(client=client, name=name)

    assert collection.count() == 0

    again = get_collection(client=client, name=name)

    assert again.count() == 0


def test_add_and_query_roundtrip():
    client = chromadb.EphemeralClient()
    collection = get_collection(client=client, name=unique_name())

    collection.add(
        ids=["c1", "c2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["alpha text", "beta text"],
        metadatas=[
            {"filename": "a.pdf", "page_number": 1},
            {"filename": "b.pdf", "page_number": 2},
        ],
    )

    assert collection.count() == 2

    results = collection.query(
        query_embeddings=[[1.0, 0.0]],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )

    assert results["ids"][0][0] == "c1"
    assert results["documents"][0][0] == "alpha text"
    assert results["distances"][0][0] == 0.0
    assert results["distances"][0][1] == 1.0


def test_upsert_is_idempotent():
    client = chromadb.EphemeralClient()
    collection = get_collection(client=client, name=unique_name())

    collection.upsert(
        ids=["c1"],
        embeddings=[[1.0, 0.0]],
        documents=["alpha text"],
        metadatas=[{"filename": "a.pdf", "page_number": 1}],
    )

    collection.upsert(
        ids=["c1"],
        embeddings=[[1.0, 0.0]],
        documents=["alpha text"],
        metadatas=[{"filename": "a.pdf", "page_number": 1}],
    )

    assert collection.count() == 1
