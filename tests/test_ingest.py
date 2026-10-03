import uuid

import chromadb
import pytest

import knowledge_agent2_0.ingest.pipeline as pipeline_module
from knowledge_agent2_0.ingest.pipeline import ingest_documents
from knowledge_agent2_0.retrieve.retriever import retrieve
from knowledge_agent2_0.store.vectorstore import get_collection


SAMPLE_PAGES = [
    {
        "document_id": "paper",
        "filename": "paper.pdf",
        "page_number": 1,
        "content": " ".join(["alpha"] * 600),
    },
    {
        "document_id": "paper",
        "filename": "paper.pdf",
        "page_number": 2,
        "content": " ".join(["beta"] * 600),
    },
]


def unique_name(prefix: str = "test") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def fake_embed_batch(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    return get_collection(client=client, name=unique_name())


def test_ingest_chunks_embeds_and_stores(monkeypatch, collection):
    monkeypatch.setattr(
        pipeline_module,
        "load_all_pdfs",
        lambda: SAMPLE_PAGES,
    )

    count = ingest_documents(
        collection=collection,
        embed_fn=fake_embed_batch,
    )

    assert count == 4
    assert collection.count() == 4


def test_ingest_then_retrieve_pipeline(monkeypatch, collection):
    monkeypatch.setattr(
        pipeline_module,
        "load_all_pdfs",
        lambda: SAMPLE_PAGES,
    )

    ingest_documents(collection=collection, embed_fn=fake_embed_batch)

    results = retrieve(
        "alpha question",
        collection=collection,
        embed_fn=lambda text: [1.0, 0.0],
    )

    assert results[0]["filename"] == "paper.pdf"
    assert results[0]["page_number"] == 1
    assert "alpha" in results[0]["content"]
    assert results[0]["score"] == pytest.approx(1.0)


def test_ingest_skips_existing_store(monkeypatch, collection):
    monkeypatch.setattr(
        pipeline_module,
        "load_all_pdfs",
        lambda: SAMPLE_PAGES,
    )

    ingest_documents(collection=collection, embed_fn=fake_embed_batch)

    def boom(texts: list[str]) -> list[list[float]]:
        raise AssertionError("should not re-embed an existing store")

    count = ingest_documents(collection=collection, embed_fn=boom)

    assert count == 4
    assert collection.count() == 4


def test_ingest_force_rebuilds(monkeypatch, collection):
    monkeypatch.setattr(
        pipeline_module,
        "load_all_pdfs",
        lambda: SAMPLE_PAGES,
    )

    ingest_documents(collection=collection, embed_fn=fake_embed_batch)

    def opposite_embed(texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] if "alpha" in text else [1.0, 0.0] for text in texts]

    count = ingest_documents(
        collection=collection,
        force=True,
        embed_fn=opposite_embed,
    )

    assert count == 4
    assert collection.count() == 4

    results = retrieve(
        "alpha question",
        collection=collection,
        embed_fn=lambda text: [1.0, 0.0],
    )

    assert "beta" in results[0]["content"]


def test_ingest_without_pages_raises(monkeypatch, collection):
    monkeypatch.setattr(
        pipeline_module,
        "load_all_pdfs",
        lambda: [],
    )

    with pytest.raises(RuntimeError, match="No extractable PDF text"):
        ingest_documents(collection=collection, embed_fn=fake_embed_batch)
