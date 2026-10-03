import pytest
from fastapi.testclient import TestClient

import knowledge_agent2_0.api as api_module
from knowledge_agent2_0.api import app


client = TestClient(app)


class FakeCollection:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


SAMPLE_CHUNKS = [
    {
        "chunk_id": "paper_p1_c0",
        "document_id": "paper",
        "filename": "paper.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "score": 0.9,
        "content": "relevant content",
    },
    {
        "chunk_id": "paper_p1_c1",
        "document_id": "paper",
        "filename": "paper.pdf",
        "page_number": 1,
        "chunk_index": 1,
        "score": 0.5,
        "content": "less relevant content",
    },
]


def test_health(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "get_collection",
        lambda: FakeCollection(1109),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "vector_store_size": 1109,
    }


def test_ingest(monkeypatch):
    def fake_ingest(force: bool = False) -> int:
        assert force is False
        return 4

    monkeypatch.setattr(api_module, "ingest_documents", fake_ingest)

    response = client.post("/ingest", json={})

    assert response.status_code == 200
    assert response.json() == {"chunks_stored": 4}


def test_ingest_with_force(monkeypatch):
    def fake_ingest(force: bool = False) -> int:
        assert force is True
        return 4

    monkeypatch.setattr(api_module, "ingest_documents", fake_ingest)

    response = client.post("/ingest", json={"force": True})

    assert response.status_code == 200
    assert response.json() == {"chunks_stored": 4}


def test_ingest_error(monkeypatch):
    def boom(force: bool = False) -> int:
        raise RuntimeError("No PDF files found")

    monkeypatch.setattr(api_module, "ingest_documents", boom)

    response = client.post("/ingest", json={})

    assert response.status_code == 500
    assert "No PDF files found" in response.json()["detail"]


def test_retrieve(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "retrieve",
        lambda question, top_k=5: SAMPLE_CHUNKS[:top_k],
    )

    response = client.post(
        "/retrieve",
        json={"question": "what is rag?", "top_k": 1},
    )

    assert response.status_code == 200

    body = response.json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["filename"] == "paper.pdf"
    assert body["sources"][0]["score"] == 0.9
    assert body["sources"][0]["content"] == "relevant content"


def test_retrieve_empty_store(monkeypatch):
    def boom(question: str, top_k: int = 5):
        raise RuntimeError("Vector store is empty. Ingest documents first.")

    monkeypatch.setattr(api_module, "retrieve", boom)

    response = client.post("/retrieve", json={"question": "hi"})

    assert response.status_code == 503
    assert "Vector store is empty" in response.json()["detail"]


def test_ask(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "retrieve",
        lambda question, top_k=5: SAMPLE_CHUNKS,
    )
    monkeypatch.setattr(
        api_module,
        "generate_answer",
        lambda question, chunks: "RAG retrieves then generates.",
    )

    response = client.post(
        "/ask",
        json={"question": "what is rag?", "top_k": 2},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["answer"] == "RAG retrieves then generates."
    assert len(body["sources"]) == 2
    assert body["sources"][0]["chunk_id"] == "paper_p1_c0"


def test_ask_empty_store(monkeypatch):
    def boom(question: str, top_k: int = 5):
        raise RuntimeError("Vector store is empty. Ingest documents first.")

    monkeypatch.setattr(api_module, "retrieve", boom)

    response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == 503


def test_ask_generation_error(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "retrieve",
        lambda question, top_k=5: SAMPLE_CHUNKS,
    )

    def boom(question: str, chunks) -> str:
        raise ConnectionError("ollama not running")

    monkeypatch.setattr(api_module, "generate_answer", boom)

    response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == 502
    assert "ollama not running" in response.json()["detail"]


@pytest.mark.parametrize(
    "endpoint",
    ["/retrieve", "/ask"],
)
def test_missing_question_returns_422(endpoint):
    response = client.post(endpoint, json={"top_k": 3})

    assert response.status_code == 422
