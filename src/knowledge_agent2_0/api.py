from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from knowledge_agent2_0.config import TOP_K
from knowledge_agent2_0.generate.llm import generate_answer
from knowledge_agent2_0.ingest.pipeline import ingest_documents
from knowledge_agent2_0.retrieve.retriever import retrieve
from knowledge_agent2_0.store.vectorstore import get_collection

app = FastAPI(
    title="Knowledge Base Agent API",
    description="RAG API over research PDFs: ingest, retrieve, and ask questions.",
    version="0.1.0",
)


class IngestRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="Re-chunk and re-embed all documents, overwriting the vector store.",
    )


class IngestResponse(BaseModel):
    chunks_stored: int


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=TOP_K, ge=1)


class Source(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float
    content: str


class RetrieveResponse(BaseModel):
    sources: list[Source]


class AskRequest(RetrieveRequest):
    pass


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _to_sources(chunks: list[dict[str, Any]]) -> list[Source]:
    return [Source(**chunk) for chunk in chunks]


@app.get("/health")
def health() -> dict[str, Any]:
    """
    Report status and the number of chunks in the vector store.
    """

    collection = get_collection()

    return {
        "status": "ok",
        "vector_store_size": collection.count(),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """
    Load PDFs from documents/, chunk, embed, and store in ChromaDB.
    """

    try:
        count = ingest_documents(force=request.force)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(chunks_stored=count)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(request: RetrieveRequest) -> RetrieveResponse:
    """
    Retrieve the most similar chunks without generating an answer.
    """

    try:
        chunks = retrieve(
            request.question,
            top_k=request.top_k,
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding failed: {exc}",
        )

    return RetrieveResponse(sources=_to_sources(chunks))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Retrieve context from the vector store and generate an answer.
    """

    try:
        chunks = retrieve(
            request.question,
            top_k=request.top_k,
        )

        answer = generate_answer(request.question, chunks)

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Generation failed: {exc}",
        )

    return AskResponse(answer=answer, sources=_to_sources(chunks))


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Run the API server.
    """

    import uvicorn

    uvicorn.run(app, host=host, port=port)
