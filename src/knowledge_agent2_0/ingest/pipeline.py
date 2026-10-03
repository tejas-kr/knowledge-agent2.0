from typing import Any, Callable

from knowledge_agent2_0.ingest.chunker import create_chunks
from knowledge_agent2_0.ingest.embedder import embed_texts
from knowledge_agent2_0.ingest.loader import load_all_pdfs
from knowledge_agent2_0.store.vectorstore import get_collection


def chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a chunk into ChromaDB-compatible metadata.
    """

    return {
        "document_id": chunk["document_id"],
        "filename": chunk["filename"],
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"],
    }


def ingest_documents(
    force: bool = False,
    collection: Any = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> int:
    """
    RAG ingestion pipeline: load PDFs, chunk, embed, and store
    the vectors in ChromaDB.
    """

    if collection is None:
        collection = get_collection()

    embed = embed_fn or embed_texts

    existing_count = collection.count()

    if existing_count > 0 and not force:
        print(
            f"Vector store already contains {existing_count} chunks. "
            "Use --force to rebuild."
        )

        return existing_count

    pages = load_all_pdfs()

    if not pages:
        raise RuntimeError("No extractable PDF text found.")

    chunks = create_chunks(pages)

    if not chunks:
        raise RuntimeError("No chunks were generated.")

    texts = [chunk["content"] for chunk in chunks]

    embeddings = embed(texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[chunk_metadata(chunk) for chunk in chunks],
    )

    print(f"Stored {len(chunks)} chunks in vector store.")

    return len(chunks)
