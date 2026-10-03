from typing import Any

from knowledge_agent2_0.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be >= 0 and < chunk_size")

    words = text.split()

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk_words = words[start:start + chunk_size]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


def create_chunks(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert extracted pages into chunks with source metadata.
    """

    chunks = []

    for page in pages:

        page_chunks = chunk_text(page["content"])

        for chunk_index, content in enumerate(page_chunks):

            chunks.append({
                "chunk_id": (
                    f'{page["document_id"]}_'
                    f'p{page["page_number"]}_'
                    f'c{chunk_index}'
                ),
                "document_id": page["document_id"],
                "filename": page["filename"],
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "content": content,
            })

    print(f"Created {len(chunks)} chunks.")

    return chunks
