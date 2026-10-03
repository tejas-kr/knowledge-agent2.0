import pytest

from knowledge_agent2_0.ingest.chunker import chunk_text, create_chunks


def test_chunk_text_splits_into_chunks():
    words = [str(i) for i in range(10)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=4, overlap=0)

    assert chunks == ["0 1 2 3", "4 5 6 7", "8 9"]


def test_chunk_text_overlap():
    words = [str(i) for i in range(8)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=4, overlap=2)

    assert chunks == ["0 1 2 3", "2 3 4 5", "4 5 6 7", "6 7"]


def test_chunk_text_single_chunk_when_text_fits():
    chunks = chunk_text("alpha beta gamma", chunk_size=10, overlap=2)

    assert chunks == ["alpha beta gamma"]


def test_chunk_text_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("some text", chunk_size=0)


@pytest.mark.parametrize("overlap", [-1, 5])
def test_chunk_text_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="Overlap must be >= 0 and < chunk_size"):
        chunk_text("some text", chunk_size=4, overlap=overlap)


def test_create_chunks_metadata():
    pages = [{
        "document_id": "paper",
        "filename": "paper.pdf",
        "page_number": 2,
        "content": " ".join(str(i) for i in range(10)),
    }]

    chunks = create_chunks(pages)

    assert len(chunks) == 1

    chunk = chunks[0]
    assert chunk["chunk_id"] == "paper_p2_c0"
    assert chunk["document_id"] == "paper"
    assert chunk["filename"] == "paper.pdf"
    assert chunk["page_number"] == 2
    assert chunk["chunk_index"] == 0
