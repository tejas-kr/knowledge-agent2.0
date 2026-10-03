from pathlib import Path
from typing import Any

import pymupdf

from knowledge_agent2_0.config import DOCUMENTS_DIR


def load_pdf(path: Path) -> list[dict[str, Any]]:
    """
    Extract text from a PDF while preserving page metadata.
    """

    documents = []

    with pymupdf.open(path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append({
                "document_id": path.stem,
                "filename": path.name,
                "page_number": page_number,
                "content": text,
            })

    return documents


def load_all_pdfs() -> list[dict[str, Any]]:
    """
    Discover and load all PDFs from documents/.
    """

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENTS_DIR}"
        )

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    print(f"\nFound {len(pdf_files)} PDF files.")

    all_pages = []

    for pdf_path in pdf_files:

        print(f"Loading: {pdf_path.name}")

        try:
            pages = load_pdf(pdf_path)
            all_pages.extend(pages)

            print(f"  Extracted {len(pages)} pages.")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nTotal pages extracted: {len(all_pages)}")

    return all_pages
