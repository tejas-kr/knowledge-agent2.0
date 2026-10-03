import argparse

from knowledge_agent2_0.config import (
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    TOP_K,
)
from knowledge_agent2_0.generate.pipeline import ask_question
from knowledge_agent2_0.ingest.pipeline import ingest_documents


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint: ingest documents into the vector store,
    then answer questions.
    """

    parser = argparse.ArgumentParser(
        description="Personal knowledge base agent (RAG over research PDFs).",
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Ask a single question, then exit.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-chunk and re-embed all documents, overwriting the vector store.",
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("PERSONAL KNOWLEDGE BASE AGENT")
    print("=" * 70)

    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"Generation model: {GENERATION_MODEL}")

    try:
        ingest_documents(force=args.force)

    except Exception as exc:
        print(f"\nInitialization failed: {exc}")
        return 1

    print("\nKnowledge base ready.")

    if args.question:
        ask_question(
            args.question.strip(),
            top_k=args.top_k,
        )
        return 0

    print("Ask questions about your research papers.")

    while True:

        try:
            question = input("\nAsk a question (or 'quit'): ").strip()

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            break

        try:
            ask_question(
                question,
                top_k=args.top_k,
            )

        except Exception as exc:
            print(f"\nError: {exc}")

    return 0
