from typing import Any

from knowledge_agent2_0.config import TOP_K
from knowledge_agent2_0.generate.llm import generate_answer
from knowledge_agent2_0.retrieve.retriever import retrieve


def ask_question(
    question: str,
    collection: Any = None,
    top_k: int = TOP_K,
) -> str:
    """
    Retrieve context from the vector store and generate an answer.
    """

    print("\nSearching knowledge base...")

    retrieved_chunks = retrieve(
        question,
        collection=collection,
        top_k=top_k,
    )

    print("\nRetrieved sources:")

    for index, chunk in enumerate(retrieved_chunks, start=1):

        print(
            f"  [S{index}] {chunk['filename']} "
            f"| Page {chunk['page_number']} "
            f"| Similarity: {chunk['score']:.4f}"
        )

    print("\nGenerating answer...\n")

    answer = generate_answer(
        question,
        retrieved_chunks,
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)
    print("=" * 70)

    return answer
