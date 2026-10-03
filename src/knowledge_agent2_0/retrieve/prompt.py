from typing import Any

SYSTEM_PROMPT = """
You are a technical research assistant.

Answer the user's question using the supplied context.

Rules:
1. Ground your answer in the supplied context.
2. Do not invent facts or citations.
3. If the context is insufficient, explicitly say so.
4. Explain technical concepts clearly.
5. Cite supporting sources using the source labels provided.
6. Do not claim that a source supports something unless
   the retrieved passage actually supports it.

The context contains excerpts from research papers.
"""


def build_user_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Build the user prompt with retrieved context and source labels.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        source_label = (
            f"[S{index}: {chunk['filename']}, "
            f"page {chunk['page_number']}]"
        )

        context_parts.append(
            f"{source_label}\n{chunk['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""
CONTEXT:

{context}

QUESTION:

{question}

Provide a detailed answer based on the context.
Include source labels such as [S1] in your answer.
"""
