from typing import Any

from knowledge_agent2_0.config import (
    GENERATION_MODEL,
    GENERATION_NUM_CTX,
    GENERATION_TEMPERATURE,
)
from knowledge_agent2_0.ollama_client import client
from knowledge_agent2_0.retrieve.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)


def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Generate an answer using Qwen through Ollama.
    """

    user_prompt = build_user_prompt(question, retrieved_chunks)

    response = client.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": GENERATION_TEMPERATURE,
            "num_ctx": GENERATION_NUM_CTX,
        },
    )

    return response["message"]["content"]
