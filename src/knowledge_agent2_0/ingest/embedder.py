from knowledge_agent2_0.config import EMBED_BATCH_SIZE, EMBEDDING_MODEL
from knowledge_agent2_0.ollama_client import client


def embed_texts(
    texts: list[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """
    Generate embeddings using the local Ollama embedding model.

    Processes inputs in batches to avoid excessively large requests.
    """

    embeddings = []

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        response = client.embed(
            model=EMBEDDING_MODEL,
            input=batch,
            truncate=True,
        )

        embeddings.extend(response["embeddings"])

        completed = min(start + len(batch), len(texts))

        print(
            f"\rEmbedding: {completed}/{len(texts)}",
            end="",
            flush=True,
        )

    print()

    return embeddings


def embed_query(text: str) -> list[float]:
    """
    Generate an embedding for a single query.
    """

    response = client.embed(
        model=EMBEDDING_MODEL,
        input=text,
        truncate=True,
    )

    return response["embeddings"][0]
