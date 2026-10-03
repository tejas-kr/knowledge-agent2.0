import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "documents"

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "research-papers"

EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
GENERATION_MODEL = "qwen3:1.7b"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

EMBED_BATCH_SIZE = 32
GENERATION_TEMPERATURE = 0.2
GENERATION_NUM_CTX = 8192
