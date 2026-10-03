import ollama

from knowledge_agent2_0.config import OLLAMA_HOST

# Ollama runs locally by default at http://localhost:11434
client = ollama.Client(host=OLLAMA_HOST)
