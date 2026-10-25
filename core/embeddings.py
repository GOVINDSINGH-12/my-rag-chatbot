"""Load and reuse the local text-embedding model used by indexing and search."""

# Step 1: Import the cache and the embedding model implementation.
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


# Step 2: Keep a small cache so reruns do not reload the same model repeatedly.
@lru_cache(maxsize=4)
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Reuse the loaded local embedding model across indexing and retrieval."""
    return HuggingFaceEmbeddings(model_name=model_name)
