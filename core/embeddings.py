from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=4)
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Reuse the loaded local embedding model across indexing and retrieval."""
    return HuggingFaceEmbeddings(model_name=model_name)
