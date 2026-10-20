# =============================================================================
# core/retrieval.py
# WHAT THIS FILE DOES: Searches ChromaDB for chunks relevant to a question
# Same logic as langchain_rag/lc_retrieval.py but adapted for Streamlit
#
# FLOW:
#   User Question
#       --> HuggingFaceEmbeddings  (convert question to vector)
#       --> Chroma.as_retriever()  (search ChromaDB for similar chunks)
#       --> Returns top-k chunks with source filenames
# =============================================================================

import os
from langchain_community.vectorstores import Chroma

from core.embeddings import get_embeddings


def load_retriever(config: dict):
    """
    Step 1: connect to the existing Chroma collection.
    Step 2: configure retrieval to return the top-k closest chunks.
    """
    embeddings = get_embeddings(config["embedding_model"])

    vectorstore = Chroma(
        persist_directory=config["chroma_db_path"],
        embedding_function=embeddings,
        collection_name=config["collection_name"]
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config["top_k"]},
    )


def retrieve(question: str, retriever, config: dict) -> tuple:
    """
    Step 1: search Chroma for the configured number of closest chunks.
    Step 2: return those exact chunks with their source names and page details.
    """
    documents = retriever.invoke(question)

    chunks = [document.page_content for document in documents]

    sources = sorted({
        os.path.basename(document.metadata.get("source", "unknown"))
        for document in documents
    })

    # Keep citation details aligned with the same ordered chunks sent to the LLM.
    chunk_details = [
        {
            "text": document.page_content,
            "source": os.path.basename(document.metadata.get("source", "unknown")),
            "citation": (
                f"{os.path.basename(document.metadata.get('source', 'unknown'))} "
                f"(page {document.metadata['page'] + 1})"
                if isinstance(document.metadata.get("page"), int)
                else os.path.basename(document.metadata.get("source", "unknown"))
            ),
        }
        for document in documents
    ]

    return chunks, sources, chunk_details
