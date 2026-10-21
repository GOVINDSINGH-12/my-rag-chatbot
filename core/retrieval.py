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


def retrieve(
    question: str,
    retriever,
    config: dict,
    source_path: str | None = None,
) -> tuple:
    """
    Search for the configured number of closest chunks, optionally within one file.

    Chroma stores each source path in chunk metadata, so a named-document query
    can filter by that exact path without changing the index format.
    """
    if source_path:
        vectorstore = getattr(retriever, "vectorstore", None)
        if vectorstore is None:
            raise ValueError(
                "The active retriever does not expose its vector store for "
                "document-filtered search."
            )
        normalized_source = os.path.normcase(os.path.abspath(source_path))
        documents = vectorstore.similarity_search(
            question,
            k=config["top_k"],
            filter={"source": normalized_source},
        )
    else:
        documents = retriever.invoke(question)

    chunks = []
    chunk_details = []
    selected_source_name = os.path.basename(source_path) if source_path else None
    for document in documents:
        source = selected_source_name or os.path.basename(
            document.metadata.get("source", "unknown")
        )
        page = document.metadata.get("page")
        citation = (
            f"{source} (page {page + 1})"
            if isinstance(page, int)
            else source
        )
        # Keep provenance in the model context as well as in the visible citation.
        chunks.append(f"Source: {citation}\n{document.page_content}")
        chunk_details.append(
            {
                "text": document.page_content,
                "source": source,
                "citation": citation,
            }
        )

    sources = sorted({detail["source"] for detail in chunk_details})

    return chunks, sources, chunk_details
