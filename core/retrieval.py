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

# Step 1: Import Chroma and the shared embedding-model loader.
import os
from langchain_community.vectorstores import Chroma

from core.embeddings import get_embeddings


# Step 2: Open the collection and configure its similarity-search retriever.
def load_retriever(config: dict):
    """Connect to Chroma and configure similarity search for the top-k chunks."""
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

    Returns model-ready chunks, unique source filenames, and citation details
    used to display the retrieved passages in the chat.
    """
    # Step 3: Search either the requested file alone or the full collection.
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

    # Step 4: Add citations and prepare context for the model and chat UI.
    chunks = []
    chunk_details = []
    selected_source_name = os.path.basename(source_path) if source_path else None
    for document in documents:
        source = selected_source_name or os.path.basename(
            document.metadata.get("source", "unknown")
        )
        page = document.metadata.get("page")
        sheet = document.metadata.get("sheet")
        if sheet:
            citation = f"{source} (sheet: {sheet})"
        elif isinstance(page, int):
            citation = f"{source} (page {page + 1})"
        else:
            citation = source
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
