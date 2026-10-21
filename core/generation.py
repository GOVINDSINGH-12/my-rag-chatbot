# =============================================================================
# core/generation.py
# WHAT THIS FILE DOES: Builds the prompt and calls Groq LLM to generate answer
# Same logic as langchain_rag/lc_generation.py but adapted for Streamlit
#
# FLOW:
#   Retrieved Chunks + Question
#       --> ChatPromptTemplate  (builds structured prompt)
#       --> ChatGroq            (calls Groq LLM)
#       --> StrOutputParser     (extracts plain text from response)
#       --> Answer string
# =============================================================================

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: inspect the retrieved chunks for document-specific evidence.
# Step 2: answer from general knowledge when the chunks are unrelated.
PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are DocuChat, a helpful assistant. You are given the user's question and
the top matching passages retrieved from their documents.

Match the answer's length and detail to the request. Give short, direct answers
to simple factual questions. For requests to summarize or explain, provide an
appropriately detailed, organized answer using the available information.

For questions about the user's documents or personal details, use the passages
when they contain the answer. If they do not, say "I don't have enough
information to answer this." Never invent personal or document-specific facts.
If a passage directly states an answer, answer it directly without unnecessary
prefaces or repetition.
Use each passage's Source label for attribution, but do not infer document
content from its filename alone.

For general-knowledge questions, use your general knowledge. Use retrieved
passages only when they are relevant; ignore unrelated passages and do not
claim that a general-knowledge answer came from a document.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")


def build_chain(config: dict):
    """
    Step 1: configure the Groq chat model.
    Step 2: connect prompt, model, and parser into one generation chain.
    """
    llm = ChatGroq(
        api_key=config["groq_api_key"],
        model=config["model_name"],
        temperature=config["llm_temperature"],
        max_tokens=config["llm_max_tokens"]
    )
    chain = PROMPT_TEMPLATE | llm | StrOutputParser()
    return chain


def generate(question: str, chunks: list, chain) -> str:
    """
    Combines retrieved chunks into context and calls the LLM chain.
    Returns: answer as plain string
    """
    # Retrieved chunks are supplied as context; the prompt guides how they are used.
    context = "\n\n---\n\n".join(chunks)
    answer  = chain.invoke({
        "context": context,
        "question": question
    })
    return answer
