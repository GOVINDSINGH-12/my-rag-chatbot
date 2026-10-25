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

# Step 1: Import the Groq chat model and LangChain prompt/chain helpers.
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from core.answer_modes import AI_ONLY, DOCUMENTS_AND_AI, KNOWLEDGE_BASE_ONLY


class GenerationChains:
    """Hold the answer chain and the follow-up-question rewrite chain."""

    def __init__(self, answer: Runnable, contextualize: Runnable) -> None:
        self.answer = answer
        self.contextualize = contextualize


# Step 2: Define prompts for clarifying follow-ups and generating grounded answers.
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_template("""
Rewrite the latest user question as a standalone search query using the chat
history only when needed to understand references or follow-ups.
If it is already clear on its own, keep its meaning and do not add unrelated
topics. Return only the standalone query, without an answer or explanation.

CHAT HISTORY:
{history}

LATEST QUESTION:
{question}

STANDALONE SEARCH QUERY:
""")

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are AdhivAI, a helpful assistant. Use the chat history to understand
follow-up questions and references to earlier messages. You are given the
user's latest question and the top matching passages retrieved from their
documents.

ANSWER MODE INSTRUCTIONS:
{mode_instructions}

Match the answer's length and detail to the request. Give short, direct answers
to simple factual questions. For requests to summarize or explain, provide an
appropriately detailed, organized answer using the available information.

For questions about the user's documents or personal details, use the retrieved
passages when the selected answer mode allows document use. Never treat earlier
assistant answers as proof of document-specific facts or invent personal or
document-specific facts.
If a passage directly states an answer, answer it directly without unnecessary
prefaces or repetition.
Use each passage's Source label for attribution, but do not infer document
content from its filename alone.

Follow the answer-mode instructions above. Do not use general knowledge in
Knowledge Base only mode, or document context in AI-only mode.

CHAT HISTORY:
{history}

CONTEXT:
{context}

LATEST QUESTION:
{question}

ANSWER:
""")


def build_chain(config: dict):
    """Create the Groq model and the two prompt-to-answer processing chains."""
    # Step 3: Create one model and connect it to each prompt/parser pipeline.
    llm = ChatGroq(
        api_key=config["groq_api_key"],
        model=config["model_name"],
        temperature=config["llm_temperature"],
        max_tokens=config["llm_max_tokens"]
    )
    return GenerationChains(
        answer=PROMPT_TEMPLATE | llm | StrOutputParser(),
        contextualize=CONTEXTUALIZE_PROMPT | llm | StrOutputParser(),
    )


def contextualize_question(
    question: str,
    conversation_history: list[dict[str, str]],
    chains: GenerationChains,
) -> str:
    """Turn a follow-up into a standalone query for document retrieval."""
    if not conversation_history:
        return question

    # Step 4: Use prior turns only to clarify the latest question.
    history = "\n".join(
        f"{'User' if message['role'] == 'user' else 'AdhivAI'}: "
        f"{message['content']}"
        for message in conversation_history
    )
    standalone_question = chains.contextualize.invoke(
        {"history": history, "question": question}
    ).strip()
    if not standalone_question:
        raise ValueError("The model returned an empty standalone search query.")
    return standalone_question


def generate(
    question: str,
    chunks: list,
    chains: GenerationChains,
    conversation_history: list[dict[str, str]] | None = None,
    answer_mode: str = DOCUMENTS_AND_AI,
) -> str:
    """
    Combines recent conversation and retrieved chunks to generate an answer.
    Returns: answer as plain string
    """
    # Step 5: Give the answer model recent chat context and retrieved passages.
    history = "\n".join(
        f"{'User' if message['role'] == 'user' else 'AdhivAI'}: "
        f"{message['content']}"
        for message in (conversation_history or [])
    )
    context = "\n\n---\n\n".join(chunks)
    mode_instructions = {
        DOCUMENTS_AND_AI: (
            "Use retrieved passages when relevant to the question. You may use "
            "general knowledge for general questions or when the documents do "
            "not cover a topic, and do not present general knowledge as a "
            "document fact."
        ),
        KNOWLEDGE_BASE_ONLY: (
            "Answer only with facts supported by the retrieved document "
            "passages. Do not use general knowledge to fill gaps. If the "
            "passages do not support an answer, say that the knowledge base "
            "does not contain enough information."
        ),
        AI_ONLY: (
            "Answer using general AI knowledge and the conversation history. "
            "Do not use or refer to document context, even if any is provided, "
            "and do not claim that the answer came from the knowledge base."
        ),
    }.get(answer_mode)
    if mode_instructions is None:
        raise ValueError(f"Unsupported answer mode: {answer_mode}")

    return chains.answer.invoke({
        "history": history,
        "mode_instructions": mode_instructions,
        "context": context,
        "question": question,
    })
