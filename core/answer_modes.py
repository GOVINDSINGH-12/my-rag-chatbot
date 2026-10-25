"""Names and descriptions for the chat answer modes."""

DOCUMENTS_AND_AI = "Documents + AI"
KNOWLEDGE_BASE_ONLY = "Knowledge Base only"
AI_ONLY = "AI-only"

ANSWER_MODES = (
    DOCUMENTS_AND_AI,
    KNOWLEDGE_BASE_ONLY,
    AI_ONLY,
)

ANSWER_MODE_DESCRIPTIONS = {
    DOCUMENTS_AND_AI: (
        "Searches your knowledge base and can also use general AI knowledge."
    ),
    KNOWLEDGE_BASE_ONLY: (
        "Answers from retrieved passages in your knowledge base only."
    ),
    AI_ONLY: (
        "Answers with general AI knowledge without searching your documents."
    ),
}
