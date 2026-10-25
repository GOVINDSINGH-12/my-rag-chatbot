"""Small reusable Streamlit UI components."""

import streamlit as st

from core.answer_modes import ANSWER_MODE_DESCRIPTIONS, ANSWER_MODES


EXPLORE_TOPICS = (
    (
        "How does AdhivAI answer questions?",
        (
            "Ask a question in Chat, and AdhivAI searches your uploaded documents "
            "for relevant passages. It gives those passages to the language model "
            "to help write an answer, and shows the related sources when "
            "available. Choose **AI-only** to skip document search, or "
            "**Knowledge Base only** to keep the answer focused on retrieved "
            "document content."
        ),
    ),
    (
        "What are the answer modes?",
        (
            "**Documents + AI** searches the knowledge base and may also use "
            "general AI knowledge. **Knowledge Base only** answers from retrieved "
            "knowledge-base passages. **AI-only** answers with general AI "
            "knowledge without searching documents."
        ),
    ),
    (
        "What files can I upload?",
        (
            "You can upload TXT, Markdown (`.md`), text-based PDF, and Excel "
            "workbooks (`.xls` and `.xlsx`). Scanned PDFs need OCR before their "
            "text can be searched."
        ),
    ),
    (
        "How do I add or remove a document?",
        (
            "Open **Knowledge base** to upload documents or review the shared "
            "document list. To remove one, the workspace owner must unlock "
            "**Owner controls** with the configured owner password, then confirm "
            "the deletion. Deletion is permanent, and documents in this "
            "knowledge base are shared and searchable by all users."
        ),
    ),
    (
        "What if an answer is incomplete or seems wrong?",
        (
            "Check that the relevant document is uploaded and that the knowledge "
            "base is synchronized. Try asking a more specific question. For an "
            "answer based only on retrieved documents, choose **Knowledge Base "
            "only**; for general context as well, choose **Documents + AI**."
        ),
    ),
)


def render_answer_mode_selector() -> str:
    """Show the chat's answer-mode dropdown and its selected-mode description."""
    with st.container(key="answer-mode-panel", border=True):
        st.markdown("**:material/tune: ANSWER MODE**")
        answer_mode = st.selectbox(
            "Choose how AdhivAI answers",
            options=ANSWER_MODES,
            index=0,
            key="answer_mode",
            label_visibility="collapsed",
        )
        st.caption(ANSWER_MODE_DESCRIPTIONS[answer_mode])
    return answer_mode


def render_explore_adhivai() -> None:
    """Show a short interactive guide to AdhivAI's features."""
    st.markdown(
        '<div class="adhiv-eyebrow">Get to know your workspace</div>',
        unsafe_allow_html=True,
    )
    st.title("Explore AdhivAI", icon=":material/explore:")
    st.caption("Learn how it works and explore what you can do.")

    for question, answer in EXPLORE_TOPICS:
        with st.expander(question):
            st.markdown(answer)
