import hmac
import os
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.generation import build_chain, generate
from core.embeddings import get_embeddings
from core.exporting import create_export_file
from core.indexing import SUPPORTED_EXTENSIONS, build_index
from core.retrieval import load_retriever, retrieve


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CONFIG = {
    "groq_api_key": os.getenv("GROQ_API_KEY"),
    "model_name": os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b"),
    "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", "3000")),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
    "top_k": int(os.getenv("TOP_K", "3")),
    "chroma_db_path": os.path.join(
        BASE_DIR, os.getenv("CHROMA_DB_PATH", "chroma_db")
    ),
    "collection_name": os.getenv("CHROMA_COLLECTION_NAME", "streamlit_rag"),
    "documents_folder": os.path.join(
        BASE_DIR, os.getenv("DOCUMENTS_FOLDER", "documents")
    ),
}

st.set_page_config(
    page_title="AdhivAI",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f3fa;
        color: #292449;
    }

    [data-testid="stCaptionContainer"] {
        color: #514d63 !important;
        opacity: 1;
    }

    .block-container {
        max-width: 1680px;
        padding-top: 2.3rem;
        padding-bottom: 5rem;
    }

    section[data-testid="stSidebar"] {
        background: #e8edf5;
        border-right: 1px solid #d1d9e6;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #20243a !important;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        opacity: 1;
    }

    section[data-testid="stSidebar"] input {
        background: #ffffff;
        color: #20243a;
    }

    section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
        background: #f8f9fc;
        border-color: #c6cfdd;
        color: #20243a;
    }

    section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
        background: #ffffff;
        border-color: #8172dc;
        color: #292449;
    }

    .adhiv-brand {
        align-items: center;
        color: #292449;
        display: flex;
        font-size: 1.35rem;
        font-weight: 700;
        gap: 0.55rem;
        letter-spacing: -0.03em;
    }

    .adhiv-brand-icon {
        align-items: center;
        background: #efedff;
        border-radius: 0.7rem;
        color: #5b4bc4;
        display: inline-flex;
        font-size: 1.1rem;
        height: 2.1rem;
        justify-content: center;
        width: 2.1rem;
    }

    .adhiv-eyebrow {
        color: #6a5bc2;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }

    .adhiv-hero {
        margin: 0.25rem 0 1.5rem;
        text-align: center;
    }

    .adhiv-hero-title {
        color: #292449;
        font-size: clamp(1.9rem, 3vw, 2.6rem);
        font-weight: 750;
        letter-spacing: -0.04em;
        line-height: 1.15;
    }

    .adhiv-hero-subtitle {
        color: #514d63;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    .st-key-welcome-panel {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid #e1deef;
        border-radius: 1rem;
        padding: 1rem 1.1rem;
    }

    .st-key-example_prompt_0 button,
    .st-key-example_prompt_1 button,
    .st-key-example_prompt_general button {
        background: #ffffff;
        border: 1px solid #ded9f2;
        color: #332b62;
        justify-content: flex-start;
        min-height: 3.2rem;
        text-align: left;
        white-space: normal;
    }

    .st-key-example_prompt_0 button:hover,
    .st-key-example_prompt_1 button:hover,
    .st-key-example_prompt_general button:hover {
        background: #f2f0ff;
        border-color: #8172dc;
        color: #292449;
    }

    .stButton > button {
        border-radius: 0.7rem;
        font-weight: 600;
        min-height: 2.65rem;
    }

    .stButton > button:focus-visible,
    [data-testid="stChatInput"] textarea:focus-visible,
    input:focus-visible {
        outline: 3px solid #6a5bc2 !important;
        outline-offset: 2px;
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #5b4bc4;
        border-color: #5b4bc4;
        color: #ffffff;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #4d40aa;
        border-color: #4d40aa;
        color: #ffffff;
    }

    .block-container h1 {
        font-size: clamp(1.65rem, 2.4vw, 2.2rem);
        letter-spacing: -0.035em;
    }

    .st-key-workspace-navigation {
        background: #eceaf8;
        border: 1px solid #d9d5ee;
        border-radius: 0.9rem;
        margin-top: 0.65rem;
        padding: 0.35rem 0.65rem;
    }

    .st-key-workspace-navigation [data-testid="stSegmentedControl"] {
        overflow: visible;
    }

    .st-key-workspace-navigation [data-testid="stSegmentedControl"] button {
        min-height: 2.75rem;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }

    .st-key-workspace-navigation
        [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background: #5b4bc4;
        color: #ffffff;
    }

    .adhiv-footer {
        border-top: 1px solid #cbd4e1;
        color: #3c4559;
        font-size: 0.78rem;
        line-height: 1.65;
        margin-top: 1.5rem;
        padding-top: 1rem;
    }

    .adhiv-footer-title {
        color: #292449;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .adhiv-footer-stack {
        color: #514d63;
        margin-top: 0.2rem;
    }

    .adhiv-footer-signature {
        color: #292449;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-top: 0.35rem;
    }

    .adhiv-footer-signature-name {
        color: #292449;
    }

    [data-testid="stChatInput"] [data-baseweb="textarea"] {
        background: #ffffff;
        border: 1px solid #dedbea;
        border-radius: 1rem;
        box-shadow: 0 5px 20px rgba(38, 32, 73, 0.07);
    }

    [data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
        border-color: #8172dc;
        box-shadow: 0 0 0 2px rgba(91, 75, 196, 0.13);
    }

    [data-testid="stChatInput"] textarea {
        color: #292449;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #c9c4e5;
        border-radius: 0.9rem;
        padding: 0.45rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 0.9rem;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3.5rem;
        }

        .adhiv-hero {
            margin-bottom: 1rem;
        }

        .adhiv-hero-subtitle {
            font-size: 0.92rem;
        }

        .st-key-welcome-panel {
            padding: 0.8rem;
        }

        .stButton > button {
            min-height: 2.9rem;
            white-space: normal;
        }

        [data-testid="stChatInput"] textarea {
            font-size: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "retriever": None,
        "chain": None,
        "index_status": "not_checked",
        "index_error": None,
        "index_warnings": [],
        "setup_timings": {},
        "chain_signature": None,
        "chain_error": None,
        "upload_notice": None,
        "upload_errors": [],
        "document_uploader_version": 0,
        "admin_authenticated": False,
        "admin_login_error": None,
        "pending_document_deletion": None,
        "active_model_name": CONFIG["model_name"],
        "active_temperature": min(max(CONFIG["llm_temperature"], 0.0), 2.0),
        "active_top_k": max(CONFIG["top_k"], 1),
        "draft_temperature": min(max(CONFIG["llm_temperature"], 0.0), 2.0),
        "draft_top_k": max(CONFIG["top_k"], 1),
        "model_draft": CONFIG["model_name"],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_error(exc: Exception, secret: str | None = None) -> str:
    detail = str(exc).strip()
    if secret:
        detail = detail.replace(secret, "[redacted]")
    detail = detail[:500]
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _configured_admin_password() -> str | None:
    password = os.getenv("ADHIVAI_ADMIN_PASSWORD")
    if password:
        return password
    try:
        password = st.secrets["ADHIVAI_ADMIN_PASSWORD"]
    except (FileNotFoundError, KeyError):
        return None
    return password if isinstance(password, str) and password else None


def _authenticate_admin() -> None:
    expected_password = _configured_admin_password()
    entered_password = st.session_state.get("admin_password_input", "")
    if expected_password and hmac.compare_digest(
        entered_password, expected_password
    ):
        st.session_state.admin_authenticated = True
        st.session_state.admin_login_error = None
    else:
        st.session_state.admin_authenticated = False
        st.session_state.admin_login_error = "The owner password is incorrect."
    st.session_state.admin_password_input = ""


def _list_documents(folder: str) -> tuple[list[dict], str | None]:
    documents = []
    try:
        for filename in sorted(os.listdir(folder), key=str.casefold):
            path = os.path.join(folder, filename)
            if (
                os.path.isfile(path)
                and Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = None
                documents.append(
                    {
                        "name": filename,
                        "extension": Path(filename).suffix.lower(),
                        "size": size,
                    }
                )
    except OSError as exc:
        return [], _safe_error(exc)
    return documents, None


def _format_size(size: int | None) -> str:
    if size is None:
        return "Size unavailable"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _save_uploaded_files(
    uploaded_files: list, folder: str
) -> tuple[list[str], list[str], list[str]]:
    saved = []
    unchanged = []
    errors = []
    allowed_extensions = {extension.lower() for extension in SUPPORTED_EXTENSIONS}

    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as exc:
        return [], [], [f"Could not access the documents folder: {_safe_error(exc)}"]

    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name.replace("\\", "/")).name
        extension = Path(filename).suffix.lower()
        if not filename or extension not in allowed_extensions:
            errors.append(f"{filename or 'Unnamed file'}: unsupported file type.")
            continue

        destination_path = os.path.join(folder, filename)
        try:
            content = uploaded_file.getvalue()
            existing_content = None
            if os.path.isfile(destination_path):
                with open(destination_path, "rb") as existing_file:
                    existing_content = existing_file.read()
            if content == existing_content:
                unchanged.append(filename)
                continue
            with open(destination_path, "wb") as destination:
                destination.write(content)
            saved.append(filename)
        except OSError as exc:
            errors.append(f"{filename}: {_safe_error(exc)}")

    return saved, unchanged, errors


def _delete_document_file(filename: str, folder: str) -> tuple[bool, str | None]:
    if not filename:
        return False, "No file name was provided for deletion."

    target_path = os.path.join(folder, Path(filename).name)
    if not os.path.isfile(target_path):
        return False, f"{filename} was not found in the knowledge base."

    try:
        os.remove(target_path)
    except OSError as exc:
        return False, f"Could not delete {filename}: {_safe_error(exc)}"

    return True, None


def _render_sources(chunk_details: list[dict]) -> None:
    if not chunk_details:
        return

    with st.expander(
        f"Sources and retrieved context · {len(chunk_details)}",
        icon=":material/source:",
    ):
        st.caption("Passages returned by the document retriever for this answer.")
        for index, chunk in enumerate(chunk_details, start=1):
            st.markdown(f"**Chunk {index}**")
            citation = chunk.get("citation") or chunk.get("source")
            if citation:
                st.caption(f"Source: {citation}")
            text = chunk.get("text")
            if text:
                st.text(text)
            if index < len(chunk_details):
                st.divider()


def _requested_export_format(question: str) -> str | None:
    """Recognize explicit requests to export chat content as TXT or PDF."""
    normalized_question = question.casefold()
    asks_to_create_file = re.search(
        r"\b(generate|create|make|export|save|download|convert|turn|put)\b",
        normalized_question,
    )
    asks_for_pdf = re.search(r"\bpdf\b|\.pdf\b", normalized_question)
    asks_for_text = re.search(
        r"\btxt\b|\.txt\b|\btext file\b", normalized_question
    )

    if not asks_to_create_file or not (asks_for_pdf or asks_for_text):
        return None
    return "pdf" if asks_for_pdf else "txt"


def _requests_full_conversation_export(question: str) -> bool:
    normalized_question = question.casefold()
    return bool(
        re.search(
            r"\b(?:all|entire|complete|whole)\s+(?:the\s+)?"
            r"(?:chat|conversation|chat history|conversation history|"
            r"messages|answers|responses)\b",
            normalized_question,
        )
        or re.search(
            r"\b(?:all|every)\s+(?:the\s+)?"
            r"(?:assistant\s+)?(?:answers?|responses?|replies)\b",
            normalized_question,
        )
    )


def _latest_export_content(messages: list[dict]) -> str | None:
    """Use the latest successful answer, preserving its text across file requests."""
    for message in reversed(messages):
        if (
            message.get("role") == "assistant"
            and not message.get("is_error")
            and not message.get("is_file_notice")
        ):
            return message.get("export_content", message["content"])
    return None


def _conversation_export_content(messages: list[dict]) -> str | None:
    """Format chat messages as a chronological transcript for file export."""
    transcript = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        if message.get("is_file_notice") or message.get("generated_file"):
            continue

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        speaker = "You" if role == "user" else "AdhivAI"
        transcript.append(f"{speaker}:\n{content.strip()}")

        if role == "assistant":
            citations = list(
                dict.fromkeys(
                    chunk.get("citation") or chunk.get("source")
                    for chunk in message.get("chunk_details", [])
                    if chunk.get("citation") or chunk.get("source")
                )
            )
            if citations:
                transcript.append(f"Sources: {', '.join(citations)}")

    return "\n\n".join(transcript) if transcript else None


def _adjust_stepper(
    key: str,
    delta: int | float,
    minimum: int | float,
    maximum: int | float,
    step: int | float,
) -> None:
    updated = min(maximum, max(minimum, st.session_state[key] + delta))
    st.session_state[key] = round(updated, 1) if isinstance(step, float) else updated


def _render_stepper(
    label: str,
    key: str,
    minimum: int | float,
    maximum: int | float,
    step: int | float,
    help_text: str,
) -> None:
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.caption(help_text)
        minus, value, plus = st.columns([1, 1.2, 1], vertical_alignment="center")
        current = st.session_state[key]

        minus.button(
            "−",
            key=f"{key}_minus",
            disabled=current <= minimum,
            help=f"Decrease {label.lower()}",
            width="stretch",
            on_click=_adjust_stepper,
            args=(key, -step, minimum, maximum, step),
        )
        current = st.session_state[key]
        value.markdown(f"### {current:g}", text_alignment="center")
        plus.button(
            "+",
            key=f"{key}_plus",
            disabled=current >= maximum,
            help=f"Increase {label.lower()}",
            width="stretch",
            on_click=_adjust_stepper,
            args=(key, step, minimum, maximum, step),
        )


_initialize_session_state()

try:
    os.makedirs(CONFIG["documents_folder"], exist_ok=True)
except OSError as exc:
    st.error(f"Could not access the documents folder: {_safe_error(exc)}")

document_files, documents_error = _list_documents(CONFIG["documents_folder"])
question_count = sum(
    message.get("role") == "user" for message in st.session_state.messages
)

with st.sidebar:
    st.markdown(
        """
        <div class="adhiv-brand">
            <span class="adhiv-brand-icon">✦</span>
            <span>AdhivAI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("🔗 Your knowledge, thoughtfully connected.")
    st.caption(
        f"📚 {len(document_files)} documents  ·  "
        f"💬 {question_count} questions in this chat"
    )

    if st.button(
        "New chat",
        icon=":material/add_comment:",
        type="primary",
        width="stretch",
    ):
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "Clear conversation",
        icon=":material/delete_sweep:",
        width="stretch",
    ):
        st.session_state.messages = []
        st.rerun()

    with st.expander("⚙️ settings", expanded=False):
        st.caption("Adjust the response style and retrieved evidence.")
        st.text_input("Groq model", key="model_draft")
        _render_stepper(
            "🌡️ Temperature",
            "draft_temperature",
            0.0,
            2.0,
            0.1,
            "Lower values give more consistent answers.",
        )
        _render_stepper(
            "🔎 Retrieved passages",
            "draft_top_k",
            1,
            max(10, CONFIG["top_k"]),
            1,
            "Passages retrieved for each question.",
        )
        if st.button(
            "Apply settings",
            key="apply_settings",
            type="primary",
            width="stretch",
        ):
            model_name = st.session_state.model_draft.strip()
            if not model_name:
                st.error("Enter a Groq model name before applying settings.")
            else:
                model_changed = model_name != st.session_state.active_model_name
                temperature_changed = (
                    st.session_state.draft_temperature
                    != st.session_state.active_temperature
                )
                top_k_changed = (
                    st.session_state.draft_top_k != st.session_state.active_top_k
                )
                st.session_state.active_model_name = model_name
                st.session_state.active_temperature = (
                    st.session_state.draft_temperature
                )
                st.session_state.active_top_k = st.session_state.draft_top_k
                if model_changed or temperature_changed:
                    st.session_state.chain = None
                    st.session_state.chain_signature = None
                    st.session_state.chain_error = None
                if top_k_changed:
                    st.session_state.retriever = None
                st.toast("Settings applied to new questions.")

        st.caption(
            f"Active: {st.session_state.active_model_name} · "
            f"temperature {st.session_state.active_temperature:.1f} · "
            f"{st.session_state.active_top_k} passages"
        )
        if (
            st.session_state.model_draft.strip()
            != st.session_state.active_model_name
            or st.session_state.draft_temperature
            != st.session_state.active_temperature
            or st.session_state.draft_top_k != st.session_state.active_top_k
        ):
            st.caption("Settings changed · click Apply settings to use them.")

    st.markdown(
        """
        <div class="adhiv-footer">
            <div class="adhiv-footer-title">
                ✦ AI-Powered RAG System for Document Q&amp;A and Knowledge Exploration
            </div>
            <div class="adhiv-footer-stack">
                ⚙️ Powered by Python · LangChain · ChromaDB · Streamlit
            </div>
            <div class="adhiv-footer-signature">
                👨‍💻 Built by
                <span class="adhiv-footer-signature-name">Govind Singh Adhikari</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="adhiv-hero">
        <div class="adhiv-hero-title">✦ AdhivAI</div>
        <div class="adhiv-hero-subtitle">
            💬 Ask freely. AdhivAI answers using your documents and its own general knowledge. ✨
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

active_config = {
    **CONFIG,
    "model_name": st.session_state.active_model_name,
    "llm_temperature": st.session_state.active_temperature,
    "top_k": st.session_state.active_top_k,
}

chain_signature = (
    active_config["model_name"],
    active_config["llm_temperature"],
)
needs_chain_setup = st.session_state.chain_signature != chain_signature
needs_index_sync = st.session_state.index_status in {"not_checked", "needs_sync"}
setup_placeholder = st.empty()
setup_status = None
if needs_chain_setup or needs_index_sync:
    with setup_placeholder.container():
        setup_status = st.status(
            "⏳ Loading the AdhivAI workspace",
            expanded=True,
        )
        setup_tasks = []
        if needs_chain_setup and CONFIG["groq_api_key"]:
            setup_tasks.append("preparing the answer model")
        if needs_index_sync:
            setup_tasks.append("syncing your knowledge base")
        if setup_tasks:
            setup_status.write(
                "Getting AdhivAI ready: " + " and ".join(setup_tasks) + "."
            )
        else:
            setup_status.write("Checking the app configuration.")

if needs_chain_setup:
    st.session_state.chain = None
    st.session_state.chain_error = None
    st.session_state.chain_signature = chain_signature
    if not CONFIG["groq_api_key"]:
        st.session_state.chain_error = (
            "GROQ_API_KEY is not configured in the project's .env file."
        )
    else:
        try:
            with st.spinner("Preparing AdhivAI with your current settings..."):
                st.session_state.chain = build_chain(active_config)
        except Exception as exc:
            st.session_state.chain_error = (
                "Could not initialize the answer model: "
                + _safe_error(exc, CONFIG["groq_api_key"])
            )

# Keep the index ready for document questions even if the user stays on Chat.
if needs_index_sync:
    st.session_state.index_status = "syncing"
    st.session_state.index_warnings = []
    try:
        with st.spinner("Preparing AdhivAI and updating your knowledge base..."):
            # Load the local embedding model before the first question needs it.
            # This shifts its cold-start cost into workspace preparation.
            embedding_started = time.perf_counter()
            get_embeddings(active_config["embedding_model"])
            st.session_state.setup_timings["embedding_load_seconds"] = (
                time.perf_counter() - embedding_started
            )

            index_started = time.perf_counter()
            _, newly_indexed, files_without_text = build_index(active_config)
            st.session_state.setup_timings["index_sync_seconds"] = (
                time.perf_counter() - index_started
            )

            st.session_state.index_status = "ready"
            st.session_state.index_error = None
            st.session_state.index_warnings = files_without_text
            # Keep the Chroma connection ready so the first search only embeds
            # the question and queries the already-open collection.
            retriever_started = time.perf_counter()
            st.session_state.retriever = load_retriever(active_config)
            st.session_state.setup_timings["retriever_setup_seconds"] = (
                time.perf_counter() - retriever_started
            )
            if newly_indexed:
                st.toast(f"Indexed or updated {len(newly_indexed)} file(s).")
    except Exception as exc:
        st.session_state.index_status = "error"
        st.session_state.index_error = _safe_error(exc, CONFIG["groq_api_key"])
        st.session_state.retriever = None

# Applying a different retrieval count clears the per-session retriever. Rebuild
# it here rather than making the next question pay the setup cost.
if (
    st.session_state.index_status == "ready"
    and st.session_state.retriever is None
):
    try:
        retriever_started = time.perf_counter()
        st.session_state.retriever = load_retriever(active_config)
        st.session_state.setup_timings["retriever_setup_seconds"] = (
            time.perf_counter() - retriever_started
        )
    except Exception as exc:
        st.session_state.index_status = "error"
        st.session_state.index_error = _safe_error(exc, CONFIG["groq_api_key"])

if setup_status is not None:
    setup_failed = bool(
        (needs_chain_setup and st.session_state.chain_error)
        or (needs_index_sync and st.session_state.index_error)
    )
    if setup_failed:
        setup_status.update(
            label="AdhivAI setup finished with an issue",
            state="error",
            expanded=True,
        )
    else:
        setup_placeholder.empty()

model_status = (
    "Model ready"
    if st.session_state.chain is not None
    else "Model unavailable"
    if st.session_state.chain_error
    else "Model loading"
)
index_status_labels = {
    "ready": "index synced",
    "error": "index issue",
    "syncing": "indexing",
    "needs_sync": "index sync needed",
    "not_checked": "index starting",
}
model_status_color = (
    "green"
    if st.session_state.chain is not None
    else "red"
    if st.session_state.chain_error
    else "gray"
)
index_status_color = {
    "ready": "green",
    "error": "red",
    "syncing": "orange",
    "needs_sync": "orange",
    "not_checked": "gray",
}.get(st.session_state.index_status, "gray")
status_columns = st.columns(3, gap="small")
status_columns[0].badge(
    model_status,
    icon=":material/smart_toy:",
    color=model_status_color,
)
status_columns[1].badge(
    f"{len(document_files)} shared documents",
    icon=":material/library_books:",
    color="blue",
)
status_columns[2].badge(
    index_status_labels.get(st.session_state.index_status, "index status unknown").title(),
    icon=(
        ":material/check_circle:"
        if st.session_state.index_status == "ready"
        else ":material/error:"
        if st.session_state.index_status == "error"
        else ":material/sync:"
    ),
    color=index_status_color,
)

if st.session_state.setup_timings:
    timings = st.session_state.setup_timings
    with st.expander("Setup performance", expanded=False):
        st.caption(
            f"Embedding model {timings.get('embedding_load_seconds', 0.0):.2f}s · "
            f"index sync {timings.get('index_sync_seconds', 0.0):.2f}s · "
            f"retriever setup {timings.get('retriever_setup_seconds', 0.0):.2f}s"
        )

with st.container(key="workspace-navigation", border=True):
    selected_workspace = st.segmented_control(
        "Workspace section",
        options=["Chat", "Knowledge base"],
        format_func=lambda section: (
            "🤖 Chat" if section == "Chat" else "📚 Knowledge base"
        ),
        default="Chat",
        key="selected_workspace",
        label_visibility="collapsed",
        width="content",
    )

if selected_workspace == "Knowledge base":
    st.markdown(
        '<div class="adhiv-eyebrow">Your workspace</div>',
        unsafe_allow_html=True,
    )
    st.title("📚 Knowledge base")
    st.caption("Add documents, review sync status, and tune how AdhivAI answers.")

    upload_column, status_column = st.columns([1.35, 1], gap="large")
    with upload_column:
        st.subheader("Upload documents")
        st.caption("Select files first, then upload them to your knowledge base.")
        uploaded_files = st.file_uploader(
            "Select document(s)",
            type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            help="Text-based PDFs are supported. Scanned PDFs require OCR.",
            key=f"document_uploader_{st.session_state.document_uploader_version}",
        )
        if st.button(
            "Upload to knowledge base",
            icon=":material/upload_file:",
            type="primary",
            width="stretch",
        ):
            if not uploaded_files:
                st.info("Choose at least one file before uploading.")
            else:
                with st.spinner("Uploading your selected documents..."):
                    saved, unchanged, upload_errors = _save_uploaded_files(
                        uploaded_files, CONFIG["documents_folder"]
                    )
                if saved:
                    st.session_state.index_status = "needs_sync"
                    st.session_state.index_error = None
                    st.session_state.retriever = None
                if saved or unchanged:
                    summary = []
                    if saved:
                        summary.append(f"Added or replaced {len(saved)} file(s)")
                    if unchanged:
                        summary.append(f"{len(unchanged)} already up to date")
                    st.session_state.upload_notice = ". ".join(summary) + "."
                st.session_state.upload_errors = upload_errors
                st.session_state.document_uploader_version += 1
                st.rerun()

    with status_column:
        st.subheader("Knowledge-base status")
        if st.session_state.index_status == "ready":
            if document_files:
                st.badge(
                    "Index ready",
                    icon=":material/check_circle:",
                    color="green",
                )
                st.caption("The index was synchronized successfully.")
            else:
                st.badge(
                    "Synced · empty",
                    icon=":material/info:",
                    color="blue",
                )
                st.caption("No documents indexed yet; general questions still work.")
        elif st.session_state.index_status == "error":
            st.badge("Sync failed", icon=":material/error:", color="red")
            st.error(f"Indexing failed: {st.session_state.index_error}")
        else:
            st.badge("Needs sync", icon=":material/schedule:", color="orange")

        with st.expander("Owner controls", expanded=False):
            admin_password = _configured_admin_password()
            if not admin_password:
                st.warning(
                    "Document deletion is disabled until "
                    "`ADHIVAI_ADMIN_PASSWORD` is configured in Streamlit secrets."
                )
            elif st.session_state.admin_authenticated:
                st.success("Owner access is unlocked for this session.")
                if st.button(
                    "Lock owner controls",
                    key="lock_admin_controls",
                    width="stretch",
                ):
                    st.session_state.admin_authenticated = False
                    st.session_state.pending_document_deletion = None
                    st.rerun()
            else:
                with st.form("owner_login_form"):
                    st.text_input(
                        "Owner password",
                        type="password",
                        key="admin_password_input",
                    )
                    st.form_submit_button(
                        "Unlock delete controls",
                        type="primary",
                        width="stretch",
                        on_click=_authenticate_admin,
                    )
                if st.session_state.admin_login_error:
                    st.error(st.session_state.admin_login_error)

        if st.button(
            "Sync / rebuild knowledge base",
            icon=":material/sync:",
            width="stretch",
        ):
            st.session_state.index_status = "needs_sync"
            st.session_state.retriever = None
            st.rerun()

    document_files, documents_error = _list_documents(CONFIG["documents_folder"])
    with st.expander(f"📄 Documents ({len(document_files)})", expanded=False):
        if documents_error:
            st.error(f"Could not read the documents folder: {documents_error}")
        elif not document_files:
            st.info("Your document list is empty. Upload a supported file above.")
        else:
            if not st.session_state.admin_authenticated:
                st.caption("Only the workspace owner can delete shared documents.")
            for document in document_files:
                icon = (
                    "picture_as_pdf"
                    if document["extension"] == ".pdf"
                    else "description"
                )
                with st.container(border=True):
                    document_row = st.columns([4.8, 1.2])
                    with document_row[0]:
                        st.markdown(f":material/{icon}: **{document['name']}**")
                        st.caption(
                            f"{document['extension'].lstrip('.').upper()} · "
                            f"{_format_size(document['size'])}"
                        )
                    with document_row[1]:
                        if (
                            st.session_state.admin_authenticated
                            and st.session_state.pending_document_deletion
                            != document["name"]
                            and st.button(
                                "Delete",
                                key=f"delete_document_{document['name']}",
                                type="secondary",
                                width="stretch",
                            )
                        ):
                            st.session_state.pending_document_deletion = document["name"]
                            st.rerun()

                    if (
                        st.session_state.admin_authenticated
                        and st.session_state.pending_document_deletion
                        == document["name"]
                    ):
                        st.warning(f"Delete **{document['name']}** permanently?")
                        confirm_column, cancel_column = st.columns(2)
                        if confirm_column.button(
                            "Confirm delete",
                            key=f"confirm_delete_document_{document['name']}",
                            type="primary",
                            width="stretch",
                        ):
                            deleted, delete_error = _delete_document_file(
                                document["name"], CONFIG["documents_folder"]
                            )
                            st.session_state.pending_document_deletion = None
                            if deleted:
                                st.session_state.index_status = "needs_sync"
                                st.session_state.index_error = None
                                st.session_state.retriever = None
                                st.session_state.upload_notice = (
                                    f"Deleted {document['name']} from the knowledge base."
                                )
                            else:
                                st.session_state.upload_errors = [
                                    delete_error
                                    or f"Could not delete {document['name']}."
                                ]
                            st.rerun()
                        if cancel_column.button(
                            "Cancel",
                            key=f"cancel_delete_document_{document['name']}",
                            width="stretch",
                        ):
                            st.session_state.pending_document_deletion = None
                            st.rerun()

    for upload_error in st.session_state.upload_errors:
        st.error(upload_error)
    st.session_state.upload_errors = []
    if st.session_state.upload_notice:
        st.success(st.session_state.upload_notice)
        st.session_state.upload_notice = None

    for filename in st.session_state.index_warnings:
        st.warning(
            f"No readable text was found in {filename}. "
            "Scanned PDFs need OCR before their content can be searched."
        )

else:
    st.markdown('<div class="adhiv-eyebrow">AdhivAI</div>', unsafe_allow_html=True)

    if st.session_state.chain_error:
        st.warning(st.session_state.chain_error)

    suggested_question = None
    if not st.session_state.messages:
        with st.container(border=True, key="welcome-panel"):
            st.markdown("### Start a conversation")
            st.caption(
                "Ask questions about the shared knowledge base, get a summary, "
                "or explore a topic using general knowledge."
            )
            st.markdown("**Get started with a question**")
            st.caption("About your documents")
            document_prompt_columns = st.columns(2, gap="small")
            document_questions = (
                ("Summarize the key points in my documents", ":material/summarize:"),
                (
                    "Summarize my experience and key skills from my résumé",
                    ":material/description:",
                ),
            )
            for index, (question_text, icon) in enumerate(document_questions):
                with document_prompt_columns[index]:
                    if st.button(
                        question_text,
                        key=f"example_prompt_{index}",
                        icon=icon,
                        width="stretch",
                    ):
                        suggested_question = question_text

            st.caption("General knowledge")
            general_question = "Explain a topic in simple terms"
            if st.button(
                general_question,
                key="example_prompt_general",
                icon=":material/lightbulb:",
                width="stretch",
            ):
                suggested_question = general_question

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(
            message["role"],
            avatar="🧑" if message["role"] == "user" else "🤖",
        ):
            if message.get("is_error"):
                st.error(message["content"])
            elif message.get("is_file_notice"):
                st.info(message["content"])
            else:
                st.markdown(message["content"])
            if message.get("retrieval_notice"):
                st.warning(message["retrieval_notice"])
            if message["role"] == "assistant":
                _render_sources(message.get("chunk_details") or [])
                generated_file = message.get("generated_file")
                if generated_file:
                    st.download_button(
                        f"Download {generated_file['file_name']}",
                        data=generated_file["data"],
                        file_name=generated_file["file_name"],
                        mime=generated_file["mime_type"],
                        key=f"download_generated_file_{message_index}",
                    )
                answer_timings = message.get("timings")
                if answer_timings:
                    st.caption(
                        f"Search & retrieval "
                        f"{answer_timings['retrieval_seconds']:.2f}s · "
                        f"LLM response "
                        f"{answer_timings['generation_seconds']:.2f}s"
                    )

    st.markdown("#### ✦ Ask AdhivAI...")
    typed_question = st.chat_input(
        "Type your question here",
        key="adhiv_chat_input",
    )
    question = suggested_question or typed_question

    if question and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            export_format = _requested_export_format(question)
            if export_format:
                try:
                    prior_messages = st.session_state.messages[:-1]
                    full_conversation = _requests_full_conversation_export(question)
                    if full_conversation:
                        export_content = _conversation_export_content(prior_messages)
                    else:
                        # A file request refers to the latest answer unless the
                        # user explicitly asks for the full conversation.
                        export_content = _latest_export_content(prior_messages)
                    if export_content is None:
                        if full_conversation:
                            response = (
                                "Have a conversation first, then request a TXT "
                                "or PDF export of the full chat."
                            )
                        else:
                            response = (
                                "Ask a question first, then request a TXT or PDF "
                                "export of my answer."
                            )
                        st.info(response)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "is_file_notice": True,
                            }
                        )
                    else:
                        file_data, file_name, mime_type = create_export_file(
                            export_content,
                            export_format,
                            is_conversation=full_conversation,
                        )
                        scope = (
                            "the full conversation"
                            if full_conversation
                            else "my previous answer"
                        )
                        response = f"Created **{file_name}** from {scope}."
                        st.markdown(response)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "export_content": export_content,
                                "generated_file": {
                                    "data": file_data,
                                    "file_name": file_name,
                                    "mime_type": mime_type,
                                },
                            }
                        )
                except Exception as exc:
                    error_message = (
                        "I couldn't create that file. "
                        + _safe_error(exc, CONFIG["groq_api_key"])
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "is_error": True,
                        }
                    )
            else:
                try:
                    chunks = []
                    sources = []
                    chunk_details = []
                    retrieval_notice = None
                    answer_timings = {
                        "retrieval_seconds": 0.0,
                        "generation_seconds": 0.0,
                    }
                    with st.spinner("AdhivAI is searching for your answer..."):
                        retrieval_started = time.perf_counter()
                        if st.session_state.index_status == "ready":
                            try:
                                if st.session_state.retriever is None:
                                    st.session_state.retriever = load_retriever(
                                        active_config
                                    )
                                chunks, sources, chunk_details = retrieve(
                                    question,
                                    st.session_state.retriever,
                                    active_config,
                                )
                            except Exception as exc:
                                st.session_state.retriever = None
                                retrieval_notice = (
                                    "The knowledge base could not be searched, so "
                                    "this answer uses general knowledge only. "
                                    + _safe_error(exc, CONFIG["groq_api_key"])
                                )
                        else:
                            retrieval_notice = (
                                "The knowledge base is not ready, so this answer "
                                "uses general knowledge only."
                            )
                            if st.session_state.index_error:
                                retrieval_notice += (
                                    f" Index status: "
                                    f"{st.session_state.index_error}"
                                )

                        answer_timings["retrieval_seconds"] = (
                            time.perf_counter() - retrieval_started
                        )

                        if st.session_state.chain is None:
                            raise RuntimeError(
                                st.session_state.chain_error
                                or "The answer model is not available."
                            )

                        generation_started = time.perf_counter()
                        answer = generate(question, chunks, st.session_state.chain)
                        answer_timings["generation_seconds"] = (
                            time.perf_counter() - generation_started
                        )

                    if retrieval_notice:
                        st.warning(retrieval_notice)
                    st.markdown(answer)
                    _render_sources(chunk_details)
                    st.caption(
                        f"Search & retrieval "
                        f"{answer_timings['retrieval_seconds']:.2f}s · "
                        f"LLM response "
                        f"{answer_timings['generation_seconds']:.2f}s"
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "chunk_details": chunk_details,
                            "retrieval_notice": retrieval_notice,
                            "timings": answer_timings,
                            "export_content": answer,
                        }
                    )
                except Exception as exc:
                    error_message = (
                        "I couldn't complete that answer. "
                        + _safe_error(exc, CONFIG["groq_api_key"])
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                            "chunk_details": [],
                            "is_error": True,
                        }
                    )
        st.rerun()
