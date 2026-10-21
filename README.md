# AdhivAI — Streamlit RAG workspace
# Phase 3 of the RAG Learning Project

---

## What We Built

A RAG (Retrieval-Augmented Generation) system with a shared document knowledge
base and chat workspace.
- Ask questions about shared PDF, Markdown, and TXT documents or general topics
- See retrieved passages and PDF page citations when available
- Upload documents and synchronize the shared ChromaDB index
- Restrict document deletion to the workspace owner
- Export the latest answer or the complete conversation as PDF or TXT
- Adjust the Groq model, temperature, and number of retrieved passages

The interface uses the **✦ AdhivAI** brand, a short RAG-system description,
status badges, and a welcome panel with three optional starter questions:
summarize document key points, summarize résumé experience and skills, or
explain a topic in simple terms. The footer credits the project author and
lists the main technologies.

---

## How to Run

```bash
cd streamlit_app_test
python -m streamlit run app.py
```

Opens automatically at: http://localhost:8501

## Owner-only document deletion

Documents in the knowledge base are shared and searchable by all users. Only
the workspace owner can delete them. Configure a strong owner password in the
Streamlit Cloud app's **Settings → Secrets**:

```toml
ADHIVAI_ADMIN_PASSWORD = "use-a-long-unique-password"
```

For local development, put the same setting in `.streamlit/secrets.toml` or
set it in the project's `.env` file. Never commit the password or secrets file.
In the app, open **Knowledge base → Owner controls** and enter the owner
password to unlock the delete controls. Each deletion requires a second
confirmation and the index is synchronized to remove the deleted document's
vectors.

This password gate protects deletion controls; it does not make documents
private. Everyone using the app can still search the shared knowledge base.

---

## How Streamlit renders the UI

Streamlit is a Python library that converts Python code into a real web page automatically.

You write this in Python:
```python
st.title("Your Knowledge. One Intelligent Conversation.")
question = st.chat_input("Ask about your documents or anything else...")
st.markdown(answer)
```

Streamlit converts it into a real webpage with:
- Proper fonts and styling
- Responsive layout
- Interactive components (buttons, inputs, file uploader)
- Chat bubbles (like ChatGPT)

The app uses Streamlit widgets and a small amount of CSS for its branded theme;
the RAG orchestration remains in the existing `core/` modules.

---

## How the UI Works — Component by Component

### 1. Page Config
```python
st.set_page_config(page_title="AdhivAI", layout="wide")
```
- Sets the browser tab title and uses the wide layout for the workspace

### 2. Workspace navigation
```python
with st.sidebar:
    st.button("New chat")
```
- The sidebar contains conversation actions, model settings, and AdhivAI
  branding
- The Knowledge base section manages uploads, synchronization, and documents
- The Chat section contains suggested prompts, chat history, and source context
- The welcome screen offers two document prompts and one general-knowledge
  prompt; these are suggestions, not limits on what users may ask
- Status badges show model readiness, shared document count, and index state
- Setup timings are available in the collapsed **Setup performance** section

### 3. Chat Messages
```python
with st.chat_message("user"):
    st.markdown(question)

with st.chat_message("assistant"):
    st.markdown(answer)
```
- `st.chat_message("user")` creates a distinct user message block
- `st.chat_message("assistant")` creates a separate assistant message block
- Streamlit manages the message layout and Markdown rendering

### 4. Chat Input Box
```python
question = st.chat_input("Ask a question...")
```
- Creates the text input bar pinned to the bottom of the page
- Returns the user's typed text when they press Enter

### 5. Spinner
```python
with st.spinner("Thinking..."):
    answer = generate(...)
```
- Shows a loading animation while the LLM is generating
- Disappears automatically when done

### 6. Session State
```python
st.session_state.messages = []
```
- Streamlit reruns the entire script on every user interaction
- `st.session_state` is a dictionary that persists data across reruns
- Used to store chat history, retriever, and chain so they are not rebuilt on every message

### 7. Chat exports

Ask AdhivAI to export the latest answer as a PDF or TXT, or explicitly request
the full/complete chat or all answers to export a chronological conversation
transcript. Full-chat exports include user questions, assistant responses, and
available source citations.

---

## Project Structure

```
streamlit_app/
├── app.py                  ← Main UI file (entry point)
├── core/
│   ├── indexing.py         ← Load docs + chunk + embed + store in ChromaDB
│   ├── retrieval.py        ← Search ChromaDB for relevant chunks
│   ├── embeddings.py       ← Share the loaded local embedding model
│   └── generation.py      ← Build prompt + call Groq LLM + return answer
├── documents/              ← Your knowledge base (.txt, .md, and .pdf)
├── chroma_db/              ← Auto-created, stores vector embeddings locally
├── index_tracker.json      ← Auto-created, tracks which files are indexed
├── .env                    ← Local config (API key, model, chunk size, admin password)
└── .gitignore              ← Prevents .env and chroma_db from being committed
```

---

## Full RAG Flow (What Happens on Every Question)

```
User types question in browser
        ↓
app.py receives question via st.chat_input()
        ↓
core/retrieval.py → retrieve(question, retriever, config)
    - Converts question to a vector using HuggingFaceEmbeddings
    - Searches ChromaDB and always returns the top-K matching chunks
    - Sends those chunks and their source metadata to answer generation
        ↓
core/generation.py → generate(question, chunks, chain)
    - Joins chunks into context block
    - Uses matching chunks for document-specific questions
    - Uses general knowledge when retrieved chunks do not help answer the question
    - Sends prompt to Groq LLM (ChatGroq)
    - Returns plain text answer
        ↓
app.py displays answer in chat bubble
app.py keeps retrieved chunks and their sources in a collapsed expander
app.py saves message to st.session_state.messages (chat history)
```

---

## What Happens on File Upload

```
User selects files in the Knowledge base and clicks Upload to knowledge base
        ↓
app.py saves the file to the configured documents/ folder
        ↓
st.session_state.retriever = None  (refreshes retrieval after index sync)
        ↓
st.rerun() — Streamlit reruns the script
        ↓
core/indexing.py → build_index()
    - Hashes current supported files and compares them with index_tracker.json
    - New or changed files → replace their old chunks and store current chunks
    - Deleted files → remove their chunks from ChromaDB
    - Unchanged files → skip embedding
    - Updates index_tracker.json after the index is synchronized
        ↓
New file is now searchable immediately
```

After an upload attempt, the file picker is cleared. To remove a document from
the app, the workspace owner unlocks **Owner controls** in the Knowledge base,
chooses **Delete**, and confirms. The app deletes the file and synchronizes the
index, removing its vectors. Other users can still search all documents in the
shared knowledge base but cannot delete them.

When files are changed directly in `documents/`, use **Sync / rebuild
knowledge base** in the Knowledge base section. The app also synchronizes the
index when it starts. Text-based PDFs are supported. Scanned PDFs containing
only images need OCR before their content can be searched.

The app does not apply a relevance-score cutoff. It always sends the top
`TOP_K` matching chunks (3 by default) to the LLM. For general-knowledge
questions, the LLM is instructed to ignore chunks unrelated to the question.
When a question includes an uploaded document's full name (without its file
extension), retrieval filters ChromaDB to that document and includes its
filename/page citation with each passage. If more than one uploaded file has
the same name, AdhivAI asks the user to clarify. Named-document questions still
use the configured `TOP_K` limit; they do not automatically summarize every
page of a long document.

---

## What Happens on App Restart

```
App restarts → st.session_state is wiped (always empty on restart)
        ↓
build_index() checks documents and synchronizes Chroma
        ↓
Content hashes and Chroma sources checked against the current document folder
        ↓
Unchanged files skip embedding; missing, changed, or deleted sources are repaired
        ↓
The local embedding model is loaded only if files need indexing
        ↓
App ready — chat history is cleared (new session)
```

The app loads the local embedding model when needed for indexing and retrieval.
The model is shared by both and is created only once per process. Setup
performance timings are available in a collapsed section. On the first run,
downloading/loading the model and embedding new files can take time; later
starts do not re-embed unchanged files.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| UI | Streamlit | Web interface without HTML/CSS/JS |
| Document Loading | LangChain TextLoader + PyPDFLoader | Read .txt, .md, and .pdf files |
| Chunking | RecursiveCharacterTextSplitter | Split docs into overlapping chunks |
| Embeddings | HuggingFaceEmbeddings (all-MiniLM-L6-v2) | Convert text to vectors locally |
| Vector Store | ChromaDB | Store and search vectors locally |
| LLM | Groq API (openai/gpt-oss-120b) | Generate answers from context |
| Prompt | LangChain ChatPromptTemplate | Structure the RAG prompt |
| Chain | LangChain LCEL (pipe operator) | Connect prompt → LLM → parser |
| Config | python-dotenv | Load settings from .env file |
| Incremental Index | index_tracker.json | Skip already indexed files |

---

## Key Concepts Learned

### RAG (Retrieval-Augmented Generation)
For questions about your documents, relevant passages are supplied to the LLM
so it can answer from your files. Unrelated general-knowledge questions are
answered from the model's knowledge instead.

### Embeddings
Text converted to a list of numbers (vector) that represents its meaning.
Similar meaning = similar numbers = found together in search.

### Vector Store (ChromaDB)
A database optimized for storing and searching vectors.
Finds semantically similar content, not just keyword matches.

### Incremental Indexing
Track supported files by content hash and chunk count. New and changed files
are re-embedded, deleted files are removed from Chroma, and unchanged files are
skipped. The app checks Chroma as well as the tracker so a manually cleared
index is rebuilt.

### Session State
Streamlit reruns the whole script on every interaction.
st.session_state persists objects (retriever, chain, chat history) across reruns.

### LCEL Pipe Operator
```python
chain = prompt | llm | parser
```
LangChain's way of connecting components.
Output of left becomes input of right — like a pipeline.

---

## What's Next — Phase 4

Deploy this app to Streamlit Cloud (free):
- Push code to GitHub
- Connect GitHub repo to Streamlit Cloud
- Get a permanent public URL like https://yourapp.streamlit.app
- Share with anyone, anywhere — no laptop needed
