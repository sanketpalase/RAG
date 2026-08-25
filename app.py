"""
app.py
Streamlit POC UI.

Left sidebar : Step 1 (Load ChromaDB, automatic), upload files, Step 2
               (Run Ingestion button), temperature slider
Main area    : chat — Step 3 (search) + generation happen per question

Every stage prints to the terminal (where you ran `streamlit run app.py`)
via timing_utils.stage(), so you can watch what's loading and how long each
step takes.
"""
import os
import streamlit as st

import config
from rag_core import (
    load_chromadb,
    ingest_documents,
    search_similar,
    generate_answer,
    get_files_in_folder,
)

st.set_page_config(page_title="RAG POC", layout="wide")


GREETING_RESPONSES = {
    "hi", "hello", "hey", "hiya", "good morning", "good afternoon",
    "good evening", "how are you", "how are you?",
}
DEFAULT_GREETING = "Hello! Upload a document and ask me a question about it."


def is_greeting(text: str) -> bool:
    return text.strip().lower() in GREETING_RESPONSES


@st.cache_resource(show_spinner="Step 1: Loading ChromaDB (persist_dir, cosine similarity)...")
def get_collection():
    """STEP 1 — load_chromadb() runs exactly once per server process (cached),
    kept deliberately separate from ingestion and search below."""
    _, collection = load_chromadb()
    return collection


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": DEFAULT_GREETING,
        "sources": [],
    }]

if "ingestion_result" not in st.session_state:
    st.session_state.ingestion_result = None

try:
    collection = get_collection()
except Exception as error:
    st.error(f"Unable to load ChromaDB: {type(error).__name__}: {error}")
    st.stop()


# ================= Sidebar =================
with st.sidebar:
    st.title("💬 RAG POC")
    st.caption("PDF / DOCX / JSON / TXT / images (OCR) → HF transformer embeddings → "
           "ChromaDB (cosine) → Gemini Flash")
    st.divider()
    st.header("1️⃣ ChromaDB")
    st.caption(f"persist_dir: `{config.PERSIST_DIR}`")
    st.caption(f"collection: `{config.COLLECTION_NAME}` (cosine similarity)")
    try:
        chunk_count = collection.count()
        st.metric("Chunks in DB", chunk_count)
    except Exception as error:
        st.metric("Chunks in DB", " unavailable")
        st.error(f"Unable to read ChromaDB count: {type(error).__name__}: {error}")

    if st.session_state.ingestion_result:
        result = st.session_state.ingestion_result
        st.success(f"Ingested {result['files_processed']} file(s), {result['chunks_added']} chunk(s) added.")
        if result["errors"]:
            st.error("Some files failed:\n" + "\n".join(result["errors"]))
        st.session_state.ingestion_result = None

    st.divider()
    st.header("2️⃣ Ingest")

    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / JSON / TXT / images",
        type=["pdf", "docx", "json", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "tif"],
        accept_multiple_files=True,
        help="Images are OCR'd with pytesseract — no vision/multimodal model.",
    )
    if uploaded_files:
        try:
            for uf in uploaded_files:
                with open(os.path.join(config.UPLOAD_FOLDER, uf.name), "wb") as file:
                    file.write(uf.getbuffer())
            st.success(f"Saved {len(uploaded_files)} file(s) — click Run Ingestion.")
        except Exception as error:
            st.error(f"Unable to save uploaded file(s): {type(error).__name__}: {error}")

    st.caption(f"Bulk-drop folder: `{config.DOCS_FOLDER}`")

    if st.button("🚀 Run Ingestion", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(fname, current, total):
            progress_bar.progress(current / max(total, 1))
            status_text.text(f"Processing {fname} ({current}/{total})")

        try:
            files = get_files_in_folder(config.DOCS_FOLDER) + get_files_in_folder(config.UPLOAD_FOLDER)
            if not files:
                raise ValueError("No supported documents found to ingest.")

            with st.spinner("Ingesting (loading + chunking + embedding + writing)..."):
                stats = ingest_documents(collection, files, progress_callback=on_progress)
        except Exception as error:
            stats = {
                "files_processed": 0,
                "chunks_added": 0,
                "errors": [f"{type(error).__name__}: {error}"],
            }
        finally:
            progress_bar.empty()
            status_text.empty()

        st.session_state.ingestion_result = stats
        st.rerun()

    st.divider()
    st.header("3️⃣ Chat settings")
    temperature = st.slider("Temperature", 0.0, 1.0, config.DEFAULT_TEMPERATURE, 0.05)
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, config.TOP_K, 1)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": DEFAULT_GREETING,
            "sources": [],
        }]
        st.rerun()


# ================= Main area =================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

user_question = st.chat_input("Ask a question about your ingested documents...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if is_greeting(user_question):
            answer = "Hello! Ask me anything about your ingested documents."
            sources = []
        else:
            with st.spinner("Searching + generating..."):
                try:
                    if collection.count() == 0:
                        answer = ("No documents ingested yet — upload files and click "
                                  "**Run Ingestion** in the sidebar first.")
                        sources = []
                    else:
                        history_for_api = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                            if m["role"] in ("user", "assistant")
                        ]
                        # STEP 3: search (separate call) -> generation (separate call)
                        results = search_similar(collection, user_question, top_k=top_k)
                        answer, sources = generate_answer(
                            user_question, results,
                            chat_history=history_for_api, temperature=temperature,
                        )
                except Exception as error:
                    answer = (
                        f"I couldn't process that request ({type(error).__name__}: {error}). "
                        "Please check the terminal logs and try again."
                    )
                    sources = []
        st.markdown(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
