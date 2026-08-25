"""
rag_core.py
All the POC logic in one place, organized in the order things actually run:

  1. File loaders (PDF / DOCX / JSON / TXT / image-to-text via OCR) + a router
  2. Chunking (simple character sliding window — no tiktoken, no LangChain)
  3. Embeddings — raw HuggingFace `transformers` (AutoTokenizer + AutoModel +
      model pooler), NOT the sentence-transformers wrapper and NOT a
     multimodal model. This is the "advanced" part: it's exactly what
     sentence-transformers does internally, just written out by hand so you
     can see every step.
  4. ChromaDB — three DELIBERATELY SEPARATE steps, matching your ask:
       load_chromadb()   -> connect/create the persistent collection (cosine)
       ingest_documents() -> chunk + embed + write, called separately
       search_similar()   -> embed the query + cosine search, called separately
    5. Generation — a single Gemini call through `langchain-google-genai`.
"""
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import chromadb
import pytesseract
import torch
from docx import Document as DocxDocument
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer

import config
import prompts
from timing_utils import stage

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


# =========================================================================
# 1. FILE LOADERS — one function per type, plus a router that picks the
#    right one by extension (same pattern as the main project).
# =========================================================================

def load_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx_text(file_path: str) -> str:
    doc = DocxDocument(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_json_text(file_path: str) -> str:
    """Pretty-prints the JSON so it reads as plain text for embedding.
    Simple by design — for deeply nested files you may want to extract
    specific fields instead, but this keeps every value searchable."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=False)


def load_txt_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_image_text(file_path: str) -> str:
    """OCR only — no vision/multimodal model. Extracted text goes through
    the exact same text embedding pipeline as every other file type."""
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract OCR engine not found. Install it separately (the "
            "pytesseract pip package is just a wrapper) and set "
            "TESSERACT_CMD in .env if it's not on PATH."
        ) from e


LOADER_MAP = {
    ".pdf": load_pdf_text,
    ".docx": load_docx_text,
    ".json": load_json_text,
    ".txt": load_txt_text,
    ".png": load_image_text,
    ".jpg": load_image_text,
    ".jpeg": load_image_text,
    ".bmp": load_image_text,
    ".tiff": load_image_text,
    ".tif": load_image_text,
}


def load_file_text(file_path: str) -> str:
    """Router: extension -> loader function. Single place that decides
    'which file type -> which function'."""
    ext = Path(file_path).suffix.lower()
    loader_fn = LOADER_MAP.get(ext)
    if loader_fn is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_fn(file_path)


def load_files_parallel(file_paths: List[str], max_workers: int = None) -> dict:
    """Runs load_file_text() for every file concurrently (I/O + OCR bound,
    so threads are enough — no need for multiprocessing). Returns
    {file_path: text_or_Exception}, preserving errors per-file instead of
    letting one bad file abort the whole batch.

    Kept as a separate step from chunk/embed/write: ChromaDB writes and the
    embedding model call happen sequentially afterwards in
    ingest_documents(), only the raw file->text extraction is parallelized.
    """
    if not file_paths:
        return {}
    max_workers = max_workers or min(config.MAX_INGEST_WORKERS, len(file_paths))

    results = {}
    with stage(f"Loading {len(file_paths)} file(s) in parallel (workers={max_workers})"):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(load_file_text, fp): fp for fp in file_paths
            }
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    results[file_path] = future.result()
                except Exception as error:
                    results[file_path] = error
    return results


def get_files_in_folder(folder_path: str) -> List[str]:
    return [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and Path(f).suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]


# =========================================================================
# 2. CHUNKING — plain character sliding window with overlap.
# =========================================================================

def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    chunks = []
    step = max(chunk_size - chunk_overlap, 1)
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# =========================================================================
# 3. EMBEDDINGS — raw HuggingFace transformers with the model's pooler.
# =========================================================================

_tokenizer = None
_model = None


def load_embedding_model():
    """Loads the tokenizer + model once (module-level cache) and reuses
    them on every subsequent call — the actual load is the slow part."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        with stage(f"Loading HF tokenizer + model '{config.EMB_MODEL_NAME}'"):
            _tokenizer = AutoTokenizer.from_pretrained(config.EMB_MODEL_NAME)
            _model = AutoModel.from_pretrained(config.EMB_MODEL_NAME)
            _model.eval()
    return _tokenizer, _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Tokenize -> forward pass -> pool -> L2-normalize (so cosine
    similarity in ChromaDB behaves as plain dot product)."""
    if not texts:
        return []
    tokenizer, model = load_embedding_model()

    encoded = tokenizer(
        texts, padding=True, truncation=True,
        max_length=config.EMB_MAX_LENGTH, return_tensors="pt",
    )
    with torch.no_grad():
        model_output = model(**encoded)

    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


# =========================================================================
# 4. CHROMADB — three deliberately separate steps.
# =========================================================================

def load_chromadb(persist_dir: str = None, collection_name: str = None):
    """STEP 1 — call this first. Creates/connects to the persistent
    ChromaDB collection with cosine similarity. Ingestion and search both
    reuse the collection this returns; they never create their own."""
    persist_dir = persist_dir or config.PERSIST_DIR
    collection_name = collection_name or config.COLLECTION_NAME

    with stage(f"Loading ChromaDB (persist_dir='{persist_dir}', collection='{collection_name}', cosine)"):
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # only takes effect at creation
        )
    return client, collection


def ingest_documents(collection, file_paths: List[str],
                      progress_callback=None) -> dict:
    """STEP 2 — separate from load_chromadb(). Call load_chromadb() first
    and pass its collection in.

    File loading (text extraction / OCR) is done in parallel via
    load_files_parallel() since that's the I/O-bound part and files don't
    depend on each other. Chunking, embedding, and collection.add() then
    run sequentially per file — the embedding model and the ChromaDB client
    aren't safe (or beneficial) to hit concurrently here."""
    stats = {"files_processed": 0, "files_skipped": 0, "chunks_added": 0, "errors": []}

    texts_by_path = load_files_parallel(file_paths)

    for idx, file_path in enumerate(file_paths, start=1):
        source = os.path.basename(file_path)
        try:
            with stage(f"Ingesting {source}"):
                text = texts_by_path[file_path]
                if isinstance(text, Exception):
                    raise text

                chunks = chunk_text(text)

                if not chunks:
                    stats["files_skipped"] += 1
                    print(f"   {source}: no text extracted, skipped")
                else:
                    embeddings = embed_texts(chunks)
                    ids = [f"{source}_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
                    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

                    collection.add(
                        documents=chunks,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids,
                    )
                    stats["files_processed"] += 1
                    stats["chunks_added"] += len(chunks)
                    print(f"   {source}: {len(chunks)} chunk(s) embedded + stored")
        except Exception as e:
            stats["errors"].append(f"{source}: {e}")
            print(f"   {source}: ERROR — {e}")

        if progress_callback:
            progress_callback(source, idx, len(file_paths))

    stats["total_chunks_in_db"] = collection.count()
    return stats


def search_similar(collection, query: str, top_k: int = None) -> dict:
    """STEP 3 — separate from both of the above. Embeds the query with the
    exact same model/pooling as ingestion, then runs a cosine-similarity
    search (the collection's hnsw:space="cosine" set at creation)."""
    top_k = top_k or config.TOP_K
    with stage(f"Searching (cosine, top_k={top_k}) for: '{query}'"):
        query_embedding = embed_texts([query])[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results


# =========================================================================
# 5. GENERATION — a single Gemini call through LangChain.
# =========================================================================

_gemini_client = None
_gemini_client_config = None


def get_gemini_client(model: str = None, temperature: float = None):
    global _gemini_client, _gemini_client_config
    client_config = (
        model or config.GEMINI_MODEL,
        config.DEFAULT_TEMPERATURE if temperature is None else temperature,
    )
    if _gemini_client is None or _gemini_client_config != client_config:
        _gemini_client = ChatGoogleGenerativeAI(
            model=client_config[0],
            google_api_key=config.GOOGLE_API_KEY,
            temperature=client_config[1],
            max_retries=2,
        )
        _gemini_client_config = client_config
    return _gemini_client


def build_context(results: dict) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    parts = [f"[Source: {m.get('source', 'unknown')}]\n{d}" for d, m in zip(docs, metas)]
    return "\n\n---\n\n".join(parts)


def extract_response_text(content) -> str:
    """Normalize Gemini's string or structured content into display text."""
    if isinstance(content, list):
        text_parts = content[0].get("text", [])
        print(str(text_parts))
        return text_parts
    return str(content)


def generate_answer(query: str, results: dict, chat_history: list = None,
                     temperature: float = None, model: str = None):
    """Build the prompt from retrieved chunks, ask Gemini, and return
    (answer_text, sorted_source_filenames)."""
    temperature = config.DEFAULT_TEMPERATURE if temperature is None else temperature
    model = model or config.GEMINI_MODEL

    context = build_context(results)
    messages = prompts.build_messages(query, context, chat_history=chat_history)

    with stage(f"Generating answer (model={model}, temperature={temperature})"):
        response = get_gemini_client(model=model, temperature=temperature).invoke(messages)
    answer = extract_response_text(response.content)
    sources = sorted(set(m.get("source", "unknown") for m in results.get("metadatas", [[]])[0]))
    return answer, sources