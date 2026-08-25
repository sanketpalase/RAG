# RAG POC — PDF / DOCX / JSON / TXT / Images + ChromaDB + Gemini

A **simple, locally-runnable proof of concept** — deliberately lighter than
the full production project (no LangChain, no parallel workers), but the
embedding step is done at a lower, more "advanced" level: raw HuggingFace
`transformers` (`AutoTokenizer` + `AutoModel`) with manual mean-pooling,
instead of the `sentence-transformers` convenience wrapper. It's exactly
what that wrapper does internally — just written out so you can see it.

## Files

```
rag_poc/
├── config.py         # paths, model name, chunk size, .env loading
├── rag_core.py        # everything: loaders, chunking, embeddings, ChromaDB, generation
├── timing_utils.py     # stage() — prints "what's loading" + elapsed time to the terminal
├── app.py               # Streamlit UI
├── documents/             # drop files here for bulk ingestion
├── uploads/                # Streamlit-uploaded files land here
├── .env.example
└── requirements.txt
```

Everything except the UI and timing helper lives in one file, `rag_core.py`,
organized in the order it actually runs — read it top to bottom and that's
the whole pipeline.

## The three separate ChromaDB steps (as asked)

```python
client, collection = load_chromadb()              # STEP 1 — persist_dir, cosine, once
stats = ingest_documents(collection, file_paths)   # STEP 2 — separate call
results = search_similar(collection, query)        # STEP 3 — separate call
```

`load_chromadb()` is cached in `app.py` via `@st.cache_resource`, so it truly
only runs once per server process. Ingestion and search are always separate
function calls — ingestion never searches, search never ingests.

## Embeddings: raw transformers, not sentence-transformers

```python
tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_NAME)
model = AutoModel.from_pretrained(EMB_MODEL_NAME)
...
model_output = model(**encoded)                    # forward pass -> pooled output
pooled = model_output.pooler_output
normalized = F.normalize(pooled, p=2, dim=1)         # so cosine similarity == dot product
```

Same model (`all-MiniLM-L6-v2`, 384-dim) is used directly through its model
pooler. Embeddings are
computed once per ingest/search call and passed straight into
`collection.add(embeddings=...)` / `collection.query(query_embeddings=...)`
— no ChromaDB embedding_function involved either, for the same reason.

Because this goes through `transformers.AutoModel` directly (not
`sentence-transformers`, not `langchain_huggingface`), it also sidesteps the
`torchvision`/vision-processor import chain that a fuller
sentence-transformers-based stack can occasionally hit — there's no vision
code path here to accidentally touch.

## Images: OCR only, not multimodal

`load_image_text()` uses `pytesseract` to extract plain text from images,
which then goes through the exact same chunk → embed → store pipeline as
every other file type. No vision-language model anywhere.

**Requires the Tesseract OCR engine installed separately** (pytesseract is
just the Python wrapper):
- Windows: https://github.com/UB-Mannheim/tesseract/wiki, then set
  `TESSERACT_CMD` in `.env` to the installed `tesseract.exe` path.
- Linux: `sudo apt install tesseract-ocr` (usually already on PATH).
- Mac: `brew install tesseract`.

## Setup

```bash
cd rag_poc
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# edit .env: GOOGLE_API_KEY=..., and TESSERACT_CMD if on Windows
```

> If you see `ModuleNotFoundError: No module named 'exceptions'` — that's
> the old, abandoned PyPI package literally named `docx` (not
> `python-docx`) shadowing things. Run `pip uninstall docx` then
> `pip install python-docx --force-reinstall`. This project imports
> `from docx import Document`, which needs `python-docx`.

## Run

```bash
streamlit run app.py
```

1. ChromaDB loads automatically on first request (Step 1, cached — watch the
   terminal for `>> Loading ChromaDB ...`).
2. Sidebar: upload files and/or drop them in `documents/`, click
   **Run Ingestion** (Step 2).
3. Main area: ask a question — search (Step 3) and generation happen per
   question.

## Terminal output

Every stage prints to the terminal via `timing_utils.stage()`:

```
>> Loading ChromaDB (persist_dir='./chroma_db', collection='poc_rag_collection', cosine) ...
   done (0.02s)
>> Loading HF tokenizer + model 'sentence-transformers/all-MiniLM-L6-v2' ...
   done (3.41s)
>> Ingesting faq.pdf ...
   faq.pdf: 3 chunk(s) embedded + stored
   done (0.18s)
>> Searching (cosine, top_k=4) for: 'what courses are offered?' ...
   done (0.02s)
>> Generating answer (model=gpt-4o-mini, temperature=0.3) ...
   done (1.05s)
```

The tokenizer/model load only happens once (module-level cache in
`rag_core.py`) — first ingest or first search pays that cost, everything
after reuses it.

## What's deliberately simpler than the main project

- No parallel ingestion (`ThreadPoolExecutor`) — files process one at a time.
- No LangChain anywhere (loaders, splitter, vector store, or LLM call are
  all plain library calls).
- Chunking is a plain character sliding window (`CHUNK_SIZE`/`CHUNK_OVERLAP`
  in `config.py`) — no tiktoken, no network dependency for chunking.
- One core file (`rag_core.py`) instead of split loader/store/chain modules.

If any of these need to scale up later (parallel ingestion, LangChain
components, token-based chunking), that's exactly what the main project
already does — this POC is meant to be the from-scratch version to learn
from or demo quickly, not a replacement for it.
