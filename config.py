"""
config.py
Minimal constants for the POC — no HNSW tuning knobs, no multi-file
abstraction layers, just what's needed to run locally.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "documents")      # drop files here for bulk ingestion
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")       # Streamlit-uploaded files land here
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")       # ChromaDB persistent storage

COLLECTION_NAME = "poc_rag_collection"

# Raw HuggingFace transformer model (loaded via AutoTokenizer/AutoModel, not
# the sentence-transformers wrapper — see embed_texts() in rag_core.py for why).
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_MAX_LENGTH = 256  # tokens per chunk fed into the embedding model

CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 100     # characters

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".json", ".txt",
                        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_TEMPERATURE = 0.3
TOP_K = 4

# Thread pool size for parallel file loading during ingestion (I/O + OCR
# bound, so threads are fine — capped so a big folder drop doesn't spawn
# hundreds of threads at once).
MAX_INGEST_WORKERS = int(os.getenv("MAX_INGEST_WORKERS", "4"))

TESSERACT_CMD = os.getenv("TESSERACT_CMD")  # Windows: path to tesseract.exe
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
