import os

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# OPENROUTER / LLM
# ============================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "google/gemini-2.0-flash-001"
)


# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# VECTOR DATABASE
# ============================================================

VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    "vector_store"
)


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000"
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200"
    )
)


# ============================================================
# SUPPORTED DOCUMENT FILES
# ============================================================

SUPPORTED_FILES = [
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".xlsx",
    ".xls",
]


# ============================================================
# VALIDATION
# ============================================================

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
    )

if not OPENROUTER_API_KEY:
    print(
        "WARNING: OPENROUTER_API_KEY is not configured."
    )