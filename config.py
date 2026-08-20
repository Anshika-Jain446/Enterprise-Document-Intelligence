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
    "",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "google/gemini-2.0-flash-001",
)


# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# ============================================================
# VECTOR DATABASE
# ============================================================

# Existing local FAISS database directory.
VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    "vector_store",
)


# ============================================================
# POSTGRESQL DATABASE
# ============================================================

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost",
)

POSTGRES_PORT = int(
    os.getenv(
        "POSTGRES_PORT",
        "5432",
    )
)

POSTGRES_DB = os.getenv(
    "POSTGRES_DB",
    "rag_database",
)

POSTGRES_USER = os.getenv(
    "POSTGRES_USER",
    "postgres",
)

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD",
    "",
)


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200",
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
    ".pptx",
]


# ============================================================
# VALIDATION
# ============================================================

if CHUNK_SIZE <= 0:
    raise ValueError(
        "CHUNK_SIZE must be greater than 0."
    )

if CHUNK_OVERLAP < 0:
    raise ValueError(
        "CHUNK_OVERLAP cannot be negative."
    )

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
    )


if POSTGRES_PORT <= 0:
    raise ValueError(
        "POSTGRES_PORT must be greater than 0."
    )


# ============================================================
# API KEY WARNING
# ============================================================

if not OPENROUTER_API_KEY:
    print(
        "WARNING: OPENROUTER_API_KEY is not configured."
    )


# ============================================================
# POSTGRESQL WARNING
# ============================================================

if not POSTGRES_PASSWORD:
    print(
        "WARNING: POSTGRES_PASSWORD is not configured."
    )