import os
from urllib.parse import quote_plus

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GOOGLE GEMINI API / LLM
# ============================================================

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY", "").strip()
    or os.getenv("GEMINI_API_KEY", "").strip()
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-2.5-flash",
).strip()


# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()


# ============================================================
# VECTOR DATABASE
# ============================================================

VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    "vector_store",
).strip()


# ============================================================
# POSTGRESQL / SUPABASE
# ============================================================

# IMPORTANT:
# If DATABASE_URL is present, it is used directly.
# This is the recommended configuration for Streamlit Cloud.

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "",
).strip()


# ------------------------------------------------------------
# Individual PostgreSQL variables
# ------------------------------------------------------------

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "",
).strip()

POSTGRES_PORT = os.getenv(
    "POSTGRES_PORT",
    "5432",
).strip()

POSTGRES_DB = os.getenv(
    "POSTGRES_DB",
    "postgres",
).strip()

POSTGRES_USER = os.getenv(
    "POSTGRES_USER",
    "postgres",
).strip()

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD",
    "",
).strip()


# ============================================================
# BUILD DATABASE URL IF DATABASE_URL WAS NOT PROVIDED
# ============================================================

if not DATABASE_URL:

    if (
        POSTGRES_HOST
        and POSTGRES_USER
        and POSTGRES_PASSWORD
        and POSTGRES_DB
    ):

        encoded_user = quote_plus(
            POSTGRES_USER
        )

        encoded_password = quote_plus(
            POSTGRES_PASSWORD
        )

        encoded_db = quote_plus(
            POSTGRES_DB
        )

        DATABASE_URL = (
            f"postgresql://"
            f"{encoded_user}:"
            f"{encoded_password}@"
            f"{POSTGRES_HOST}:"
            f"{POSTGRES_PORT}/"
            f"{encoded_db}"
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

try:
    POSTGRES_PORT = int(
        POSTGRES_PORT
    )
except ValueError:
    raise ValueError(
        "POSTGRES_PORT must be a valid integer."
    )

if POSTGRES_PORT <= 0:
    raise ValueError(
        "POSTGRES_PORT must be greater than 0."
    )


# ============================================================
# CONFIGURATION STATUS
# ============================================================

if not GOOGLE_API_KEY:
    print(
        "WARNING: GOOGLE_API_KEY is not configured. "
        "Get a free key at https://aistudio.google.com/apikey"
    )


if not DATABASE_URL:
    print(
        "WARNING: DATABASE_URL is not configured."
    )

    print(
        "Set DATABASE_URL in Streamlit Secrets."
    )


# ============================================================
# DATABASE DEBUG INFORMATION
# ============================================================

# Do NOT print the actual password.
# This only helps confirm which configuration is being used.

if DATABASE_URL:

    try:
        from urllib.parse import urlparse

        parsed = urlparse(
            DATABASE_URL
        )

        print(
            "PostgreSQL configuration loaded:"
        )

        print(
            f"  Host: {parsed.hostname}"
        )

        print(
            f"  Port: {parsed.port or 5432}"
        )

        print(
            f"  Database: {parsed.path.lstrip('/')}"
        )

        print(
            f"  User: {parsed.username}"
        )

    except Exception:
        print(
            "PostgreSQL DATABASE_URL is present."
        )