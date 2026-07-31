# config.py

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_DB_PATH = "outputs/vector_db"

SUPPORTED_FILES = [
    ".pdf",
    ".docx",
    ".txt"
]