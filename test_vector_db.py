from extractor import DocumentExtractor
from chunking import ChunkingEngine
from vector_db import VectorDatabase


# ============================================================
# DOCUMENT PATH
# ============================================================

pdf = input(
    "Enter document path: "
).strip()


# ============================================================
# EXTRACTION
# ============================================================

extractor = DocumentExtractor(
    pdf
)

document = extractor.extract_document()


# ============================================================
# CHUNKING
# ============================================================

engine = ChunkingEngine(
    text=document["text"],
    metadata=document["metadata"],
    tables=document["tables"],
    source=document["metadata"].get(
        "file_name",
        pdf
    )
)


# ============================================================
# USE RECURSIVE CHUNKING
# ============================================================

chunks = engine.recursive_chunking()


print(
    f"Created {len(chunks)} chunks."
)


# ============================================================
# VECTOR DATABASE
# ============================================================

db = VectorDatabase()


db.create_index(
    chunks
)

db.save()


# ============================================================
# SEARCH
# ============================================================

query = input(
    "\nAsk Question: "
)


results = db.search(
    query,
    top_k=5
)


db.display_results(
    results
)