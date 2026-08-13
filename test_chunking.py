from extractor import DocumentExtractor
from chunking import ChunkingEngine


pdf = "data/sample.pdf"


# ============================================================
# EXTRACTION
# ============================================================

extractor = DocumentExtractor(
    pdf
)

document = (
    extractor.extract_document()
)


# ============================================================
# CHUNKING ENGINE
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
# COMPARE ALL METHODS
# ============================================================

results = (
    engine.compare_chunking()
)


# ============================================================
# DISPLAY
# ============================================================

for name, data in results.items():

    print("=" * 60)

    print(
        name
    )

    print("=" * 60)

    chunks = data[
        "chunks"
    ]

    statistics = data[
        "statistics"
    ]

    print(
        "Chunks:",
        len(chunks)
    )

    print(
        "Statistics:",
        statistics
    )

    if chunks:

        print(
            "\nFirst Chunk:"
        )

        print(
            chunks[0]
        )

    print(
        "\n\n"
    )