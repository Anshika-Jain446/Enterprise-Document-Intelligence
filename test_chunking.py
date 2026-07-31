from extractor import DocumentExtractor
from chunking import ChunkingEngine

pdf = "data/sample.pdf"

extractor = DocumentExtractor(pdf)

document = extractor.extract_document()

text = document["text"]

engine = ChunkingEngine(text)

results = engine.compare_chunking()

for name, chunks in results.items():

    print("=" * 60)

    print(name)

    print("Chunks:", len(chunks))

    print("=" * 60)

    print(chunks[0])

    print("\n\n")