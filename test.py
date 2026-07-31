from extractor import DocumentExtractor

pdf_path = "data/sample.pdf"

extractor = DocumentExtractor(pdf_path)

document = extractor.extract_document()

print("\n========== Metadata ==========\n")

for key, value in document["metadata"].items():
    print(f"{key}: {value}")

print("\n========== Text Preview ==========\n")

print(document["text"][:1000])

print("\n========== Tables ==========\n")

print("Tables Found:", len(document["tables"]))

for table in document["tables"]:

    print(f"\nTable on Page {table['page']}")

    print(table["table"])