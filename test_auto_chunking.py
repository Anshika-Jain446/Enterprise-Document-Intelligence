from auto_chunking import AutomaticChunkingSelector


selector = AutomaticChunkingSelector()


tests = [
    {
        "name": "Normal PDF",
        "file_name": "document.pdf",
        "text": "This is a normal text document.",
    },
    {
        "name": "Markdown",
        "file_name": "README.md",
        "text": "# Introduction\n## Installation",
    },
    {
        "name": "Excel",
        "file_name": "data.xlsx",
    },
    {
        "name": "CSV",
        "file_name": "data.csv",
    },
    {
        "name": "PDF with Tables",
        "file_name": "report.pdf",
        "text": "Financial report",
        "tables": [{"table": 1}, {"table": 2}],
    },
    {
        "name": "PDF with Images",
        "file_name": "images.pdf",
        "text": "Document with images",
        "images": ["image1", "image2"],
    },
    {
        "name": "Multimodal PDF",
        "file_name": "research.pdf",
        "text": "Research paper",
        "tables": [{"table": 1}],
        "images": ["image1"],
        "visuals": ["diagram1"],
    },
]


print("=" * 60)
print("AUTOMATIC CHUNKING TEST")
print("=" * 60)

for test in tests:
    method = selector.select_method(
        file_name=test.get("file_name"),
        text=test.get("text"),
        tables=test.get("tables"),
        images=test.get("images"),
        visuals=test.get("visuals"),
    )

    print(f"{test['name']:<25} → {method}")

print("=" * 60)