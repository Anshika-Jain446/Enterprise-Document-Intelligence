from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

import tiktoken
import pandas as pd

from config import CHUNK_SIZE, CHUNK_OVERLAP


class ChunkingEngine:

    def __init__(
        self,
        text,
        metadata=None,
        tables=None,
        source="Unknown",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ):
        self.text = text
        self.metadata = metadata if metadata else {}
        self.tables = tables if tables else []
        self.source = source

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ============================================================
    # CHARACTER CHUNKING
    # ============================================================

    def character_chunking(self):

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        chunks = splitter.split_text(self.text)

        results = []

        for i, chunk in enumerate(chunks):

            results.append(
                {
                    "chunk_id": i + 1,
                    "chunk_type": "Character",
                    "content": chunk,
                    "characters": len(chunk),
                    "metadata": self.metadata,
                    "source": self.source,
                }
            )

        return results

    # ============================================================
    # RECURSIVE CHUNKING
    # ============================================================

    def recursive_chunking(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = splitter.split_text(self.text)

        results = []

        for i, chunk in enumerate(chunks):

            results.append(
                {
                    "chunk_id": i + 1,
                    "chunk_type": "Recursive",
                    "content": chunk,
                    "characters": len(chunk),
                    "metadata": self.metadata,
                    "source": self.source,
                }
            )

        return results

    # ============================================================
    # TOKEN CHUNKING
    # ============================================================

    def token_chunking(self):

        encoder = tiktoken.get_encoding(
            "cl100k_base"
        )

        tokens = encoder.encode(
            self.text
        )

        chunk_size = self.chunk_size
        overlap = self.chunk_overlap

        if overlap >= chunk_size:

            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size."
            )

        chunks = []

        step = chunk_size - overlap

        for i in range(
            0,
            len(tokens),
            step
        ):

            token_chunk = tokens[
                i:i + chunk_size
            ]

            chunk = encoder.decode(
                token_chunk
            )

            if not chunk.strip():
                continue

            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "chunk_type": "Token",
                    "content": chunk,
                    "tokens": len(token_chunk),
                    "metadata": self.metadata,
                    "source": self.source,
                }
            )

        return chunks

    # ============================================================
    # MARKDOWN CHUNKING
    # ============================================================

    def markdown_chunking(self):

        headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers
        )

        docs = splitter.split_text(
            self.text
        )

        chunks = []

        for i, doc in enumerate(docs):

            chunks.append(
                {
                    "chunk_id": i + 1,
                    "chunk_type": "Markdown",
                    "metadata": {
                        **self.metadata,
                        **doc.metadata,
                    },
                    "content": doc.page_content,
                    "source": self.source,
                }
            )

        return chunks

    # ============================================================
    # TABLE CHUNKING
    # ============================================================

    def table_chunking(self):

        chunks = []

        for i, table in enumerate(
            self.tables
        ):

            if not isinstance(
                table,
                dict
            ):
                continue

            rows = table.get(
                "table",
                []
            )

            page = table.get(
                "page",
                0
            )

            text = (
                f"Table extracted from Page "
                f"{page}\n\n"
            )

            for row_number, row in enumerate(
                rows,
                start=1
            ):

                text += (
                    f"Row {row_number}\n"
                )

                if isinstance(
                    row,
                    dict
                ):

                    for key, value in row.items():

                        text += (
                            f"{key}: {value}\n"
                        )

                else:

                    text += (
                        f"{row}\n"
                    )

                text += "\n"

            chunks.append(
                {
                    "chunk_id": i + 1,
                    "chunk_type": "Table",
                    "page": page,
                    "rows": len(rows),
                    "content": text,
                    "metadata": self.metadata,
                    "source": self.source,
                    "table": rows,
                }
            )

        return chunks

    # ============================================================
    # CONTEXTUAL CHUNKING
    # ============================================================

    def contextual_chunking(self):

        paragraphs = self.text.split(
            "\n\n"
        )

        chunks = []

        count = 1

        for para in paragraphs:

            para = para.strip()

            if len(para) < 100:
                continue

            chunks.append(
                {
                    "chunk_id": count,
                    "chunk_type": "Context",
                    "content": para,
                    "characters": len(para),
                    "metadata": self.metadata,
                    "source": self.source,
                }
            )

            count += 1

        return chunks

    # ============================================================
    # CHUNK STATISTICS
    # ============================================================

    def chunk_statistics(
        self,
        chunks
    ):

        if not chunks:

            return {
                "Total Chunks": 0,
                "Average Length": 0,
                "Maximum Length": 0,
                "Minimum Length": 0,
            }

        lengths = [
            len(
                chunk.get(
                    "content",
                    ""
                )
            )
            for chunk in chunks
        ]

        return {
            "Total Chunks": len(chunks),
            "Average Length": round(
                sum(lengths) / len(lengths),
                2,
            ),
            "Maximum Length": max(
                lengths
            ),
            "Minimum Length": min(
                lengths
            ),
        }

    # ============================================================
    # EXPORT CHUNKS DATAFRAME
    # ============================================================

    def export_chunks_dataframe(
        self,
        chunks
    ):

        rows = []

        for chunk in chunks:

            rows.append(
                {
                    "Chunk ID": chunk.get(
                        "chunk_id"
                    ),
                    "Type": chunk.get(
                        "chunk_type"
                    ),
                    "Length": len(
                        chunk.get(
                            "content",
                            ""
                        )
                    ),
                    "Source": chunk.get(
                        "source",
                        "Unknown",
                    ),
                    "Content": chunk.get(
                        "content",
                        "",
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )

    # ============================================================
    # COMPARE ALL CHUNKING METHODS
    # ============================================================

    def compare_chunking(self):

        methods = {
            "Character":
                self.character_chunking(),

            "Recursive":
                self.recursive_chunking(),

            "Token":
                self.token_chunking(),

            "Markdown":
                self.markdown_chunking(),

            "Table":
                self.table_chunking(),

            "Context":
                self.contextual_chunking(),
        }

        report = {}

        for name, chunks in methods.items():

            report[name] = {
                "statistics":
                    self.chunk_statistics(
                        chunks
                    ),

                "chunks":
                    chunks,
            }

        return report

    # ============================================================
    # DISPLAY SUMMARY
    # ============================================================

    def display_summary(self):

        report = (
            self.compare_chunking()
        )

        print(
            "=" * 60
        )

        print(
            "CHUNKING SUMMARY"
        )

        print(
            "=" * 60
        )

        for method, data in report.items():

            print(
                f"\nMethod : {method}"
            )

            stats = data[
                "statistics"
            ]

            print(
                f"Total Chunks   : "
                f"{stats['Total Chunks']}"
            )

            print(
                f"Average Length : "
                f"{stats['Average Length']}"
            )

            print(
                f"Maximum Length : "
                f"{stats['Maximum Length']}"
            )

            print(
                f"Minimum Length : "
                f"{stats['Minimum Length']}"
            )

        print(
            "=" * 60
        )

