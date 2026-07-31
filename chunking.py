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
        source="Unknown"
    ):

        self.text = text
        self.metadata = metadata if metadata else {}
        self.tables = tables if tables else []
        self.source = source

    # ------------------------------------
    # Character Chunking
    # ------------------------------------

    def character_chunking(self):

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )

        chunks = splitter.split_text(self.text)

        results = []

        for i, chunk in enumerate(chunks):

            results.append({

                "chunk_id": i + 1,
                "chunk_type": "Character",

                "content": chunk,

                "characters": len(chunk),

                "metadata": self.metadata,

                "source": self.source

            })

        return results

    # ------------------------------------
    # Recursive Chunking
    # ------------------------------------

    def recursive_chunking(self):

        splitter = RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP

        )

        chunks = splitter.split_text(self.text)

        results = []

        for i, chunk in enumerate(chunks):

            results.append({

                "chunk_id": i + 1,

                "chunk_type": "Recursive",

                "content": chunk,

                "characters": len(chunk),

                "metadata": self.metadata,

                "source": self.source

            })

        return results

    # ------------------------------------
    # Token Chunking
    # ------------------------------------

    def token_chunking(self):

        encoder = tiktoken.get_encoding("cl100k_base")

        tokens = encoder.encode(self.text)

        chunk_size = 500

        chunks = []

        for i in range(0, len(tokens), chunk_size):

            chunk = encoder.decode(tokens[i:i + chunk_size])

            chunks.append({

                "chunk_id": len(chunks) + 1,

                "chunk_type": "Token",

                "content": chunk,

                "tokens": len(encoder.encode(chunk)),

                "metadata": self.metadata,

                "source": self.source

            })

        return chunks

    # ------------------------------------
    # Markdown Chunking
    # ------------------------------------

    def markdown_chunking(self):

        headers = [

            ("#", "Header 1"),

            ("##", "Header 2"),

            ("###", "Header 3")

        ]

        splitter = MarkdownHeaderTextSplitter(headers)

        docs = splitter.split_text(self.text)

        chunks = []

        for i, doc in enumerate(docs):

            chunks.append({

                "chunk_id": i + 1,

                "chunk_type": "Markdown",

                "metadata": doc.metadata,

                "content": doc.page_content,

                "source": self.source

            })

        return chunks

    # ------------------------------------
    # Table Chunking
    # ------------------------------------

    def table_chunking(self):

        chunks = []

        for i, table in enumerate(self.tables):

            if isinstance(table, dict):

                df = table["table"]

                markdown = df.to_markdown(index=False)

                chunks.append({

                    "chunk_id": i + 1,

                    "chunk_type": "Table",

                    "page": table["page"],

                    "rows": table["rows"],

                    "columns": table["columns"],

                    "content": markdown,

                    "source": self.source

                })

        return chunks

    # ------------------------------------
    # Context Chunking
    # ------------------------------------

    def contextual_chunking(self):

        paragraphs = self.text.split("\n\n")

        chunks = []

        count = 1

        for para in paragraphs:

            para = para.strip()

            if len(para) < 100:
                continue

            chunks.append({

                "chunk_id": count,

                "chunk_type": "Context",

                "content": para,

                "characters": len(para),

                "source": self.source

            })

            count += 1

        return chunks

    # ------------------------------------
    # Chunk Statistics
    # ------------------------------------

    def chunk_statistics(self, chunks):

        if len(chunks) == 0:

            return {

                "Total Chunks": 0,

                "Average Length": 0,

                "Maximum Length": 0,

                "Minimum Length": 0

            }

        lengths = []

        for chunk in chunks:

            lengths.append(len(chunk["content"]))

        return {

            "Total Chunks": len(chunks),

            "Average Length": round(sum(lengths) / len(lengths), 2),

            "Maximum Length": max(lengths),

            "Minimum Length": min(lengths)

        }

    # ------------------------------------
    # Compare All Chunking Methods
    # ------------------------------------

    def compare_chunking(self):

        methods = {

            "Character": self.character_chunking(),

            "Recursive": self.recursive_chunking(),

            "Token": self.token_chunking(),

            "Markdown": self.markdown_chunking(),

            "Table": self.table_chunking(),

            "Context": self.contextual_chunking()

        }

        report = {}

        for name, chunks in methods.items():

            report[name] = {

                "statistics": self.chunk_statistics(chunks),

                "chunks": chunks

            }

        return report