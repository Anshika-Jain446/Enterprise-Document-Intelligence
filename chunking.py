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
        text="",
        metadata=None,
        tables=None,
        images=None,
        visuals=None,
        pages=None,
        source="Unknown",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ):
        self.text = text or ""
        self.metadata = metadata or {}
        self.tables = tables or []
        self.images = images or []
        self.visuals = visuals or []
        self.pages = pages or []
        self.source = source or "Unknown"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

    # ============================================================
    # COMMON METADATA
    # ============================================================

    def _base_metadata(self):
        return {
            **self.metadata,
            "source": self.source,
        }

    def _make_chunk(
        self,
        chunk_id,
        chunk_type,
        content,
        **extra,
    ):
        metadata = {
            **self._base_metadata(),
            **extra.get("metadata", {}),
        }

        result = {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "content": content,
            "characters": len(content),
            "metadata": metadata,
            "source": self.source,
        }

        for key, value in extra.items():
            if key != "metadata":
                result[key] = value

        return result

    # ============================================================
    # PAGE LOOKUP
    # ============================================================

    def _page_number_for_position(self, position):
        """
        Best-effort page detection for text chunks.

        If page-aware extraction data exists, this attempts to
        associate the chunk with the page containing its text.
        """

        if not self.pages:
            return None

        running_position = 0

        for page in self.pages:

            if not isinstance(page, dict):
                continue

            page_text = str(
                page.get("text", "")
            )

            start = running_position
            end = start + len(page_text)

            if start <= position <= end:
                return page.get("page", 1)

            running_position = end + 1

        return None

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

        search_position = 0

        for i, chunk in enumerate(chunks):

            position = self.text.find(
                chunk,
                search_position,
            )

            page = self._page_number_for_position(
                position
            )

            if position >= 0:
                search_position = (
                    position + len(chunk)
                )

            results.append(
                self._make_chunk(
                    chunk_id=i + 1,
                    chunk_type="Character",
                    content=chunk,
                    page=page,
                )
            )

        return results

    # ============================================================
    # RECURSIVE CHUNKING
    # ============================================================

    def recursive_chunking(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

        chunks = splitter.split_text(self.text)

        results = []

        search_position = 0

        for i, chunk in enumerate(chunks):

            position = self.text.find(
                chunk,
                search_position,
            )

            page = self._page_number_for_position(
                position
            )

            if position >= 0:
                search_position = (
                    position + len(chunk)
                )

            results.append(
                self._make_chunk(
                    chunk_id=i + 1,
                    chunk_type="Recursive",
                    content=chunk,
                    page=page,
                )
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

        step = (
            self.chunk_size
            - self.chunk_overlap
        )

        chunks = []

        for i in range(
            0,
            len(tokens),
            step,
        ):

            token_chunk = tokens[
                i:i + self.chunk_size
            ]

            chunk = encoder.decode(
                token_chunk
            )

            if not chunk.strip():
                continue

            chunks.append(
                self._make_chunk(
                    chunk_id=len(chunks) + 1,
                    chunk_type="Token",
                    content=chunk,
                    tokens=len(token_chunk),
                )
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
            headers_to_split_on=headers,
            strip_headers=False,
        )

        docs = splitter.split_text(
            self.text
        )

        chunks = []

        for i, doc in enumerate(docs):

            chunks.append(
                self._make_chunk(
                    chunk_id=i + 1,
                    chunk_type="Markdown",
                    content=doc.page_content,
                    metadata={
                        **doc.metadata,
                    },
                )
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

        buffer = ""
        count = 1

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(paragraph) < 100:

                buffer += (
                    paragraph
                    + "\n\n"
                )

                continue

            if buffer:

                paragraph = (
                    buffer
                    + paragraph
                )

                buffer = ""

            chunks.append(
                self._make_chunk(
                    chunk_id=count,
                    chunk_type="Context",
                    content=paragraph,
                )
            )

            count += 1

        if buffer.strip():

            chunks.append(
                self._make_chunk(
                    chunk_id=count,
                    chunk_type="Context",
                    content=buffer.strip(),
                )
            )

        return chunks

    # ============================================================
    # TABLE CHUNKING
    # ============================================================

    def table_chunking(self):

        chunks = []

        for table_index, table in enumerate(
            self.tables,
            start=1,
        ):

            if not isinstance(
                table,
                dict,
            ):
                continue

            rows = table.get(
                "table",
                [],
            )

            page = table.get(
                "page",
                table.get(
                    "sheet",
                    0,
                ),
            )

            table_source = table.get(
                "source",
                self.source,
            )

            lines = [
                f"TABLE",
                f"Source: {table_source}",
                f"Page/Sheet: {page}",
                "",
            ]

            for row_number, row in enumerate(
                rows,
                start=1,
            ):

                lines.append(
                    f"Row {row_number}:"
                )

                if isinstance(row, dict):

                    for key, value in row.items():

                        lines.append(
                            f"{key}: {value}"
                        )

                else:

                    lines.append(
                        str(row)
                    )

                lines.append("")

            content = "\n".join(
                lines
            )

            chunks.append(
                self._make_chunk(
                    chunk_id=table_index,
                    chunk_type="Table",
                    content=content,
                    page=page,
                    rows=len(rows),
                    table=rows,
                    table_source=table_source,
                    metadata={
                        "table_index": table_index,
                        "page": page,
                        "sheet": table.get(
                            "sheet"
                        ),
                        "table_source": table_source,
                    },
                )
            )

        return chunks

    # ============================================================
    # IMAGE CHUNKING
    # ============================================================

    def image_chunking(self):

        """
        Creates retrievable image records.

        The extractor is responsible for OCR / visual analysis.
        This class preserves that analysis so the Agentic RAG
        system can retrieve image-derived knowledge.

        Expected image fields may include:

        path
        page
        image_no
        ocr_text
        visual_description
        analysis
        """

        chunks = []

        chunk_id = 1

        for page_data in self.images:

            if not isinstance(
                page_data,
                dict,
            ):
                continue

            page = page_data.get(
                "page",
                1,
            )

            image_list = page_data.get(
                "images",
                [],
            )

            for image in image_list:

                if not isinstance(
                    image,
                    dict,
                ):
                    continue

                image_path = image.get(
                    "path",
                    "",
                )

                ocr_text = str(
                    image.get(
                        "ocr_text",
                        "",
                    )
                    or ""
                )

                visual_description = str(
                    image.get(
                        "visual_description",
                        "",
                    )
                    or ""
                )

                analysis = str(
                    image.get(
                        "analysis",
                        "",
                    )
                    or ""
                )

                content_parts = [
                    "IMAGE CONTENT",
                    f"Source: {self.source}",
                    f"Page: {page}",
                    f"Image Number: "
                    f"{image.get('image_no', chunk_id)}",
                ]

                if ocr_text:

                    content_parts.extend(
                        [
                            "",
                            "OCR TEXT:",
                            ocr_text,
                        ]
                    )

                if visual_description:

                    content_parts.extend(
                        [
                            "",
                            "VISUAL DESCRIPTION:",
                            visual_description,
                        ]
                    )

                if analysis:

                    content_parts.extend(
                        [
                            "",
                            "IMAGE ANALYSIS:",
                            analysis,
                        ]
                    )

                content = "\n".join(
                    content_parts
                )

                chunks.append(
                    self._make_chunk(
                        chunk_id=chunk_id,
                        chunk_type="Image",
                        content=content,
                        page=page,
                        image_path=image_path,
                        image_no=image.get(
                            "image_no",
                            chunk_id,
                        ),
                        ocr_text=ocr_text,
                        visual_description=(
                            visual_description
                        ),
                        image_analysis=analysis,
                        metadata={
                            "page": page,
                            "image_path": image_path,
                            "image_no": image.get(
                                "image_no",
                                chunk_id,
                            ),
                            "has_ocr": bool(
                                ocr_text.strip()
                            ),
                            "has_visual_analysis": bool(
                                visual_description.strip()
                                or analysis.strip()
                            ),
                        },
                    )
                )

                chunk_id += 1

        return chunks

    # ============================================================
    # VISUAL / DIAGRAM CHUNKING
    # ============================================================

    def visual_chunking(self):

        chunks = []

        for index, visual in enumerate(
            self.visuals,
            start=1,
        ):

            if not isinstance(
                visual,
                dict,
            ):
                continue

            page = visual.get(
                "page",
                1,
            )

            visual_description = str(
                visual.get(
                    "visual_description",
                    "",
                )
                or ""
            )

            analysis = str(
                visual.get(
                    "analysis",
                    "",
                )
                or ""
            )

            ocr_text = str(
                visual.get(
                    "ocr_text",
                    "",
                )
                or ""
            )

            content_parts = [
                "DOCUMENT VISUAL / DIAGRAM",
                f"Source: {self.source}",
                f"Page: {page}",
            ]

            if ocr_text:

                content_parts.extend(
                    [
                        "",
                        "OCR TEXT:",
                        ocr_text,
                    ]
                )

            if visual_description:

                content_parts.extend(
                    [
                        "",
                        "VISUAL DESCRIPTION:",
                        visual_description,
                    ]
                )

            if analysis:

                content_parts.extend(
                    [
                        "",
                        "VISUAL ANALYSIS:",
                        analysis,
                    ]
                )

            content = "\n".join(
                content_parts
            )

            chunks.append(
                self._make_chunk(
                    chunk_id=index,
                    chunk_type="Visual",
                    content=content,
                    page=page,
                    visual_path=visual.get(
                        "path"
                    ),
                    visual_type=visual.get(
                        "type",
                        "Document Visual",
                    ),
                    ocr_text=ocr_text,
                    visual_description=(
                        visual_description
                    ),
                    visual_analysis=analysis,
                    metadata={
                        "page": page,
                        "visual_path": visual.get(
                            "path"
                        ),
                        "visual_type": visual.get(
                            "type",
                            "Document Visual",
                        ),
                    },
                )
            )

        return chunks

    # ============================================================
    # MULTIMODAL / AGENTIC DOCUMENT CHUNKING
    # ============================================================

    def multimodal_chunking(self):

        """
        Builds the complete retrieval corpus.

        Text + tables + images + diagrams are represented as
        independent retrievable chunks.

        This is important for Agentic RAG because the agent can
        decide whether it needs text, tables, images, or visual
        evidence.
        """

        chunks = []

        chunks.extend(
            self.recursive_chunking()
        )

        chunks.extend(
            self.table_chunking()
        )

        chunks.extend(
            self.image_chunking()
        )

        chunks.extend(
            self.visual_chunking()
        )

        # Reassign globally unique IDs.
        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            chunk["chunk_id"] = index

        return chunks

    # ============================================================
    # CHUNK STATISTICS
    # ============================================================

    def chunk_statistics(
        self,
        chunks,
    ):

        if not chunks:

            return {
                "Total Chunks": 0,
                "Average Length": 0,
                "Maximum Length": 0,
                "Minimum Length": 0,
                "Text Chunks": 0,
                "Table Chunks": 0,
                "Image Chunks": 0,
                "Visual Chunks": 0,
            }

        lengths = [
            len(
                str(
                    chunk.get(
                        "content",
                        "",
                    )
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
            "Maximum Length": max(lengths),
            "Minimum Length": min(lengths),
            "Text Chunks": len(
                [
                    c
                    for c in chunks
                    if c.get("chunk_type")
                    in {
                        "Character",
                        "Recursive",
                        "Token",
                        "Markdown",
                        "Context",
                    }
                ]
            ),
            "Table Chunks": len(
                [
                    c
                    for c in chunks
                    if c.get("chunk_type")
                    == "Table"
                ]
            ),
            "Image Chunks": len(
                [
                    c
                    for c in chunks
                    if c.get("chunk_type")
                    == "Image"
                ]
            ),
            "Visual Chunks": len(
                [
                    c
                    for c in chunks
                    if c.get("chunk_type")
                    == "Visual"
                ]
            ),
        }

    # ============================================================
    # EXPORT CHUNKS DATAFRAME
    # ============================================================

    def export_chunks_dataframe(
        self,
        chunks,
    ):

        rows = []

        for chunk in chunks:

            rows.append(
                {
                    "Chunk ID":
                        chunk.get("chunk_id"),

                    "Type":
                        chunk.get("chunk_type"),

                    "Page":
                        chunk.get("page", ""),

                    "Length":
                        len(
                            str(
                                chunk.get(
                                    "content",
                                    "",
                                )
                            )
                        ),

                    "Source":
                        chunk.get(
                            "source",
                            "Unknown",
                        ),

                    "Image":
                        chunk.get(
                            "image_path",
                            chunk.get(
                                "visual_path",
                                "",
                            ),
                        ),

                    "Content":
                        chunk.get(
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

            "Image":
                self.image_chunking(),

            "Visual":
                self.visual_chunking(),

            "Multimodal":
                self.multimodal_chunking(),
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

        print("=" * 70)
        print("AGENTIC RAG MULTIMODAL CHUNKING SUMMARY")
        print("=" * 70)

        for method, data in report.items():

            stats = data[
                "statistics"
            ]

            print(
                f"\nMethod: {method}"
            )

            print(
                f"Total Chunks: "
                f"{stats['Total Chunks']}"
            )

            print(
                f"Average Length: "
                f"{stats['Average Length']}"
            )

            print(
                f"Text Chunks: "
                f"{stats['Text Chunks']}"
            )

            print(
                f"Table Chunks: "
                f"{stats['Table Chunks']}"
            )

            print(
                f"Image Chunks: "
                f"{stats['Image Chunks']}"
            )

            print(
                f"Visual Chunks: "
                f"{stats['Visual Chunks']}"
            )

        print("=" * 70)


# ================================================================
# EXAMPLE
# ================================================================

if __name__ == "__main__":

    engine = ChunkingEngine(
        text="Example enterprise document text.",
        metadata={
            "document_type": "PDF",
        },
        tables=[],
        images=[],
        visuals=[],
        pages=[],
        source="example.pdf",
    )

    chunks = engine.multimodal_chunking()

    print(
        f"Generated {len(chunks)} multimodal chunks."
    )

    print(
        engine.chunk_statistics(
            chunks
        )
    )