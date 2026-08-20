from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

import tiktoken
import pandas as pd

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


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

        self.text = str(text or "")
        self.metadata = metadata or {}
        self.tables = tables or []
        self.images = images or []
        self.visuals = visuals or []
        self.pages = pages or []
        self.source = source or "Unknown"

        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

    # =========================================================
    # METADATA
    # =========================================================

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

        content = str(content or "").strip()

        metadata = {
            **self._base_metadata(),
            **extra.pop("metadata", {}),
        }

        return {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "content": content,
            "characters": len(content),
            "metadata": metadata,
            "source": self.source,
            **extra,
        }

    # =========================================================
    # PAGE LOOKUP
    # =========================================================

    def _page_number_for_position(self, position):

        if not self.pages:
            return None

        if position is None or position < 0:
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

    # =========================================================
    # CHARACTER
    # =========================================================

    def character_chunking(self):

        if not self.text.strip():
            return []

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        raw_chunks = splitter.split_text(
            self.text
        )

        results = []
        search_position = 0

        for chunk in raw_chunks:

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
                    chunk_id=len(results) + 1,
                    chunk_type="Character",
                    content=chunk,
                    page=page,
                )
            )

        return results

    # =========================================================
    # RECURSIVE
    # =========================================================

    def recursive_chunking(self):

        if not self.text.strip():
            return []

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

        raw_chunks = splitter.split_text(
            self.text
        )

        results = []
        search_position = 0

        for chunk in raw_chunks:

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
                    chunk_id=len(results) + 1,
                    chunk_type="Recursive",
                    content=chunk,
                    page=page,
                )
            )

        return results

    # =========================================================
    # TOKEN
    # =========================================================

    def token_chunking(self):

        if not self.text.strip():
            return []

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

        results = []

        for start in range(
            0,
            len(tokens),
            step,
        ):

            token_chunk = tokens[
                start:start + self.chunk_size
            ]

            if not token_chunk:
                continue

            content = encoder.decode(
                token_chunk
            )

            if not content.strip():
                continue

            results.append(
                self._make_chunk(
                    chunk_id=len(results) + 1,
                    chunk_type="Token",
                    content=content,
                    tokens=len(token_chunk),
                )
            )

        return results

    # =========================================================
    # MARKDOWN
    # =========================================================

    def markdown_chunking(self):

        if not self.text.strip():
            return []

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

        documents = splitter.split_text(
            self.text
        )

        results = []

        for document in documents:

            results.append(
                self._make_chunk(
                    chunk_id=len(results) + 1,
                    chunk_type="Markdown",
                    content=document.page_content,
                    metadata=document.metadata,
                )
            )

        return results

    # =========================================================
    # CONTEXTUAL
    # =========================================================

    def contextual_chunking(self):

        if not self.text.strip():
            return []

        paragraphs = self.text.split(
            "\n\n"
        )

        results = []
        buffer = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(paragraph) < 100:

                buffer += (
                    paragraph + "\n\n"
                )

                continue

            if buffer:

                paragraph = (
                    buffer + paragraph
                )

                buffer = ""

            results.append(
                self._make_chunk(
                    chunk_id=len(results) + 1,
                    chunk_type="Context",
                    content=paragraph,
                )
            )

        if buffer.strip():

            results.append(
                self._make_chunk(
                    chunk_id=len(results) + 1,
                    chunk_type="Context",
                    content=buffer,
                )
            )

        return results

    # =========================================================
    # TABLE
    # =========================================================

    def table_chunking(self):

        results = []

        for table_index, table in enumerate(
            self.tables,
            start=1,
        ):

            if not isinstance(table, dict):
                continue

            rows = (
                table.get("table")
                or table.get("rows")
                or table.get("data")
                or []
            )

            page = table.get(
                "page",
                table.get("sheet", 0),
            )

            table_source = table.get(
                "source",
                self.source,
            )

            lines = [
                "TABLE",
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

            content = "\n".join(lines)

            results.append(
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
                        "sheet": table.get("sheet"),
                    },
                )
            )

        return results

    # =========================================================
    # IMAGE
    # =========================================================

    def image_chunking(self):

        results = []

        for page_data in self.images:

            if not isinstance(page_data, dict):
                continue

            page = page_data.get(
                "page",
                1,
            )

            for image in page_data.get(
                "images",
                [],
            ):

                if not isinstance(image, dict):
                    continue

                ocr_text = str(
                    image.get("ocr_text", "")
                    or ""
                )

                description = str(
                    image.get(
                        "image_description",
                        image.get(
                            "visual_description",
                            "",
                        ),
                    )
                    or ""
                )

                analysis = str(
                    image.get("analysis", "")
                    or ""
                )

                image_no = image.get(
                    "image_no",
                    len(results) + 1,
                )

                parts = [
                    "IMAGE CONTENT",
                    f"Source: {self.source}",
                    f"Page: {page}",
                    f"Image Number: {image_no}",
                ]

                if ocr_text:
                    parts.extend([
                        "",
                        "OCR TEXT:",
                        ocr_text,
                    ])

                if description:
                    parts.extend([
                        "",
                        "VISUAL DESCRIPTION:",
                        description,
                    ])

                if analysis:
                    parts.extend([
                        "",
                        "IMAGE ANALYSIS:",
                        analysis,
                    ])

                results.append(
                    self._make_chunk(
                        chunk_id=len(results) + 1,
                        chunk_type="Image",
                        content="\n".join(parts),
                        page=page,
                        image_path=image.get(
                            "path",
                            "",
                        ),
                        image_no=image_no,
                        ocr_text=ocr_text,
                        visual_description=description,
                        image_analysis=analysis,
                        metadata={
                            "page": page,
                            "image_path": image.get(
                                "path",
                                "",
                            ),
                            "image_no": image_no,
                            "has_ocr": bool(
                                ocr_text.strip()
                            ),
                            "has_visual_analysis": bool(
                                description.strip()
                                or analysis.strip()
                            ),
                        },
                    )
                )

        return results

    # =========================================================
    # VISUAL
    # =========================================================

    def visual_chunking(self):

        results = []

        for visual in self.visuals:

            if not isinstance(visual, dict):
                continue

            page = visual.get(
                "page",
                1,
            )

            ocr_text = str(
                visual.get("ocr_text", "")
                or ""
            )

            description = str(
                visual.get(
                    "image_description",
                    visual.get(
                        "visual_description",
                        "",
                    ),
                )
                or ""
            )

            analysis = str(
                visual.get("analysis", "")
                or ""
            )

            parts = [
                "DOCUMENT VISUAL / DIAGRAM",
                f"Source: {self.source}",
                f"Page: {page}",
            ]

            if ocr_text:
                parts.extend([
                    "",
                    "OCR TEXT:",
                    ocr_text,
                ])

            if description:
                parts.extend([
                    "",
                    "VISUAL DESCRIPTION:",
                    description,
                ])

            if analysis:
                parts.extend([
                    "",
                    "VISUAL ANALYSIS:",
                    analysis,
                ])

            results.append(
                self._make_chunk(
                    chunk_id=len(results) + 1,
                    chunk_type="Visual",
                    content="\n".join(parts),
                    page=page,
                    visual_path=visual.get("path"),
                    visual_type=visual.get(
                        "type",
                        "Document Visual",
                    ),
                    ocr_text=ocr_text,
                    visual_description=description,
                    visual_analysis=analysis,
                )
            )

        return results

    # =========================================================
    # MULTIMODAL
    # =========================================================

    def multimodal_chunking(self):

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

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            chunk["chunk_id"] = index

        return chunks

    # =========================================================
    # GENERIC
    # =========================================================

    def chunk(self, method="Multimodal"):

        methods = {
            "Character": self.character_chunking,
            "Recursive": self.recursive_chunking,
            "Token": self.token_chunking,
            "Markdown": self.markdown_chunking,
            "Context": self.contextual_chunking,
            "Table": self.table_chunking,
            "Image": self.image_chunking,
            "Visual": self.visual_chunking,
            "Multimodal": self.multimodal_chunking,
        }

        if method not in methods:
            raise ValueError(
                f"Unknown chunking method: {method}"
            )

        chunks = methods[method]()

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            chunk["chunk_id"] = index

        return chunks

    # =========================================================
    # STATISTICS
    # =========================================================

    def chunk_statistics(self, chunks):

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

        text_types = {
            "Character",
            "Recursive",
            "Token",
            "Markdown",
            "Context",
        }

        return {
            "Total Chunks": len(chunks),

            "Average Length": round(
                sum(lengths) / len(lengths),
                2,
            ),

            "Maximum Length": max(lengths),

            "Minimum Length": min(lengths),

            "Text Chunks": sum(
                1
                for chunk in chunks
                if chunk.get("chunk_type")
                in text_types
            ),

            "Table Chunks": sum(
                1
                for chunk in chunks
                if chunk.get("chunk_type") == "Table"
            ),

            "Image Chunks": sum(
                1
                for chunk in chunks
                if chunk.get("chunk_type") == "Image"
            ),

            "Visual Chunks": sum(
                1
                for chunk in chunks
                if chunk.get("chunk_type") == "Visual"
            ),
        }

    # =========================================================
    # DATAFRAME
    # =========================================================

    def export_chunks_dataframe(self, chunks):

        rows = []

        for chunk in chunks:

            rows.append({
                "Chunk ID": chunk.get("chunk_id"),
                "Type": chunk.get("chunk_type"),
                "Page": chunk.get("page", ""),
                "Length": len(
                    str(
                        chunk.get(
                            "content",
                            "",
                        )
                    )
                ),
                "Source": chunk.get(
                    "source",
                    self.source,
                ),
                "Content": chunk.get(
                    "content",
                    "",
                ),
            })

        return pd.DataFrame(rows)

    # =========================================================
    # COMPARE
    # =========================================================

    def compare_chunking(self):

        methods = [
            "Character",
            "Recursive",
            "Token",
            "Markdown",
            "Table",
            "Context",
            "Image",
            "Visual",
            "Multimodal",
        ]

        report = {}

        for method in methods:

            chunks = self.chunk(method)

            report[method] = {
                "statistics": self.chunk_statistics(
                    chunks
                ),
                "chunks": chunks,
            }

        return report