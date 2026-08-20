import hashlib
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, VECTOR_DB_PATH


class VectorDatabase:
    """
    Persistent FAISS vector database for Enterprise Document Intelligence.

    Supports:
        - normal text chunks
        - OCR chunks
        - image chunks
        - visual/diagram chunks
        - table chunks
        - multiple documents in one database
        - add / replace / remove documents
        - persistent FAISS index
        - persistent chunk metadata
        - source filtering
        - document hashing
        - backward compatibility with the existing app.py API

    Important:
        The actual searchable representation is built from more than
        just `content`.

        A chunk can contain:

            content
            ocr_text
            image_description
            visual_description
            caption
            table_text

        This allows the agent to retrieve information contained inside
        images, diagrams, screenshots and scanned pages after extractor.py
        has produced OCR/visual information.
    """

    DATABASE_VERSION = 2

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.index = None
        self.documents: List[Dict[str, Any]] = []

        os.makedirs(
            VECTOR_DB_PATH,
            exist_ok=True,
        )

        self.index_path = os.path.join(
            VECTOR_DB_PATH,
            "faiss.index",
        )

        self.documents_path = os.path.join(
            VECTOR_DB_PATH,
            "documents.pkl",
        )

        self.metadata_path = os.path.join(
            VECTOR_DB_PATH,
            "database_metadata.json",
        )

    # ============================================================
    # NORMALIZATION HELPERS
    # ============================================================

    @staticmethod
    def _normalize_sources(
        sources: Optional[Iterable[Any]],
    ) -> List[str]:
        normalized = set()

        if not sources:
            return []

        if isinstance(
            sources,
            (str, os.PathLike),
        ):
            sources = [sources]

        for source in sources:
            if source is None:
                continue

            for part in str(source).split(","):
                part = part.strip()

                if part:
                    normalized.add(part)

        return sorted(normalized)

    @staticmethod
    def _document_key(
        value: Any,
    ) -> str:
        """
        Stable key for matching document names.

        Only the basename is used so that:

            /tmp/a/report.pdf
            C:/uploads/report.pdf
            report.pdf

        all refer to the same logical document.
        """
        if not value:
            return ""

        return os.path.basename(
            str(value).strip()
        ).casefold()

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 0,
    ) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        if isinstance(
            value,
            (dict, list, tuple),
        ):
            try:
                return json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except Exception:
                return str(value)

        return str(value).strip()

    # ============================================================
    # SEARCHABLE CONTENT
    # ============================================================

    @classmethod
    def _build_search_text(
        cls,
        chunk: Dict[str, Any],
    ) -> str:
        """
        Build the semantic representation used for embedding.

        This is intentionally richer than:

            chunk["content"]

        because image-aware extraction can provide information in:

            OCR
            image descriptions
            visual descriptions
            captions
            table text
            figure labels
        """

        if not isinstance(
            chunk,
            dict,
        ):
            return ""

        sections = []

        def add(
            label: str,
            value: Any,
        ):
            text = cls._clean_text(value)

            if text:
                sections.append(
                    f"{label}: {text}"
                )

        add(
            "Content",
            chunk.get("content"),
        )

        add(
            "OCR text",
            chunk.get("ocr_text"),
        )

        add(
            "OCR",
            chunk.get("ocr"),
        )

        add(
            "Image description",
            chunk.get("image_description"),
        )

        add(
            "Visual description",
            chunk.get("visual_description"),
        )

        add(
            "Visual analysis",
            chunk.get("visual_analysis"),
        )

        add(
            "Description",
            chunk.get("description"),
        )

        add(
            "Caption",
            chunk.get("caption"),
        )

        add(
            "Table text",
            chunk.get("table_text"),
        )

        add(
            "Figure text",
            chunk.get("figure_text"),
        )

        add(
            "Labels",
            chunk.get("labels"),
        )

        add(
            "Title",
            chunk.get("title"),
        )

        add(
            "Heading",
            chunk.get("heading"),
        )

        add(
            "Image path",
            chunk.get("image_path"),
        )

        add(
            "Visual path",
            chunk.get("visual_path"),
        )

        return "\n".join(
            sections
        ).strip()

    # ============================================================
    # DOCUMENT ID
    # ============================================================

    @classmethod
    def _ensure_document_id(
        cls,
        chunk: Dict[str, Any],
        source: Optional[str] = None,
    ):
        """
        Give every chunk a stable document_id.

        Existing document_id values are preserved.
        """

        if chunk.get("document_id"):
            return

        resolved_source = (
            source
            or chunk.get("source")
            or chunk.get("file_name")
            or "Unknown"
        )

        chunk["document_id"] = (
            cls._document_key(
                resolved_source
            )
            or hashlib.sha256(
                str(resolved_source).encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
        )

    # ============================================================
    # CHUNK NORMALIZATION
    # ============================================================

    @classmethod
    def _normalize_chunk(
        cls,
        chunk: Dict[str, Any],
        source: Optional[str] = None,
        chunk_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Normalize a chunk without throwing away metadata generated by
        extractor.py or chunking.py.
        """

        normalized = dict(chunk)

        if source:
            if not normalized.get("source"):
                normalized["source"] = source

        if not normalized.get("source"):
            normalized["source"] = (
                normalized.get("file_name")
                or "Unknown"
            )

        cls._ensure_document_id(
            normalized,
            source=source,
        )

        if (
            chunk_number is not None
            and not normalized.get("chunk_id")
        ):
            normalized["chunk_id"] = (
                chunk_number
            )

        if not normalized.get(
            "chunk_type"
        ):
            normalized["chunk_type"] = "Text"

        search_text = cls._build_search_text(
            normalized
        )

        normalized["_search_text"] = (
            search_text
        )

        return normalized

    # ============================================================
    # CONTENT HASH
    # ============================================================

    @classmethod
    def _content_hash(
        cls,
        chunks: Sequence[Dict[str, Any]],
    ) -> str:
        """
        Stable hash for a document.

        Includes semantic content and important image/OCR metadata.
        """

        normalized = []

        for chunk in chunks or []:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            normalized.append(
                {
                    "source": cls._clean_text(
                        chunk.get("source")
                    ),
                    "document_id": cls._clean_text(
                        chunk.get("document_id")
                    ),
                    "chunk_id": cls._clean_text(
                        chunk.get("chunk_id")
                    ),
                    "chunk_type": cls._clean_text(
                        chunk.get("chunk_type")
                    ),
                    "content": cls._clean_text(
                        chunk.get("content")
                    ),
                    "ocr_text": cls._clean_text(
                        chunk.get("ocr_text")
                    ),
                    "ocr": cls._clean_text(
                        chunk.get("ocr")
                    ),
                    "image_description": cls._clean_text(
                        chunk.get(
                            "image_description"
                        )
                    ),
                    "visual_description": cls._clean_text(
                        chunk.get(
                            "visual_description"
                        )
                    ),
                    "visual_analysis": cls._clean_text(
                        chunk.get(
                            "visual_analysis"
                        )
                    ),
                    "caption": cls._clean_text(
                        chunk.get("caption")
                    ),
                    "table_text": cls._clean_text(
                        chunk.get("table_text")
                    ),
                    "page": cls._clean_text(
                        chunk.get("page")
                    ),
                }
            )

        payload = json.dumps(
            normalized,
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    # ============================================================
    # DOCUMENT GROUPING
    # ============================================================

    def _group_documents_by_source(
        self,
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for chunk in self.documents:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            source = str(
                chunk.get(
                    "source",
                    "",
                )
            ).strip()

            if not source:
                source = "Unknown"

            grouped.setdefault(
                source,
                [],
            ).append(chunk)

        return grouped

    # ============================================================
    # EMBEDDINGS
    # ============================================================

    def generate_embeddings(
        self,
        chunks,
    ):
        """
        Generate embeddings for chunks.

        The embedding input includes:
            content
            OCR
            image descriptions
            visual analysis
            tables
            captions

        Therefore an image can be retrieved even when the answer is
        not present in ordinary document text.
        """

        if not chunks:
            raise ValueError(
                "No chunks available for embedding."
            )

        search_texts = []

        for chunk in chunks:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            text = self._build_search_text(
                chunk
            )

            if not text:
                text = "Empty document chunk."

            search_texts.append(
                text
            )

        if not search_texts:
            raise ValueError(
                "No valid searchable content available."
            )

        embeddings = self.model.encode(
            search_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding model returned invalid "
                "embedding dimensions."
            )

        return embeddings

    # ============================================================
    # REBUILD INDEX
    # ============================================================

    def _rebuild_from_documents(
        self,
        persist: bool = False,
    ):
        """
        Rebuild FAISS from the current document list.

        IndexFlatL2 is intentionally retained for compatibility with
        the current project.

        For the current project size this gives simple, deterministic
        retrieval and makes document replacement/removal straightforward.
        """

        if not self.documents:
            self.index = None

            if persist:
                self.save()

            return

        embeddings = self.generate_embeddings(
            self.documents
        )

        self.index = faiss.IndexFlatL2(
            int(embeddings.shape[1])
        )

        self.index.add(
            embeddings
        )

        if persist:
            self.save()

    # ============================================================
    # CREATE INDEX
    # ============================================================

    def create_index(
        self,
        chunks,
    ):
        """
        Replace the entire local vector database.
        """

        if not chunks:
            raise ValueError(
                "No chunks available to index."
            )

        valid_chunks = []

        for number, chunk in enumerate(
            chunks,
            start=1,
        ):
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            valid_chunks.append(
                self._normalize_chunk(
                    chunk,
                    chunk_number=number,
                )
            )

        if not valid_chunks:
            raise ValueError(
                "No valid chunks available to index."
            )

        self.documents = (
            valid_chunks
        )

        self._rebuild_from_documents()

        self.save()

        print(
            f"Indexed {len(self.documents)} chunks."
        )

        return self.get_stats()

    # ============================================================
    # ADD DOCUMENT
    # ============================================================

    def add_document(
        self,
        chunks,
        source=None,
    ):
        """
        Add one document without deleting other stored documents.

        If the same source already exists:
            status = exists

        The caller can then decide whether to replace it.
        """

        if not chunks:
            raise ValueError(
                "No chunks available."
            )

        incoming = []

        for number, chunk in enumerate(
            chunks,
            start=1,
        ):
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            incoming.append(
                self._normalize_chunk(
                    chunk,
                    source=source,
                    chunk_number=number,
                )
            )

        if not incoming:
            raise ValueError(
                "No valid chunks available."
            )

        incoming_source = (
            source
            or incoming[0].get(
                "source",
                "",
            )
        )

        if (
            incoming_source
            and self.document_exists(
                incoming_source
            )
        ):
            existing_hash = (
                self.get_document_hash(
                    incoming_source
                )
            )

            incoming_hash = (
                self._content_hash(
                    incoming
                )
            )

            if (
                existing_hash
                and existing_hash == incoming_hash
            ):
                return {
                    "status": "exists",
                    "source": incoming_source,
                    "added_chunks": 0,
                    "message": (
                        "This document is already indexed "
                        "and its content has not changed."
                    ),
                }

            return {
                "status": "exists",
                "source": incoming_source,
                "added_chunks": 0,
                "message": (
                    "A document with this name already exists. "
                    "Use replace_document() to replace it."
                ),
            }

        self.documents.extend(
            incoming
        )

        self._rebuild_from_documents()

        self.save()

        return {
            "status": "added",
            "source": incoming_source,
            "added_chunks": len(incoming),
            "document_hash": self._content_hash(
                incoming
            ),
        }

    # ============================================================
    # UPSERT DOCUMENT
    # ============================================================

    def upsert_document(
        self,
        chunks,
        source=None,
    ):
        """
        Add a document if new, otherwise replace it.

        Useful for applications where a user uploads a newer version
        of the same file.
        """

        if not chunks:
            raise ValueError(
                "No chunks available."
            )

        incoming_source = (
            source
            or (
                chunks[0].get("source")
                if isinstance(
                    chunks[0],
                    dict,
                )
                else None
            )
            or "Unknown"
        )

        if self.document_exists(
            incoming_source
        ):
            return self.replace_document(
                incoming_source,
                chunks,
            )

        return self.add_document(
            chunks,
            source=incoming_source,
        )

    # ============================================================
    # REMOVE DOCUMENT
    # ============================================================

    def remove_document(
        self,
        source,
    ):
        """
        Remove every chunk belonging to one source.
        """

        if not source:
            return {
                "status": "not_found",
                "removed_chunks": 0,
            }

        target = self._document_key(
            source
        )

        kept = []
        removed = 0

        for chunk in self.documents:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            chunk_source = (
                self._document_key(
                    chunk.get(
                        "source",
                        "",
                    )
                )
            )

            if chunk_source == target:
                removed += 1
            else:
                kept.append(
                    chunk
                )

        if removed == 0:
            return {
                "status": "not_found",
                "source": source,
                "removed_chunks": 0,
            }

        self.documents = kept

        self._rebuild_from_documents()

        self.save()

        return {
            "status": "removed",
            "source": source,
            "removed_chunks": removed,
        }

    # ============================================================
    # REPLACE DOCUMENT
    # ============================================================

    def replace_document(
        self,
        source,
        chunks,
    ):
        """
        Replace one document while preserving every other document.
        """

        if not chunks:
            raise ValueError(
                "No replacement chunks available."
            )

        target = self._document_key(
            source
        )

        kept = []

        for chunk in self.documents:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            chunk_source = (
                self._document_key(
                    chunk.get(
                        "source",
                        "",
                    )
                )
            )

            if chunk_source != target:
                kept.append(
                    chunk
                )

        replacement = []

        for number, chunk in enumerate(
            chunks,
            start=1,
        ):
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            normalized = self._normalize_chunk(
                chunk,
                source=source,
                chunk_number=number,
            )

            normalized["source"] = (
                source
            )

            replacement.append(
                normalized
            )

        if not replacement:
            raise ValueError(
                "No valid replacement chunks available."
            )

        self.documents = (
            kept + replacement
        )

        self._rebuild_from_documents()

        self.save()

        return {
            "status": "replaced",
            "source": source,
            "replacement_chunks": len(
                replacement
            ),
            "document_hash": self._content_hash(
                replacement
            ),
        }

    # ============================================================
    # SAVE
    # ============================================================

    def save(self):
        """
        Persist:
            FAISS index
            document/chunk metadata
            database metadata
        """

        os.makedirs(
            VECTOR_DB_PATH,
            exist_ok=True,
        )

        if self.index is not None:
            faiss.write_index(
                self.index,
                self.index_path,
            )

        elif os.path.exists(
            self.index_path
        ):
            os.remove(
                self.index_path
            )

        with open(
            self.documents_path,
            "wb",
        ) as file:
            pickle.dump(
                self.documents,
                file,
            )

        metadata = {
            "database_version":
                self.DATABASE_VERSION,

            "sources":
                self.get_sources(),

            "chunks":
                len(self.documents),

            "documents":
                len(self.get_sources()),

            "dimension":
                (
                    int(self.index.d)
                    if self.index is not None
                    else 0
                ),

            "index_type":
                "FAISS IndexFlatL2",

            "embedding_model":
                EMBEDDING_MODEL,

            "document_hashes":
                self.get_document_hashes(),

            "chunk_types":
                self.get_chunk_type_counts(),

            "image_chunks":
                self.get_chunk_type_counts().get(
                    "Image",
                    0,
                ),

            "ocr_chunks":
                self.get_chunk_type_counts().get(
                    "OCR",
                    0,
                ),

            "visual_chunks":
                self.get_chunk_type_counts().get(
                    "Visual",
                    0,
                ),

            "table_chunks":
                self.get_chunk_type_counts().get(
                    "Table",
                    0,
                ),
        }

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            "Vector Database Saved."
        )

    # ============================================================
    # EXISTS
    # ============================================================

    def exists(self):
        return (
            os.path.isfile(
                self.index_path
            )
            and os.path.isfile(
                self.documents_path
            )
            and os.path.isfile(
                self.metadata_path
            )
        )

    # ============================================================
    # LOAD
    # ============================================================

    def load(self):
        """
        Load the persistent database.

        The stored `_search_text` field is accepted for backward
        compatibility, but current searchable text is regenerated
        from the chunk metadata when embeddings are created.
        """

        if not self.exists():
            raise FileNotFoundError(
                "Vector Database not found."
            )

        self.index = faiss.read_index(
            self.index_path
        )

        with open(
            self.documents_path,
            "rb",
        ) as file:
            loaded = pickle.load(
                file
            )

        if not isinstance(
            loaded,
            list,
        ):
            loaded = list(
                loaded
            )

        normalized = []

        for number, chunk in enumerate(
            loaded,
            start=1,
        ):
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            normalized.append(
                self._normalize_chunk(
                    chunk,
                    chunk_number=number,
                )
            )

        self.documents = normalized

        # If FAISS and metadata became inconsistent, rebuild.
        if (
            self.index is not None
            and self.index.ntotal
            != len(self.documents)
        ):
            self._rebuild_from_documents()

        print(
            f"Vector Database Loaded: "
            f"{len(self.documents)} chunks."
        )

        return self.documents

    # ============================================================
    # SOURCES
    # ============================================================

    def get_sources(self):
        sources = []

        for chunk in self.documents:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            source = chunk.get(
                "source"
            )

            if source:
                sources.append(
                    str(source)
                )

        return self._normalize_sources(
            sources
        )

    def get_saved_sources(self):
        """
        Read sources directly from metadata.
        """

        if not os.path.isfile(
            self.metadata_path
        ):
            return []

        try:
            with open(
                self.metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(
                    file
                )

            sources = metadata.get(
                "sources",
                [],
            )

            if not isinstance(
                sources,
                list,
            ):
                return []

            return self._normalize_sources(
                sources
            )

        except Exception:
            return []

    # ============================================================
    # DOCUMENT HASHES
    # ============================================================

    def get_document_hashes(self):
        """
        Return:

            {
                "document.pdf": "<sha256>",
                "report.docx": "<sha256>"
            }
        """

        hashes = {}

        grouped = (
            self._group_documents_by_source()
        )

        for source, chunks in grouped.items():
            hashes[source] = (
                self._content_hash(
                    chunks
                )
            )

        return hashes

    def get_document_hash(
        self,
        source,
    ):
        target = self._document_key(
            source
        )

        for saved_source, value in (
            self.get_document_hashes().items()
        ):
            if (
                self._document_key(
                    saved_source
                )
                == target
            ):
                return value

        return None

    # ============================================================
    # DOCUMENT STATUS
    # ============================================================

    def document_exists(
        self,
        source,
    ):
        target = self._document_key(
            source
        )

        return any(
            self._document_key(
                saved_source
            )
            == target
            for saved_source in self.get_saved_sources()
        )

    # ============================================================
    # SOURCE COMPATIBILITY
    # ============================================================

    def matches_sources(
        self,
        current_sources,
    ):
        """
        Compatibility method for older app.py.

        It intentionally compares source names only.
        """

        current = {
            self._document_key(
                source
            )
            for source in self._normalize_sources(
                current_sources
            )
        }

        saved = {
            self._document_key(
                source
            )
            for source in self.get_saved_sources()
        }

        return current == saved

    # ============================================================
    # CHUNK TYPE STATISTICS
    # ============================================================

    def get_chunk_type_counts(
        self,
    ):
        counts = {}

        for chunk in self.documents:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            chunk_type = str(
                chunk.get(
                    "chunk_type",
                    "Unknown",
                )
            )

            counts[chunk_type] = (
                counts.get(
                    chunk_type,
                    0,
                )
                + 1
            )

        return counts

    # ============================================================
    # IMAGE / VISUAL STATISTICS
    # ============================================================

    def get_visual_stats(self):
        """
        Statistics specifically for the image-aware pipeline.
        """

        stats = {
            "image_chunks": 0,
            "ocr_chunks": 0,
            "visual_chunks": 0,
            "chunks_with_images": 0,
            "chunks_with_ocr": 0,
            "chunks_with_visual_analysis": 0,
        }

        for chunk in self.documents:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            chunk_type = str(
                chunk.get(
                    "chunk_type",
                    "",
                )
            ).casefold()

            if chunk_type == "image":
                stats[
                    "image_chunks"
                ] += 1

            if chunk_type == "ocr":
                stats[
                    "ocr_chunks"
                ] += 1

            if chunk_type == "visual":
                stats[
                    "visual_chunks"
                ] += 1

            if (
                chunk.get(
                    "image_path"
                )
                or chunk.get(
                    "image"
                )
            ):
                stats[
                    "chunks_with_images"
                ] += 1

            if (
                chunk.get(
                    "ocr_text"
                )
                or chunk.get(
                    "ocr"
                )
            ):
                stats[
                    "chunks_with_ocr"
                ] += 1

            if (
                chunk.get(
                    "visual_description"
                )
                or chunk.get(
                    "visual_analysis"
                )
            ):
                stats[
                    "chunks_with_visual_analysis"
                ] += 1

        return stats

    # ============================================================
    # DATABASE STATISTICS
    # ============================================================

    def get_stats(self):
        stats = {
            "status": "Not Created",
            "chunks": 0,
            "dimension": 0,
            "documents": 0,
            "index_type":
                "FAISS IndexFlatL2",
            "embedding_model":
                EMBEDDING_MODEL,
            "path":
                VECTOR_DB_PATH,
            "sources": [],
            "chunk_types": {},
            "visual": {
                "image_chunks": 0,
                "ocr_chunks": 0,
                "visual_chunks": 0,
                "chunks_with_images": 0,
                "chunks_with_ocr": 0,
                "chunks_with_visual_analysis": 0,
            },
        }

        if not self.exists():
            return stats

        try:
            index = faiss.read_index(
                self.index_path
            )

            with open(
                self.documents_path,
                "rb",
            ) as file:
                documents = pickle.load(
                    file
                )

            if not isinstance(
                documents,
                list,
            ):
                documents = list(
                    documents
                )

            sources = self._normalize_sources(
                [
                    d.get("source")
                    for d in documents
                    if isinstance(
                        d,
                        dict,
                    )
                ]
            )

            stats["status"] = (
                "Ready"
            )

            stats["chunks"] = int(
                index.ntotal
            )

            stats["dimension"] = int(
                index.d
            )

            stats["documents"] = len(
                sources
            )

            stats["sources"] = (
                sources
            )

            stats["chunk_types"] = (
                self._count_chunk_types(
                    documents
                )
            )

            stats["embedding_model"] = (
                EMBEDDING_MODEL
            )

            stats["visual"] = (
                self._calculate_visual_stats(
                    documents
                )
            )

        except Exception as e:
            stats["status"] = (
                "Error"
            )

            stats["error"] = str(
                e
            )

        return stats

    @staticmethod
    def _count_chunk_types(
        documents,
    ):
        counts = {}

        for chunk in documents:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            chunk_type = str(
                chunk.get(
                    "chunk_type",
                    "Unknown",
                )
            )

            counts[chunk_type] = (
                counts.get(
                    chunk_type,
                    0,
                )
                + 1
            )

        return counts

    @staticmethod
    def _calculate_visual_stats(
        documents,
    ):
        stats = {
            "image_chunks": 0,
            "ocr_chunks": 0,
            "visual_chunks": 0,
            "chunks_with_images": 0,
            "chunks_with_ocr": 0,
            "chunks_with_visual_analysis": 0,
        }

        for chunk in documents:
            if not isinstance(
                chunk,
                dict,
            ):
                continue

            chunk_type = str(
                chunk.get(
                    "chunk_type",
                    "",
                )
            ).casefold()

            if chunk_type == "image":
                stats[
                    "image_chunks"
                ] += 1

            if chunk_type == "ocr":
                stats[
                    "ocr_chunks"
                ] += 1

            if chunk_type == "visual":
                stats[
                    "visual_chunks"
                ] += 1

            if (
                chunk.get(
                    "image_path"
                )
                or chunk.get(
                    "image"
                )
            ):
                stats[
                    "chunks_with_images"
                ] += 1

            if (
                chunk.get(
                    "ocr_text"
                )
                or chunk.get(
                    "ocr"
                )
            ):
                stats[
                    "chunks_with_ocr"
                ] += 1

            if (
                chunk.get(
                    "visual_description"
                )
                or chunk.get(
                    "visual_analysis"
                )
            ):
                stats[
                    "chunks_with_visual_analysis"
                ] += 1

        return stats

    # ============================================================
    # CLEAR
    # ============================================================

    def clear(self):
        """
        Clear the complete local vector database.

        Does NOT delete documents from Supabase Storage.
        """

        for file_path in [
            self.index_path,
            self.documents_path,
            self.metadata_path,
        ]:
            if os.path.exists(
                file_path
            ):
                os.remove(
                    file_path
                )

        self.index = None
        self.documents = []

        print(
            "Vector Database Cleared."
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query,
        top_k=5,
        sources=None,
        chunk_types=None,
    ):
        """
        Semantic search across all persisted documents.

        Parameters
        ----------
        query:
            User question.

        top_k:
            Number of final results.

        sources:
            Optional source/document filter.

        chunk_types:
            Optional chunk-type filter.

        Example:

            db.search(
                "What does the architecture diagram show?",
                top_k=5,
                chunk_types=[
                    "Image",
                    "Visual",
                    "OCR",
                ],
            )
        """

        if not query or not str(
            query
        ).strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if self.index is None:

            if self.exists():
                self.load()

            else:
                return []

        if (
            self.index is None
            or self.index.ntotal == 0
        ):
            return []

        # --------------------------------------------------------
        # Source filter
        # --------------------------------------------------------

        allowed_sources = None

        if sources:

            allowed_sources = {
                self._document_key(
                    source
                )
                for source in self._normalize_sources(
                    sources
                )
            }

        # --------------------------------------------------------
        # Chunk type filter
        # --------------------------------------------------------

        allowed_types = None

        if chunk_types:

            if isinstance(
                chunk_types,
                str,
            ):
                chunk_types = [
                    chunk_types
                ]

            allowed_types = {
                str(value).casefold()
                for value in chunk_types
                if value
            }

        # --------------------------------------------------------
        # Query embedding
        # --------------------------------------------------------

        query_embedding = (
            self.model.encode(
                [str(query)],
                convert_to_numpy=True,
            )
            .astype("float32")
        )

        # --------------------------------------------------------
        # Candidate retrieval
        # --------------------------------------------------------

        requested_k = max(
            1,
            int(top_k),
        )

        # Retrieve a larger candidate set when filtering.
        multiplier = 10 if (
            allowed_sources
            or allowed_types
        ) else 3

        candidate_k = min(
            max(
                requested_k * multiplier,
                requested_k,
            ),
            self.index.ntotal,
        )

        distances, indices = (
            self.index.search(
                query_embedding,
                candidate_k,
            )
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0],
        ):

            if index < 0:
                continue

            if index >= len(
                self.documents
            ):
                continue

            chunk = dict(
                self.documents[index]
            )

            # ----------------------------------------------------
            # Source filtering
            # ----------------------------------------------------

            chunk_source = (
                self._document_key(
                    chunk.get(
                        "source",
                        "",
                    )
                )
            )

            if (
                allowed_sources
                is not None
                and chunk_source
                not in allowed_sources
            ):
                continue

            # ----------------------------------------------------
            # Chunk type filtering
            # ----------------------------------------------------

            chunk_type = str(
                chunk.get(
                    "chunk_type",
                    "",
                )
            ).casefold()

            if (
                allowed_types
                is not None
                and chunk_type
                not in allowed_types
            ):
                continue

            # ----------------------------------------------------
            # Score
            # ----------------------------------------------------

            distance = float(
                distance
            )

            similarity_score = (
                1.0
                / (
                    1.0
                    + max(
                        distance,
                        0.0,
                    )
                )
            )

            chunk[
                "similarity_score"
            ] = similarity_score

            chunk[
                "distance"
            ] = distance

            # Make the searchable representation available to
            # the agent for debugging/transparency.
            chunk[
                "search_text"
            ] = self._build_search_text(
                chunk
            )

            results.append(
                chunk
            )

            if len(
                results
            ) >= requested_k:
                break

        return results

    # ============================================================
    # IMAGE-SPECIFIC SEARCH
    # ============================================================

    def search_images(
        self,
        query,
        top_k=5,
        sources=None,
    ):
        """
        Search specifically through image/visual/OCR chunks.
        """

        return self.search(
            query=query,
            top_k=top_k,
            sources=sources,
            chunk_types=[
                "Image",
                "Visual",
                "OCR",
                "image",
                "visual",
                "ocr",
            ],
        )

    # ============================================================
    # TABLE-SPECIFIC SEARCH
    # ============================================================

    def search_tables(
        self,
        query,
        top_k=5,
        sources=None,
    ):
        """
        Search specifically through table chunks.
        """

        return self.search(
            query=query,
            top_k=top_k,
            sources=sources,
            chunk_types=[
                "Table",
                "table",
            ],
        )

    # ============================================================
    # DOCUMENT-SPECIFIC SEARCH
    # ============================================================

    def search_document(
        self,
        query,
        source,
        top_k=5,
    ):
        """
        Search within one stored document.
        """

        return self.search(
            query=query,
            top_k=top_k,
            sources=[
                source
            ],
        )

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    def display_results(
        self,
        results,
    ):
        print(
            "=" * 70
        )

        if not results:
            print(
                "No results found."
            )

            print(
                "=" * 70
            )

            return

        for number, chunk in enumerate(
            results,
            start=1,
        ):

            print(
                f"\nResult #{number}"
            )

            print(
                f"Chunk ID : "
                f"{chunk.get('chunk_id', 'N/A')}"
            )

            print(
                f"Document : "
                f"{chunk.get('document_id', 'N/A')}"
            )

            print(
                f"Source   : "
                f"{chunk.get('source', 'N/A')}"
            )

            print(
                f"Page     : "
                f"{chunk.get('page', 'N/A')}"
            )

            print(
                f"Type     : "
                f"{chunk.get('chunk_type', 'N/A')}"
            )

            if (
                chunk.get(
                    "image_path"
                )
            ):
                print(
                    f"Image    : "
                    f"{chunk.get('image_path')}"
                )

            if (
                chunk.get(
                    "visual_path"
                )
            ):
                print(
                    f"Visual   : "
                    f"{chunk.get('visual_path')}"
                )

            if "similarity_score" in chunk:
                print(
                    f"Score    : "
                    f"{chunk['similarity_score']:.4f}"
                )

            if "distance" in chunk:
                print(
                    f"Distance : "
                    f"{chunk['distance']:.4f}"
                )

            content = (
                chunk.get(
                    "content",
                    "",
                )
                or chunk.get(
                    "ocr_text",
                    "",
                )
                or chunk.get(
                    "visual_description",
                    "",
                )
            )

            print(
                str(content)[:1000]
            )

            print(
                "-" * 70
            )


# ================================================================
# STANDALONE TEST
# ================================================================

if __name__ == "__main__":

    database = VectorDatabase()

    print(
        "\n========== VECTOR DATABASE =========="
    )

    print(
        database.get_stats()
    )

    if database.exists():

        try:

            database.load()

            print(
                "\n========== SOURCES =========="
            )

            print(
                database.get_sources()
            )

            print(
                "\n========== CHUNK TYPES =========="
            )

            print(
                database.get_chunk_type_counts()
            )

            print(
                "\n========== VISUAL STATS =========="
            )

            print(
                database.get_visual_stats()
            )

        except Exception as error:

            print(
                f"Error loading vector database: "
                f"{error}"
            )

    else:

        print(
            "Vector database has not been created yet."
        )