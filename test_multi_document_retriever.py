"""
Multi-Document Retriever

Adds multi-document retrieval on top of the existing VectorDatabase.

Behavior:
    No document selection
        -> Search across all available documents.

    Document selection provided
        -> Search only selected document sources.

The existing VectorDatabase is NOT modified.
"""

from typing import Any, Dict, List, Optional


class MultiDocumentRetriever:
    """
    Multi-document retrieval layer.

    Uses the existing VectorDatabase.search() method.
    """

    def __init__(self, vector_db):
        if vector_db is None:
            raise ValueError("vector_db cannot be None.")

        self.vector_db = vector_db

    # ============================================================
    # MAIN RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        user_id: Optional[str],
        query: str,
        selected_document_ids: Optional[List[Any]] = None,
        selected_sources: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks.

        No selection:
            Search across all documents.

        Selection:
            Search only selected document sources.

        NOTE:
            user_id is accepted for compatibility with the
            application architecture. User-level access control
            should continue to be handled by the existing
            application/database layer.
        """

        if not query or not str(query).strip():
            return []

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5

        if top_k <= 0:
            top_k = 5

        sources = self._normalize_sources(
            selected_sources
        )

        # --------------------------------------------------------
        # If document IDs were supplied, try to resolve them
        # against the VectorDB's stored chunk metadata.
        # --------------------------------------------------------

        if selected_document_ids and not sources:
            sources = self._resolve_sources_from_document_ids(
                selected_document_ids
            )

        # --------------------------------------------------------
        # AUTOMATIC MULTI-DOCUMENT SEARCH
        # --------------------------------------------------------

        if not sources:
            return self.vector_db.search(
                query=query,
                top_k=top_k,
                sources=None,
                chunk_types=chunk_types,
            )

        # --------------------------------------------------------
        # SELECTED DOCUMENT SEARCH
        # --------------------------------------------------------

        return self.vector_db.search(
            query=query,
            top_k=top_k,
            sources=sources,
            chunk_types=chunk_types,
        )

    # ============================================================
    # RESOLVE DOCUMENT IDS -> SOURCES
    # ============================================================

    def _resolve_sources_from_document_ids(
        self,
        document_ids: List[Any],
    ) -> List[str]:
        """
        Convert document IDs to source names using the metadata
        already stored inside VectorDatabase.documents.

        This does NOT modify the VectorDatabase.
        """

        normalized_ids = {
            str(value).strip()
            for value in document_ids
            if value is not None
            and str(value).strip()
        }

        if not normalized_ids:
            return []

        documents = getattr(
            self.vector_db,
            "documents",
            [],
        )

        sources = []

        for document in documents:

            if not isinstance(document, dict):
                continue

            stored_id = (
                document.get("document_id")
                or document.get("doc_id")
                or document.get("id")
            )

            if stored_id is None:
                continue

            if str(stored_id).strip() not in normalized_ids:
                continue

            source = document.get("source")

            if source:
                source = str(source).strip()

                if source and source not in sources:
                    sources.append(source)

        return sources

    # ============================================================
    # SEARCH SCOPE
    # ============================================================

    def get_search_scope(
        self,
        selected_document_ids: Optional[List[Any]] = None,
        selected_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Return the current retrieval scope.
        """

        document_ids = self._normalize_document_ids(
            selected_document_ids
        )

        sources = self._normalize_sources(
            selected_sources
        )

        if document_ids or sources:
            return {
                "mode": "selected_documents",
                "document_ids": document_ids,
                "sources": sources,
                "automatic": False,
            }

        return {
            "mode": "all_documents",
            "document_ids": [],
            "sources": [],
            "automatic": True,
        }

    # ============================================================
    # NORMALIZE DOCUMENT IDS
    # ============================================================

    @staticmethod
    def _normalize_document_ids(
        document_ids: Optional[List[Any]],
    ) -> List[Any]:
        """
        Remove empty and duplicate document IDs.
        """

        if document_ids is None:
            return []

        if isinstance(
            document_ids,
            (list, tuple, set),
        ):
            values = list(document_ids)
        else:
            values = [document_ids]

        normalized = []

        for document_id in values:

            if document_id is None:
                continue

            if isinstance(document_id, str):
                document_id = document_id.strip()

                if not document_id:
                    continue

            if document_id not in normalized:
                normalized.append(document_id)

        return normalized

    # ============================================================
    # NORMALIZE SOURCES
    # ============================================================

    @staticmethod
    def _normalize_sources(
        sources: Optional[List[str]],
    ) -> List[str]:
        """
        Remove empty and duplicate source names.
        """

        if sources is None:
            return []

        if isinstance(
            sources,
            (list, tuple, set),
        ):
            values = list(sources)
        else:
            values = [sources]

        normalized = []

        for source in values:

            if source is None:
                continue

            source = str(source).strip()

            if not source:
                continue

            if source not in normalized:
                normalized.append(source)

        return normalized


# ================================================================
# CONVENIENCE FUNCTION
# ================================================================

def multi_document_search(
    vector_db,
    user_id: Optional[str],
    query: str,
    selected_document_ids: Optional[List[Any]] = None,
    selected_sources: Optional[List[str]] = None,
    chunk_types: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper for MultiDocumentRetriever.
    """

    retriever = MultiDocumentRetriever(
        vector_db=vector_db
    )

    return retriever.retrieve(
        user_id=user_id,
        query=query,
        selected_document_ids=selected_document_ids,
        selected_sources=selected_sources,
        chunk_types=chunk_types,
        top_k=top_k,
    )