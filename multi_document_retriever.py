"""
Multi-Document Retriever

Provides intelligent retrieval across multiple user documents
while preserving the existing Vector DB implementation.

Behavior:
    No document filter
        -> Search across all documents available to the user.

    Document filter provided
        -> Search only within the selected documents.

This module does NOT replace VectorDB.
It acts as a retrieval layer on top of the existing vector database.
"""

from typing import Any, Dict, List, Optional


class MultiDocumentRetriever:
    """
    Retrieval layer for multi-document RAG.

    The existing Vector DB remains responsible for:
        - Embedding generation
        - FAISS/vector search
        - Similarity retrieval
        - Metadata handling
        - Document/chunk filtering

    This class is responsible for:
        - Deciding whether retrieval is global or filtered
        - Passing the correct document scope
        - Returning retrieved evidence
    """

    def __init__(self, vector_db):
        """
        Initialize the retriever.

        Parameters
        ----------
        vector_db:
            Existing Vector DB instance.
        """

        if vector_db is None:
            raise ValueError(
                "vector_db cannot be None."
            )

        self.vector_db = vector_db

    # ============================================================
    # MAIN RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        user_id: str,
        query: str,
        selected_document_ids: Optional[List[Any]] = None,
        chunk_types: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a user.

        Parameters
        ----------
        user_id:
            Current authenticated user's ID.

        query:
            User's natural-language question.

        selected_document_ids:
            Optional list of document IDs selected by the user.

            If provided:
                Search only selected documents.

            If empty/None:
                Search across all documents belonging to the user.

        chunk_types:
            Optional chunk-type restriction.

        top_k:
            Maximum number of results.

        Returns
        -------
        List[Dict[str, Any]]
            Retrieved evidence/chunks from the existing Vector DB.
        """

        # --------------------------------------------------------
        # Validate query
        # --------------------------------------------------------

        if not query or not str(query).strip():
            return []

        # --------------------------------------------------------
        # Validate user
        # --------------------------------------------------------

        if user_id is None or not str(user_id).strip():
            raise ValueError(
                "user_id is required for multi-document retrieval."
            )

        # --------------------------------------------------------
        # Normalize document selection
        # --------------------------------------------------------

        document_ids = self._normalize_document_ids(
            selected_document_ids
        )

        # --------------------------------------------------------
        # Automatic multi-document retrieval
        #
        # No document selected:
        # search across all documents belonging to the user.
        # --------------------------------------------------------

        if not document_ids:

            return self._search_user_documents(
                user_id=user_id,
                query=query,
                chunk_types=chunk_types,
                top_k=top_k,
            )

        # --------------------------------------------------------
        # Optional manual document filter
        #
        # User selected one or more documents:
        # search only those documents.
        # --------------------------------------------------------

        return self._search_selected_documents(
            user_id=user_id,
            query=query,
            selected_document_ids=document_ids,
            chunk_types=chunk_types,
            top_k=top_k,
        )

    # ============================================================
    # SEARCH ALL USER DOCUMENTS
    # ============================================================

    def _search_user_documents(
        self,
        user_id: str,
        query: str,
        chunk_types: Optional[List[str]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Search across all documents belonging to the user.

        This is the default behavior when the user has not
        manually selected specific documents.
        """

        return self.vector_db.search_chunks(
            user_id=user_id,
            query=query,
            selected_document_ids=None,
            chunk_types=chunk_types,
            top_k=top_k,
        )

    # ============================================================
    # SEARCH SELECTED DOCUMENTS
    # ============================================================

    def _search_selected_documents(
        self,
        user_id: str,
        query: str,
        selected_document_ids: List[Any],
        chunk_types: Optional[List[str]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Search only within documents explicitly selected
        by the user.
        """

        return self.vector_db.search_chunks(
            user_id=user_id,
            query=query,
            selected_document_ids=selected_document_ids,
            chunk_types=chunk_types,
            top_k=top_k,
        )

    # ============================================================
    # DOCUMENT SCOPE
    # ============================================================

    def get_search_scope(
        self,
        selected_document_ids: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return information describing the current retrieval scope.

        Useful for the UI, logging, Agentic RAG, and debugging.
        """

        document_ids = self._normalize_document_ids(
            selected_document_ids
        )

        if document_ids:
            return {
                "mode": "selected_documents",
                "document_ids": document_ids,
                "automatic": False,
            }

        return {
            "mode": "all_user_documents",
            "document_ids": [],
            "automatic": True,
        }

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_document_ids(
        document_ids: Optional[List[Any]],
    ) -> List[Any]:
        """
        Normalize document IDs while preserving their original
        values.

        Supports:
            None
            []
            tuple
            set
            list
            single document ID
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

            if isinstance(
                document_id,
                str,
            ):
                document_id = document_id.strip()

                if not document_id:
                    continue

            if document_id not in normalized:
                normalized.append(document_id)

        return normalized


# ================================================================
# CONVENIENCE FUNCTION
# ================================================================

def multi_document_search(
    vector_db,
    user_id: str,
    query: str,
    selected_document_ids: Optional[List[Any]] = None,
    chunk_types: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper around MultiDocumentRetriever.

    Example
    -------

        results = multi_document_search(
            vector_db=db,
            user_id=current_user_id,
            query="What is an encoder?",
            selected_document_ids=None,
            top_k=5,
        )

    If selected_document_ids is None:
        -> search across all user documents.

    If selected_document_ids contains IDs:
        -> search only those documents.
    """

    retriever = MultiDocumentRetriever(
        vector_db=vector_db
    )

    return retriever.retrieve(
        user_id=user_id,
        query=query,
        selected_document_ids=selected_document_ids,
        chunk_types=chunk_types,
        top_k=top_k,
    )