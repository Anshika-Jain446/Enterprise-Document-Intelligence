"""
Automatic Chunking Strategy Selector

This module automatically selects the most appropriate
existing chunking strategy based on the document's
type and extracted content.

IMPORTANT:
- Does NOT replace existing chunking methods.
- Does NOT modify ChunkingEngine.
- Does NOT perform chunking itself.
- Manual chunking remains fully supported.
"""

from typing import Any, Dict, Optional


class AutomaticChunkingSelector:
    """
    Automatically selects one of the existing ChunkingEngine
    chunking methods.
    """

    SUPPORTED_METHODS = {
        "Character",
        "Recursive",
        "Token",
        "Markdown",
        "Context",
        "Table",
        "Image",
        "Visual",
        "Multimodal",
    }

    def __init__(self):
        pass

    def analyze_document(
        self,
        file_name: Optional[str] = None,
        file_type: Optional[str] = None,
        text: Optional[str] = None,
        tables: Optional[Any] = None,
        images: Optional[Any] = None,
        visuals: Optional[Any] = None,
        pages: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the document and automatically select
        the most appropriate existing chunking method.

        Returns:
            Dictionary containing:
            - recommended_method
            - reason
            - automatic
            - characteristics
        """

        file_name = file_name or ""
        file_type = str(file_type or "").lower()

        extension = self._get_extension(file_name)

        text_length = len(
            str(text or "").strip()
        )

        table_count = self._get_count(tables)
        image_count = self._get_count(images)
        visual_count = self._get_count(visuals)
        page_count = self._get_count(pages)

        method, reason = self._select_method(
            extension=extension,
            file_type=file_type,
            text_length=text_length,
            table_count=table_count,
            image_count=image_count,
            visual_count=visual_count,
        )

        return {
            "recommended_method": method,
            "reason": reason,
            "automatic": True,
            "characteristics": {
                "extension": extension,
                "file_type": file_type,
                "text_length": text_length,
                "table_count": table_count,
                "image_count": image_count,
                "visual_count": visual_count,
                "page_count": page_count,
            },
        }

    def select_method(
        self,
        file_name: Optional[str] = None,
        file_type: Optional[str] = None,
        text: Optional[str] = None,
        tables: Optional[Any] = None,
        images: Optional[Any] = None,
        visuals: Optional[Any] = None,
        pages: Optional[Any] = None,
    ) -> str:
        """
        Return only the selected chunking method.
        """

        result = self.analyze_document(
            file_name=file_name,
            file_type=file_type,
            text=text,
            tables=tables,
            images=images,
            visuals=visuals,
            pages=pages,
        )

        return result["recommended_method"]

    def _select_method(
        self,
        extension: str,
        file_type: str,
        text_length: int,
        table_count: int,
        image_count: int,
        visual_count: int,
    ):
        """
        Select the most appropriate EXISTING
        ChunkingEngine method.
        """

        # =========================================================
        # MARKDOWN DOCUMENTS
        # =========================================================

        if extension in {".md", ".markdown"}:
            return (
                "Markdown",
                "Markdown document detected. "
                "Markdown chunking is recommended."
            )

        if "markdown" in file_type:
            return (
                "Markdown",
                "Markdown content detected. "
                "Markdown chunking is recommended."
            )

        # =========================================================
        # TABULAR DOCUMENTS
        # =========================================================

        if extension in {".xlsx", ".xls", ".csv"}:
            return (
                "Table",
                "Structured/tabular document detected. "
                "Table chunking is recommended."
            )

        # =========================================================
        # DOCUMENTS WITH TABLES + VISUAL CONTENT
        # =========================================================

        if table_count > 0 and (
            image_count > 0 or visual_count > 0
        ):
            return (
                "Multimodal",
                "The document contains tables and visual content. "
                "Multimodal chunking is recommended."
            )

        # =========================================================
        # DOCUMENTS WITH TABLES
        # =========================================================

        if table_count > 0:
            return (
                "Table",
                "Tables were detected in the document. "
                "Table chunking is recommended."
            )

        # =========================================================
        # DOCUMENTS WITH IMAGES + VISUALS
        # =========================================================

        if image_count > 0 and visual_count > 0:
            return (
                "Multimodal",
                "The document contains images and visual elements. "
                "Multimodal chunking is recommended."
            )

        # =========================================================
        # DOCUMENTS WITH IMAGES
        # =========================================================

        if image_count > 0:
            return (
                "Image",
                "Images were detected in the document. "
                "Image chunking is recommended."
            )

        # =========================================================
        # DOCUMENTS WITH VISUALS / DIAGRAMS
        # =========================================================

        if visual_count > 0:
            return (
                "Visual",
                "Visual or diagram content was detected. "
                "Visual chunking is recommended."
            )

        # =========================================================
        # LARGE TEXT DOCUMENTS
        # =========================================================

        if text_length > 50000:
            return (
                "Recursive",
                "The document contains a large amount of text. "
                "Recursive chunking is recommended."
            )

        # =========================================================
        # NORMAL TEXT DOCUMENTS
        # =========================================================

        if text_length > 0:
            return (
                "Recursive",
                "The document is primarily text-based. "
                "Recursive chunking is recommended."
            )

        # =========================================================
        # FALLBACK
        # =========================================================

        return (
            "Character",
            "The document content could not be reliably classified. "
            "Character chunking is used as the fallback."
        )

    @staticmethod
    def _get_extension(file_name: str) -> str:
        """
        Extract the lowercase file extension.
        """

        if not file_name:
            return ""

        file_name = str(file_name).lower()

        if "." not in file_name:
            return ""

        return "." + file_name.rsplit(".", 1)[-1]

    @staticmethod
    def _get_count(value: Any) -> int:
        """
        Safely return the number of items in a collection.
        """

        if value is None:
            return 0

        try:
            return len(value)
        except (TypeError, AttributeError):
            return 0


def get_automatic_chunking_method(
    file_name: Optional[str] = None,
    file_type: Optional[str] = None,
    text: Optional[str] = None,
    tables: Optional[Any] = None,
    images: Optional[Any] = None,
    visuals: Optional[Any] = None,
    pages: Optional[Any] = None,
) -> str:
    """
    Convenience function for the existing application.

    Returns:
        One of the existing ChunkingEngine methods.
    """

    selector = AutomaticChunkingSelector()

    return selector.select_method(
        file_name=file_name,
        file_type=file_type,
        text=text,
        tables=tables,
        images=images,
        visuals=visuals,
        pages=pages,
    )