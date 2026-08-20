import json
import re
from collections import defaultdict


class EnterpriseRAGAgent:
    """
    Enterprise Agentic RAG controller.

    Designed to work with:
        - persistent VectorDatabase
        - upgraded DocumentExtractor
        - upgraded ChunkingEngine
        - LLM planner/evaluator/answer generator
        - document text
        - tables
        - extracted images
        - rendered visual pages
        - OCR / image-analysis results when available
        - conversation history

    The agent does NOT require the currently uploaded files to match the
    persistent vector database.

    Retrieval strategy:
        1. Plan retrieval.
        2. Search persistent vector DB.
        3. Search supplied document text when useful.
        4. Search tables when useful.
        5. Search image/OCR/visual evidence when available.
        6. Use web search when appropriate.
        7. Evaluate evidence.
        8. Re-plan when evidence is insufficient.
        9. Generate the final answer from accumulated evidence.
    """

    def __init__(
        self,
        vector_db,
        llm,
        documents=None,
        selected_sources=None,
        conversation_history=None,
        top_k=5,
        max_iterations=4,
    ):
        self.vector_db = vector_db
        self.llm = llm

        self.documents = documents or []

        self.selected_sources = (
            self._normalize_sources(
                selected_sources
            )
        )

        self.conversation_history = (
            conversation_history or []
        )

        self.top_k = max(
            1,
            int(top_k),
        )

        self.max_iterations = max(
            1,
            int(max_iterations),
        )

        self.allowed_actions = {
            "vector_search",
            "document_search",
            "table_search",
            "image_search",
            "visual_search",
            "web_search",
        }

    # ============================================================
    # SOURCE HELPERS
    # ============================================================

    @staticmethod
    def _normalize_sources(sources):
        if not sources:
            return []

        if isinstance(sources, str):
            sources = [sources]

        normalized = set()

        for source in sources:
            if not source:
                continue

            for part in str(source).split(","):
                part = part.strip()

                if part:
                    normalized.add(part)

        return sorted(normalized)

    @staticmethod
    def _source_key(value):
        if not value:
            return ""

        value = str(value).strip()

        # Avoid requiring os.path here because sources may be URLs,
        # storage paths, or logical document IDs.
        value = value.replace("\\", "/")

        return value.rstrip("/").split("/")[-1].casefold()

    def _source_matches_selection(self, result):
        """
        Check whether a retrieved object belongs to one of the selected
        document sources.

        Empty selected_sources means all persistent documents are allowed.
        """

        if not self.selected_sources:
            return True

        if not isinstance(result, dict):
            return False

        possible_sources = []

        for key in (
            "source",
            "filename",
            "file_name",
            "document",
            "document_name",
            "document_id",
            "source_file",
        ):
            value = result.get(key)

            if value:
                possible_sources.append(
                    str(value)
                )

        metadata = result.get(
            "metadata",
            {},
        )

        if isinstance(metadata, dict):
            for key in (
                "source",
                "filename",
                "file_name",
                "uploaded_file_name",
                "document",
                "document_name",
                "document_id",
            ):
                value = metadata.get(key)

                if value:
                    possible_sources.append(
                        str(value)
                    )

        result_keys = {
            self._source_key(source)
            for source in possible_sources
            if source
        }

        selected_keys = {
            self._source_key(source)
            for source in self.selected_sources
        }

        return bool(
            result_keys & selected_keys
        )

    # ============================================================
    # GENERIC TEXT HELPERS
    # ============================================================

    @staticmethod
    def _query_terms(query):
        """
        Extract useful search terms.

        Removes extremely common stop words but deliberately keeps
        technical terms and numbers.
        """

        if not query:
            return set()

        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "how",
            "does",
            "did",
            "are",
            "was",
            "were",
            "is",
            "of",
            "to",
            "in",
            "on",
            "a",
            "an",
            "as",
            "at",
            "by",
            "be",
            "it",
            "or",
            "about",
            "tell",
            "me",
            "please",
        }

        terms = set()

        for word in re.findall(
            r"\b[\w.-]+\b",
            str(query).lower(),
        ):
            if (
                len(word) > 2
                and word not in stop_words
            ):
                terms.add(word)

        return terms

    @staticmethod
    def _safe_json(value):
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            return str(value)

    def _text_score(self, query, text):
        if not query or not text:
            return 0.0

        terms = self._query_terms(
            query
        )

        if not terms:
            return 0.0

        searchable = str(
            text
        ).lower()

        score = 0

        for term in terms:
            occurrences = searchable.count(
                term
            )

            if occurrences:
                score += min(
                    occurrences,
                    3,
                )

        return score / max(
            len(terms),
            1,
        )

    # ============================================================
    # STATE
    # ============================================================

    def _create_state(self, query):
        return {
            "query": query,
            "original_query": query,
            "action": None,
            "results": [],
            "all_results": [],
            "answer": "",
            "iterations": 0,
            "success": False,
            "trace": [],
            "previous_actions": [],
            "previous_evaluations": [],
            "web_used": False,
            "document_used": False,
            "table_used": False,
            "image_used": False,
            "visual_used": False,
        }

    # ============================================================
    # TRACE
    # ============================================================

    def _trace(
        self,
        state,
        step,
        action,
        details=None,
    ):
        state["trace"].append(
            {
                "step": step,
                "action": action,
                "details": details or {},
            }
        )

    # ============================================================
    # VECTOR SEARCH
    # ============================================================

    def _vector_search(self, query):
        if not query or not query.strip():
            return []

        if self.vector_db is None:
            return []

        try:
            if (
                getattr(
                    self.vector_db,
                    "index",
                    None,
                )
                is None
            ):
                if self.vector_db.exists():
                    self.vector_db.load()
                else:
                    return []

            if (
                getattr(
                    self.vector_db,
                    "index",
                    None,
                )
                is None
            ):
                return []

            try:
                results = (
                    self.vector_db.search(
                        query,
                        top_k=self.top_k,
                        sources=(
                            self.selected_sources
                            or None
                        ),
                    )
                )

            except TypeError:
                results = (
                    self.vector_db.search(
                        query,
                        top_k=self.top_k,
                    )
                )

            normalized = []

            for result in results or []:
                if not isinstance(
                    result,
                    dict,
                ):
                    continue

                if not self._source_matches_selection(
                    result
                ):
                    continue

                item = dict(result)

                item.setdefault(
                    "source_type",
                    "document",
                )

                item.setdefault(
                    "search_type",
                    "vector_search",
                )

                normalized.append(
                    item
                )

            return normalized

        except Exception as error:
            print(
                f"Vector search failed: "
                f"{error}"
            )

            return []

    # ============================================================
    # DOCUMENT TEXT SEARCH
    # ============================================================

    def _document_search(self, query):
        if (
            not query
            or not self.documents
        ):
            return []

        results = []

        for document in self.documents:
            if not isinstance(
                document,
                dict,
            ):
                continue

            if not self._source_matches_selection(
                document
            ):
                continue

            metadata = document.get(
                "metadata",
                {},
            )

            text = str(
                document.get(
                    "text",
                    "",
                )
            )

            searchable = (
                text
                + "\n"
                + self._safe_json(
                    metadata
                )
            )

            score = self._text_score(
                query,
                searchable,
            )

            if score <= 0:
                continue

            result = dict(
                document
            )

            result[
                "similarity_score"
            ] = score

            result[
                "source_type"
            ] = "document"

            result[
                "search_type"
            ] = "document_search"

            results.append(
                result
            )

        results.sort(
            key=lambda item:
                float(
                    item.get(
                        "similarity_score",
                        0,
                    )
                ),
            reverse=True,
        )

        return results[
            : self.top_k
        ]

    # ============================================================
    # TABLE SEARCH
    # ============================================================

    def _table_search(self, query):
        if (
            not query
            or not self.documents
        ):
            return []

        results = []

        for document in self.documents:
            if not isinstance(
                document,
                dict,
            ):
                continue

            if not self._source_matches_selection(
                document
            ):
                continue

            tables = document.get(
                "tables",
                [],
            )

            if not isinstance(
                tables,
                list,
            ):
                continue

            metadata = document.get(
                "metadata",
                {},
            )

            document_source = (
                document.get(
                    "source"
                )
                or (
                    metadata.get(
                        "uploaded_file_name"
                    )
                    if isinstance(
                        metadata,
                        dict,
                    )
                    else None
                )
            )

            for table in tables:
                if not isinstance(
                    table,
                    dict,
                ):
                    continue

                table_data = table.get(
                    "table",
                    [],
                )

                searchable = (
                    self._safe_json(
                        table_data
                    )
                    + "\n"
                    + self._safe_json(
                        table
                    )
                )

                score = self._text_score(
                    query,
                    searchable,
                )

                if score <= 0:
                    continue

                result = dict(
                    table
                )

                if document_source:
                    result.setdefault(
                        "source",
                        document_source,
                    )

                result[
                    "similarity_score"
                ] = score

                result[
                    "source_type"
                ] = "table"

                result[
                    "search_type"
                ] = "table_search"

                results.append(
                    result
                )

        results.sort(
            key=lambda item:
                float(
                    item.get(
                        "similarity_score",
                        0,
                    )
                ),
            reverse=True,
        )

        return results[
            : self.top_k
        ]

    # ============================================================
    # IMAGE SEARCH
    # ============================================================

    def _image_search(self, query):
        """
        Search OCR/image-analysis information already attached to
        extracted document records.

        This does not attempt to perform OCR itself. OCR/image analysis
        belongs in extractor.py or a dedicated vision model. This layer
        consumes that evidence.
        """

        if (
            not query
            or not self.documents
        ):
            return []

        results = []

        image_keys = (
            "images",
            "image_results",
            "ocr",
            "ocr_results",
            "image_analysis",
            "image_descriptions",
            "visual_analysis",
        )

        for document in self.documents:
            if not isinstance(
                document,
                dict,
            ):
                continue

            if not self._source_matches_selection(
                document
            ):
                continue

            metadata = document.get(
                "metadata",
                {},
            )

            source = (
                document.get(
                    "source"
                )
                or (
                    metadata.get(
                        "file_name"
                    )
                    if isinstance(
                        metadata,
                        dict,
                    )
                    else None
                )
            )

            for key in image_keys:
                entries = document.get(
                    key,
                    [],
                )

                if not isinstance(
                    entries,
                    list,
                ):
                    if entries:
                        entries = [
                            entries
                        ]
                    else:
                        continue

                for entry in entries:
                    if not isinstance(
                        entry,
                        dict,
                    ):
                        continue

                    searchable_parts = [
                        entry.get(
                            "ocr_text",
                            "",
                        ),
                        entry.get(
                            "text",
                            "",
                        ),
                        entry.get(
                            "description",
                            "",
                        ),
                        entry.get(
                            "caption",
                            "",
                        ),
                        entry.get(
                            "analysis",
                            "",
                        ),
                        entry.get(
                            "content",
                            "",
                        ),
                        self._safe_json(
                            entry
                        ),
                    ]

                    searchable = "\n".join(
                        str(part)
                        for part in searchable_parts
                        if part
                    )

                    score = self._text_score(
                        query,
                        searchable,
                    )

                    if score <= 0:
                        continue

                    result = dict(
                        entry
                    )

                    if source:
                        result.setdefault(
                            "source",
                            source,
                        )

                    result[
                        "similarity_score"
                    ] = score

                    result[
                        "source_type"
                    ] = "image"

                    result[
                        "search_type"
                    ] = "image_search"

                    results.append(
                        result
                    )

        results.sort(
            key=lambda item:
                float(
                    item.get(
                        "similarity_score",
                        0,
                    )
                ),
            reverse=True,
        )

        return results[
            : self.top_k
        ]

    # ============================================================
    # VISUAL SEARCH
    # ============================================================

    def _visual_search(self, query):
        """
        Search rendered visual/page analysis.

        Supports documents where extractor.py stores rendered pages,
        diagrams, charts, flowcharts, figures, or visual-analysis text.
        """

        if (
            not query
            or not self.documents
        ):
            return []

        results = []

        for document in self.documents:
            if not isinstance(
                document,
                dict,
            ):
                continue

            if not self._source_matches_selection(
                document
            ):
                continue

            visuals = document.get(
                "visuals",
                [],
            )

            if not isinstance(
                visuals,
                list,
            ):
                continue

            metadata = document.get(
                "metadata",
                {},
            )

            source = (
                document.get(
                    "source"
                )
                or (
                    metadata.get(
                        "file_name"
                    )
                    if isinstance(
                        metadata,
                        dict,
                    )
                    else None
                )
            )

            for visual in visuals:
                if not isinstance(
                    visual,
                    dict,
                ):
                    continue

                searchable_parts = [
                    visual.get(
                        "description",
                        "",
                    ),
                    visual.get(
                        "analysis",
                        "",
                    ),
                    visual.get(
                        "ocr_text",
                        "",
                    ),
                    visual.get(
                        "text",
                        "",
                    ),
                    visual.get(
                        "caption",
                        "",
                    ),
                    self._safe_json(
                        visual
                    ),
                ]

                searchable = "\n".join(
                    str(part)
                    for part in searchable_parts
                    if part
                )

                score = self._text_score(
                    query,
                    searchable,
                )

                if score <= 0:
                    continue

                result = dict(
                    visual
                )

                if source:
                    result.setdefault(
                        "source",
                        source,
                    )

                result[
                    "similarity_score"
                ] = score

                result[
                    "source_type"
                ] = "visual"

                result[
                    "search_type"
                ] = "visual_search"

                results.append(
                    result
                )

        results.sort(
            key=lambda item:
                float(
                    item.get(
                        "similarity_score",
                        0,
                    )
                ),
            reverse=True,
        )

        return results[
            : self.top_k
        ]

    # ============================================================
    # WEB SEARCH
    # ============================================================

    def _web_search(self, query):
        if not query or not query.strip():
            return []

        if self.llm is None:
            return []

        try:
            if not hasattr(
                self.llm,
                "web_search",
            ):
                return []

            results = self.llm.web_search(
                query=query,
                max_results=self.top_k,
            )

            return self._normalize_results(
                results,
                "web_search",
            )

        except Exception as error:
            print(
                f"Web search failed: "
                f"{error}"
            )

            return []

    # ============================================================
    # TOOL EXECUTION
    # ============================================================

    def execute_tool(
        self,
        action,
        query,
    ):
        print(
            f"Executing tool: {action}"
        )

        if action == "vector_search":
            return self._vector_search(
                query
            )

        if action == "document_search":
            return self._document_search(
                query
            )

        if action == "table_search":
            return self._table_search(
                query
            )

        if action == "image_search":
            return self._image_search(
                query
            )

        if action == "visual_search":
            return self._visual_search(
                query
            )

        if action == "web_search":
            return self._web_search(
                query
            )

        return []

    # ============================================================
    # RESULT NORMALIZATION
    # ============================================================

    def _normalize_results(
        self,
        results,
        action,
    ):
        if not isinstance(
            results,
            list,
        ):
            return []

        normalized = []

        for result in results:
            if not isinstance(
                result,
                dict,
            ):
                continue

            item = dict(
                result
            )

            item.setdefault(
                "search_type",
                action,
            )

            if action == "web_search":
                item.setdefault(
                    "source_type",
                    "web",
                )

            elif action == "table_search":
                item.setdefault(
                    "source_type",
                    "table",
                )

            elif action == "image_search":
                item.setdefault(
                    "source_type",
                    "image",
                )

            elif action == "visual_search":
                item.setdefault(
                    "source_type",
                    "visual",
                )

            else:
                item.setdefault(
                    "source_type",
                    "document",
                )

            normalized.append(
                item
            )

        return normalized

    # ============================================================
    # RESULT KEY
    # ============================================================

    def _result_key(
        self,
        result,
    ):
        if not isinstance(
            result,
            dict,
        ):
            return None

        source = (
            result.get(
                "source"
            )
            or result.get(
                "filename"
            )
            or result.get(
                "file_name"
            )
            or result.get(
                "document"
            )
            or ""
        )

        chunk_id = (
            result.get(
                "chunk_id"
            )
            or result.get(
                "id"
            )
            or result.get(
                "page"
            )
            or ""
        )

        content = str(
            result.get(
                "content"
            )
            or result.get(
                "text"
            )
            or result.get(
                "description"
            )
            or result.get(
                "ocr_text"
            )
            or ""
        )

        return (
            str(source),
            str(chunk_id),
            content[:500],
        )

    def _deduplicate_results(
        self,
        results,
    ):
        unique = []
        seen = set()

        for result in results:
            key = self._result_key(
                result
            )

            if key is None:
                unique.append(
                    result
                )
                continue

            if key in seen:
                continue

            seen.add(key)
            unique.append(
                result
            )

        return unique

    # ============================================================
    # EVIDENCE PREPARATION
    # ============================================================

    def _prepare_evidence(
        self,
        results,
    ):
        """
        Create a compact evidence package for the LLM.

        Keeps the original result objects available while avoiding
        unnecessarily huge serialized payloads.
        """

        evidence = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            if not isinstance(
                result,
                dict,
            ):
                continue

            content = (
                result.get(
                    "content"
                )
                or result.get(
                    "text"
                )
                or result.get(
                    "description"
                )
                or result.get(
                    "ocr_text"
                )
                or result.get(
                    "analysis"
                )
                or ""
            )

            evidence.append(
                {
                    "evidence_id": index,
                    "source": result.get(
                        "source",
                        result.get(
                            "filename",
                            result.get(
                                "file_name",
                                "",
                            ),
                        ),
                    ),
                    "source_type": result.get(
                        "source_type",
                        "",
                    ),
                    "search_type": result.get(
                        "search_type",
                        "",
                    ),
                    "page": result.get(
                        "page"
                    ),
                    "chunk_id": result.get(
                        "chunk_id"
                    ),
                    "similarity_score": result.get(
                        "similarity_score"
                    ),
                    "content": str(
                        content
                    )[:12000],
                }
            )

        return evidence

    # ============================================================
    # CONVERSATION CONTEXT
    # ============================================================

    def _conversation_context(self):
        if not self.conversation_history:
            return ""

        recent = (
            self.conversation_history[-8:]
        )

        lines = []

        for message in recent:
            if not isinstance(
                message,
                dict,
            ):
                continue

            role = str(
                message.get(
                    "role",
                    "",
                )
            ).upper()

            content = message.get(
                "content",
                "",
            )

            if content:
                lines.append(
                    f"{role}: {content}"
                )

        return "\n".join(
            lines
        )

    # ============================================================
    # DEFAULT PLANNER
    # ============================================================

    def _fallback_plan(
        self,
        query,
        previous_actions,
    ):
        """
        Conservative planner fallback.

        If the LLM planner is unavailable, start with persistent
        document retrieval. If that fails, progressively inspect
        tables, images/visuals, then web.
        """

        if not previous_actions:
            return {
                "action": "vector_search",
                "query": query,
                "reason": (
                    "Default persistent "
                    "document retrieval."
                ),
            }

        if (
            "vector_search"
            not in previous_actions
        ):
            return {
                "action": "vector_search",
                "query": query,
                "reason": (
                    "Retry persistent "
                    "document retrieval."
                ),
            }

        if (
            "table_search"
            not in previous_actions
        ):
            return {
                "action": "table_search",
                "query": query,
                "reason": (
                    "Inspect structured "
                    "table evidence."
                ),
            }

        if (
            "image_search"
            not in previous_actions
        ):
            return {
                "action": "image_search",
                "query": query,
                "reason": (
                    "Inspect OCR/image evidence."
                ),
            }

        if (
            "visual_search"
            not in previous_actions
        ):
            return {
                "action": "visual_search",
                "query": query,
                "reason": (
                    "Inspect diagrams and "
                    "rendered visual evidence."
                ),
            }

        return {
            "action": "web_search",
            "query": query,
            "reason": (
                "Use external evidence "
                "after local retrieval."
            ),
        }

    # ============================================================
    # PLANNING
    # ============================================================

    def _plan(
        self,
        query,
        state,
    ):
        conversation_context = (
            self._conversation_context()
        )

        if self.llm is None:
            return self._fallback_plan(
                query,
                state[
                    "previous_actions"
                ],
            )

        try:
            planning_query = query

            if conversation_context:
                planning_query = (
                    "Conversation context:\n"
                    + conversation_context
                    + "\n\nCurrent question:\n"
                    + query
                )

            plan = self.llm.plan_action(
                query=planning_query,
                previous_actions=state[
                    "previous_actions"
                ],
                previous_evaluations=state[
                    "previous_evaluations"
                ],
            )

            if not isinstance(
                plan,
                dict,
            ):
                return self._fallback_plan(
                    query,
                    state[
                        "previous_actions"
                    ],
                )

            action = plan.get(
                "action",
                "vector_search",
            )

            if action not in self.allowed_actions:
                return self._fallback_plan(
                    query,
                    state[
                        "previous_actions"
                    ],
                )

            search_query = plan.get(
                "query",
                query,
            )

            if not isinstance(
                search_query,
                str,
            ) or not search_query.strip():
                search_query = query

            return {
                **plan,
                "action": action,
                "query": search_query,
            }

        except Exception as error:
            print(
                f"Planner failed: "
                f"{error}"
            )

            return self._fallback_plan(
                query,
                state[
                    "previous_actions"
                ],
            )

    # ============================================================
    # EVIDENCE EVALUATION
    # ============================================================

    def _evaluate(
        self,
        query,
        results,
        action,
    ):
        if self.llm is None:
            return {
                "sufficient": bool(
                    results
                ),
                "confidence": (
                    0.7
                    if results
                    else 0.0
                ),
                "reason": (
                    "Fallback evaluation."
                ),
                "recommended_action": (
                    "vector_search"
                ),
            }

        try:
            evaluation = (
                self.llm.evaluate_evidence(
                    query=query,
                    results=results,
                    action=action,
                )
            )

            if isinstance(
                evaluation,
                dict,
            ):
                return evaluation

        except Exception as error:
            print(
                f"Evaluator failed: "
                f"{error}"
            )

        next_action = (
            "web_search"
            if action != "web_search"
            else "vector_search"
        )

        return {
            "sufficient": False,
            "confidence": 0.0,
            "reason": (
                "Evidence evaluator failed."
            ),
            "recommended_action": next_action,
        }

    # ============================================================
    # ACTION TRACKING
    # ============================================================

    def _update_usage(
        self,
        state,
        action,
    ):
        if action == "web_search":
            state[
                "web_used"
            ] = True

        elif action in (
            "vector_search",
            "document_search",
        ):
            state[
                "document_used"
            ] = True

        elif action == "table_search":
            state[
                "table_used"
            ] = True

        elif action == "image_search":
            state[
                "image_used"
            ] = True

        elif action == "visual_search":
            state[
                "visual_used"
            ] = True

    # ============================================================
    # SOURCE SUMMARY
    # ============================================================

    def _source_summary(
        self,
        results,
    ):
        sources = []

        for result in results:
            if not isinstance(
                result,
                dict,
            ):
                continue

            source = (
                result.get(
                    "source"
                )
                or result.get(
                    "filename"
                )
                or result.get(
                    "file_name"
                )
            )

            if source:
                sources.append(
                    str(source)
                )

        return self._normalize_sources(
            sources
        )

    # ============================================================
    # FINAL ANSWER
    # ============================================================

    def _generate_answer(
        self,
        query,
        results,
        source_type,
    ):
        if self.llm is None:
            return (
                "Evidence was retrieved, "
                "but no LLM is configured "
                "to generate the answer."
            )

        try:
            return self.llm.generate_answer(
                query=query,
                chunks=results,
                source_type=source_type,
            )

        except Exception as error:
            print(
                f"Answer generation failed: "
                f"{error}"
            )

            # Compatibility fallback if the LLM expects a more
            # generic evidence representation.
            try:
                evidence = (
                    self._prepare_evidence(
                        results
                    )
                )

                return self.llm.generate_answer(
                    query=query,
                    chunks=evidence,
                    source_type=source_type,
                )

            except Exception:
                return (
                    "Relevant evidence was "
                    "retrieved, but the final "
                    "answer could not be generated."
                )

    # ============================================================
    # SOURCE TYPE
    # ============================================================

    def _determine_source_type(
        self,
        results,
    ):
        types = {
            str(
                result.get(
                    "source_type",
                    "",
                )
            ).lower()
            for result in results
            if isinstance(
                result,
                dict,
            )
        }

        if "web" in types:
            return "mixed_web"

        if "visual" in types:
            return "mixed_visual"

        if "image" in types:
            return "mixed_image"

        if "table" in types:
            return "mixed_table"

        return "document"

    # ============================================================
    # MAIN AGENT LOOP
    # ============================================================

    def run(
        self,
        query,
    ):
        if not query or not query.strip():
            return {
                "answer": (
                    "Please enter a question."
                ),
                "sources": [],
                "action": None,
                "success": False,
                "iterations": 0,
                "trace": [],
            }

        state = self._create_state(
            query
        )

        working_query = query
        forced_action = None

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ENTERPRISE AGENTIC RAG"
        )

        print(
            "=" * 70
        )

        # --------------------------------------------------------
        # ITERATIVE RETRIEVAL
        # --------------------------------------------------------

        while (
            state["iterations"]
            < self.max_iterations
        ):
            state["iterations"] += 1

            iteration = (
                state["iterations"]
            )

            # ----------------------------------------------------
            # PLAN
            # ----------------------------------------------------

            if forced_action:
                plan = {
                    "action": forced_action,
                    "query": working_query,
                    "reason": (
                        "Replanned after "
                        "insufficient evidence."
                    ),
                }

                forced_action = None

            else:
                plan = self._plan(
                    working_query,
                    state,
                )

            action = plan.get(
                "action",
                "vector_search",
            )

            search_query = plan.get(
                "query",
                working_query,
            )

            if (
                action
                not in self.allowed_actions
            ):
                action = "vector_search"

            if not isinstance(
                search_query,
                str,
            ) or not search_query.strip():
                search_query = working_query

            state["action"] = action

            state[
                "previous_actions"
            ].append(action)

            self._update_usage(
                state,
                action,
            )

            # ----------------------------------------------------
            # TRACE PLAN
            # ----------------------------------------------------

            self._trace(
                state,
                len(
                    state["trace"]
                ) + 1,
                "agent_plan",
                {
                    "iteration": iteration,
                    "tool": action,
                    "query": search_query,
                    "reason": plan.get(
                        "reason",
                        "",
                    ),
                    "selected_sources": (
                        self.selected_sources
                    ),
                },
            )

            print(
                f"\nIteration {iteration}"
            )

            print(
                f"Agent selected: "
                f"{action}"
            )

            # ----------------------------------------------------
            # EXECUTE
            # ----------------------------------------------------

            try:
                results = (
                    self.execute_tool(
                        action,
                        search_query,
                    )
                )

            except Exception as error:
                print(
                    f"Tool execution failed: "
                    f"{error}"
                )

                results = []

            results = (
                self._normalize_results(
                    results,
                    action,
                )
            )

            state["results"] = results

            state[
                "all_results"
            ].extend(results)

            state[
                "all_results"
            ] = (
                self._deduplicate_results(
                    state[
                        "all_results"
                    ]
                )
            )

            self._trace(
                state,
                len(
                    state["trace"]
                ) + 1,
                "tool_result",
                {
                    "tool": action,
                    "results": len(
                        results
                    ),
                    "total_unique_results": len(
                        state[
                            "all_results"
                        ]
                    ),
                },
            )

            print(
                f"Retrieved: "
                f"{len(results)} results"
            )

            # ----------------------------------------------------
            # EVALUATE
            # ----------------------------------------------------

            evaluation = self._evaluate(
                state[
                    "original_query"
                ],
                results,
                action,
            )

            state[
                "previous_evaluations"
            ].append(
                evaluation
            )

            sufficient = bool(
                evaluation.get(
                    "sufficient",
                    False,
                )
            )

            try:
                confidence = float(
                    evaluation.get(
                        "confidence",
                        0.0,
                    )
                )
            except Exception:
                confidence = 0.0

            self._trace(
                state,
                len(
                    state["trace"]
                ) + 1,
                "evidence_evaluation",
                evaluation,
            )

            print(
                "Evidence sufficient: "
                f"{sufficient}"
            )

            print(
                "Confidence: "
                f"{confidence:.2f}"
            )

            # ----------------------------------------------------
            # SUCCESS
            # ----------------------------------------------------

            if (
                sufficient
                and results
            ):
                break

            # ----------------------------------------------------
            # REPLAN
            # ----------------------------------------------------

            recommended = evaluation.get(
                "recommended_action"
            )

            if (
                recommended
                not in self.allowed_actions
            ):
                fallback = (
                    self._fallback_plan(
                        search_query,
                        state[
                            "previous_actions"
                        ],
                    )
                )

                recommended = fallback[
                    "action"
                ]

            # Avoid endlessly repeating
            # exactly the same failed action.
            if (
                recommended == action
                and iteration
                < self.max_iterations
            ):
                fallback = (
                    self._fallback_plan(
                        search_query,
                        state[
                            "previous_actions"
                        ],
                    )
                )

                recommended = fallback[
                    "action"
                ]

            working_query = (
                search_query
            )

            if (
                iteration
                < self.max_iterations
            ):
                forced_action = (
                    recommended
                )

                self._trace(
                    state,
                    len(
                        state["trace"]
                    ) + 1,
                    "replan",
                    {
                        "next_action":
                            recommended
                    },
                )

                print(
                    "Replanning → "
                    f"{recommended}"
                )

        # ========================================================
        # FINAL EVIDENCE
        # ========================================================

        final_results = (
            self._deduplicate_results(
                state[
                    "all_results"
                ]
            )
        )

        # Sort stronger evidence first where a score exists.
        final_results.sort(
            key=lambda result:
                float(
                    result.get(
                        "similarity_score",
                        0.0,
                    )
                    or 0.0
                ),
            reverse=True,
        )

        # Limit the final context without destroying source diversity.
        final_results = (
            self._select_final_evidence(
                final_results
            )
        )

        if not final_results:
            self._trace(
                state,
                len(
                    state["trace"]
                ) + 1,
                "no_evidence",
                {
                    "selected_sources": (
                        self.selected_sources
                    )
                },
            )

            return {
                "answer": (
                    "I could not find sufficient "
                    "evidence to answer that question."
                ),
                "sources": [],
                "action": state["action"],
                "success": False,
                "iterations": state[
                    "iterations"
                ],
                "trace": state["trace"],
            }

        # ========================================================
        # SOURCE TYPE
        # ========================================================

        source_type = (
            self._determine_source_type(
                final_results
            )
        )

        # ========================================================
        # FINAL ANSWER
        # ========================================================

        answer = self._generate_answer(
            state[
                "original_query"
            ],
            final_results,
            source_type,
        )

        state["answer"] = answer

        state["success"] = bool(
            answer
            and str(answer).strip()
        )

        # ========================================================
        # FINAL TRACE
        # ========================================================

        self._trace(
            state,
            len(
                state["trace"]
            ) + 1,
            "answer_generated",
            {
                "source_type":
                    source_type,
                "evidence_count":
                    len(
                        final_results
                    ),
                "sources":
                    self._source_summary(
                        final_results
                    ),
                "selected_sources":
                    self.selected_sources,
                "image_evidence_used":
                    state[
                        "image_used"
                    ],
                "visual_evidence_used":
                    state[
                        "visual_used"
                    ],
                "web_used":
                    state[
                        "web_used"
                    ],
            },
        )

        # ========================================================
        # RETURN
        # ========================================================

        return {
            "answer": state[
                "answer"
            ],
            "sources": final_results,
            "action": state[
                "action"
            ],
            "success": state[
                "success"
            ],
            "iterations": state[
                "iterations"
            ],
            "trace": state[
                "trace"
            ],
            "metadata": {
                "source_type":
                    source_type,
                "selected_sources":
                    self.selected_sources,
                "document_evidence_used":
                    state[
                        "document_used"
                    ],
                "table_evidence_used":
                    state[
                        "table_used"
                    ],
                "image_evidence_used":
                    state[
                        "image_used"
                    ],
                "visual_evidence_used":
                    state[
                        "visual_used"
                    ],
                "web_evidence_used":
                    state[
                        "web_used"
                    ],
            },
        }

    # ============================================================
    # FINAL EVIDENCE SELECTION
    # ============================================================

    def _select_final_evidence(
        self,
        results,
    ):
        """
        Preserve evidence diversity.

        We do not want the final LLM context to contain ten nearly
        identical text chunks while completely omitting a relevant
        table or image.
        """

        if not results:
            return []

        selected = []

        type_limits = defaultdict(
            int
        )

        limits = {
            "document": max(
                2,
                self.top_k,
            ),
            "table": max(
                2,
                self.top_k,
            ),
            "image": max(
                2,
                self.top_k,
            ),
            "visual": max(
                2,
                self.top_k,
            ),
            "web": max(
                2,
                self.top_k,
            ),
        }

        # First pass: strongest result from each source type.
        for result in results:
            source_type = str(
                result.get(
                    "source_type",
                    "document",
                )
            ).lower()

            if (
                source_type
                not in limits
            ):
                source_type = (
                    "document"
                )

            if (
                type_limits[
                    source_type
                ]
                >= 1
            ):
                continue

            selected.append(
                result
            )

            type_limits[
                source_type
            ] += 1

        # Second pass: fill remaining slots.
        max_final = max(
            self.top_k * 2,
            8,
        )

        for result in results:
            if result in selected:
                continue

            source_type = str(
                result.get(
                    "source_type",
                    "document",
                )
            ).lower()

            if (
                source_type
                not in limits
            ):
                source_type = (
                    "document"
                )

            if (
                type_limits[
                    source_type
                ]
                >= limits[
                    source_type
                ]
            ):
                continue

            selected.append(
                result
            )

            type_limits[
                source_type
            ] += 1

            if (
                len(selected)
                >= max_final
            ):
                break

        return selected

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

        for chunk in results:
            print(
                f"\nChunk/Page : "
                f"{chunk.get('chunk_id', chunk.get('page', 'N/A'))}"
            )

            print(
                f"Source     : "
                f"{chunk.get('source', 'N/A')}"
            )

            print(
                f"Type       : "
                f"{chunk.get('source_type', 'N/A')}"
            )

            print(
                f"Search     : "
                f"{chunk.get('search_type', 'N/A')}"
            )

            if (
                "similarity_score"
                in chunk
            ):
                try:
                    print(
                        f"Score      : "
                        f"{float(chunk['similarity_score']):.4f}"
                    )
                except Exception:
                    pass

            if (
                "distance"
                in chunk
            ):
                try:
                    print(
                        f"Distance   : "
                        f"{float(chunk['distance']):.4f}"
                    )
                except Exception:
                    pass

            content = (
                chunk.get(
                    "content"
                )
                or chunk.get(
                    "text"
                )
                or chunk.get(
                    "description"
                )
                or chunk.get(
                    "ocr_text"
                )
                or ""
            )

            print(
                str(
                    content
                )[:1000]
            )

            print(
                "-" * 70
            )

