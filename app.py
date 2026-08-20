import json


class EnterpriseRAGAgent:
    """
    Enterprise Agentic RAG controller.

    Compatible with the current model.py GeminiModel implementation.

    Retrieval tools:
        1. vector_search
        2. document_search
        3. table_search
        4. web_search (optional)

    The agent:
        - plans retrieval
        - executes retrieval
        - evaluates evidence
        - replans when evidence is insufficient
        - generates a grounded final answer
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

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

        self.selected_sources = self._normalize_sources(
            selected_sources
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
            "web_search",
        }

    # ============================================================
    # SOURCE NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_sources(sources):
        if not sources:
            return []

        if isinstance(sources, str):
            sources = [sources]

        normalized = set()

        for source in sources:
            if source is None:
                continue

            value = str(source).strip()

            if not value:
                continue

            for part in value.split(","):
                part = part.strip()

                if part:
                    normalized.add(part)

        return sorted(normalized)

    # ============================================================
    # BASENAME
    # ============================================================

    @staticmethod
    def _basename(value):
        if not value:
            return ""

        value = str(value).replace("\\", "/")

        return value.rstrip("/").split("/")[-1]

    # ============================================================
    # SOURCE MATCHING
    # ============================================================

    def _source_matches_selection(self, result):
        """
        If no source is selected, allow everything.

        Otherwise compare source / filename fields and metadata.
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
        ):
            value = result.get(key)

            if value:
                possible_sources.append(str(value))

        metadata = result.get("metadata", {})

        if isinstance(metadata, dict):
            for key in (
                "uploaded_file_name",
                "file_name",
                "filename",
                "source",
                "document",
                "document_name",
            ):
                value = metadata.get(key)

                if value:
                    possible_sources.append(str(value))

        if not possible_sources:
            return False

        result_sources = set(
            self._normalize_sources(
                possible_sources
            )
        )

        selected_sources = set(
            self.selected_sources
        )

        if selected_sources & result_sources:
            return True

        selected_basenames = {
            self._basename(source)
            for source in selected_sources
        }

        result_basenames = {
            self._basename(source)
            for source in result_sources
        }

        return bool(
            selected_basenames & result_basenames
        )

    # ============================================================
    # GENERIC MODEL CALL
    # ============================================================

    def _model_generate(self, prompt):
        """
        Works with the current model.py GeminiModel.

        Required interface:
            llm.generate(prompt)

        Also supports an older LLM implementation if it exposes
        generate().
        """

        if not self.llm:
            raise RuntimeError(
                "LLM/model is not configured."
            )

        generate_method = getattr(
            self.llm,
            "generate",
            None,
        )

        if not callable(generate_method):
            raise RuntimeError(
                "LLM object must provide generate(prompt)."
            )

        return str(
            generate_method(prompt)
        ).strip()

    # ============================================================
    # JSON CLEANING
    # ============================================================

    @staticmethod
    def _clean_json(raw):
        if not raw:
            return ""

        raw = str(raw).strip()

        if raw.startswith("```json"):
            raw = raw[7:]

        elif raw.startswith("```"):
            raw = raw[3:]

        if raw.endswith("```"):
            raw = raw[:-3]

        return raw.strip()

    # ============================================================
    # CONVERSATION CONTEXT
    # ============================================================

    def _conversation_context(self):
        if not self.conversation_history:
            return ""

        recent = self.conversation_history[-8:]

        lines = []

        for message in recent:
            if not isinstance(message, dict):
                continue

            role = str(
                message.get(
                    "role",
                    "",
                )
            ).upper()

            content = str(
                message.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:
                continue

            lines.append(
                f"{role}: {content}"
            )

        return "\n".join(lines)

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
        }

    # ============================================================
    # TRACE
    # ============================================================

    def _trace(
        self,
        state,
        action,
        details=None,
    ):
        state["trace"].append(
            {
                "step": len(
                    state["trace"]
                ) + 1,
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
            index = getattr(
                self.vector_db,
                "index",
                None,
            )

            if index is None:

                try:
                    exists = bool(
                        self.vector_db.exists()
                    )
                except Exception:
                    exists = False

                if not exists:
                    return []

                try:
                    self.vector_db.load()
                except Exception as exc:
                    print(
                        f"Vector database load failed: {exc}"
                    )
                    return []

            if getattr(
                self.vector_db,
                "index",
                None,
            ) is None:
                return []

            # Current VectorDatabase supports search(query, top_k, sources)
            try:
                results = self.vector_db.search(
                    query,
                    top_k=self.top_k,
                    sources=(
                        self.selected_sources
                        or None
                    ),
                )

            except TypeError:
                results = self.vector_db.search(
                    query,
                    top_k=self.top_k,
                )

            normalized = []

            for result in results or []:

                if not isinstance(result, dict):
                    continue

                if not self._source_matches_selection(
                    result
                ):
                    continue

                item = dict(result)

                item["source_type"] = "document"
                item["search_type"] = "vector_search"

                if not item.get("content"):
                    item["content"] = (
                        item.get("text")
                        or item.get("page_content")
                        or ""
                    )

                normalized.append(item)

            return normalized[: self.top_k]

        except Exception as exc:
            print(
                f"Vector search failed: {exc}"
            )
            return []

    # ============================================================
    # DOCUMENT SEARCH
    # ============================================================

    def _document_search(self, query):
        if not query or not self.documents:
            return []

        words = {
            word.lower().strip(
                ".,!?;:()[]{}\"'"
            )
            for word in query.split()
            if len(
                word.strip(
                    ".,!?;:()[]{}\"'"
                )
            ) > 2
        }

        if not words:
            return []

        results = []

        for document in self.documents:

            if not isinstance(document, dict):
                continue

            if not self._source_matches_selection(
                document
            ):
                continue

            metadata = document.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            text = str(
                document.get(
                    "text",
                    document.get(
                        "content",
                        "",
                    ),
                )
            )

            metadata_text = json.dumps(
                metadata,
                ensure_ascii=False,
                default=str,
            )

            source_text = str(
                document.get(
                    "source",
                    "",
                )
            )

            filename_text = str(
                document.get(
                    "filename",
                    document.get(
                        "file_name",
                        "",
                    ),
                )
            )

            searchable = (
                text
                + " "
                + metadata_text
                + " "
                + source_text
                + " "
                + filename_text
            ).lower()

            matched_words = [
                word
                for word in words
                if word in searchable
            ]

            score = len(matched_words)

            if score <= 0:
                continue

            result = dict(document)

            result["similarity_score"] = (
                score / max(
                    len(words),
                    1,
                )
            )

            result["source_type"] = "document"
            result["search_type"] = "document_search"

            result["content"] = (
                result.get("content")
                or result.get("text")
                or metadata_text
            )

            results.append(
                (
                    score,
                    result,
                )
            )

        results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            result
            for _, result in results[: self.top_k]
        ]

    # ============================================================
    # TABLE SEARCH
    # ============================================================

    def _table_search(self, query):
        if not query or not self.documents:
            return []

        words = {
            word.lower().strip(
                ".,!?;:()[]{}\"'"
            )
            for word in query.split()
            if len(
                word.strip(
                    ".,!?;:()[]{}\"'"
                )
            ) > 2
        }

        if not words:
            return []

        results = []

        for document in self.documents:

            if not isinstance(document, dict):
                continue

            if not self._source_matches_selection(
                document
            ):
                continue

            tables = document.get(
                "tables",
                [],
            )

            if not isinstance(tables, list):
                continue

            metadata = document.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            document_source = (
                document.get("source")
                or metadata.get(
                    "uploaded_file_name"
                )
                or metadata.get(
                    "filename"
                )
            )

            for table_index, table in enumerate(
                tables
            ):

                if not isinstance(table, dict):
                    continue

                table_data = (
                    table.get("table")
                    or table.get("data")
                    or table.get("rows")
                    or []
                )

                searchable = json.dumps(
                    table_data,
                    ensure_ascii=False,
                    default=str,
                ).lower()

                matched_words = [
                    word
                    for word in words
                    if word in searchable
                ]

                score = len(matched_words)

                if score <= 0:
                    continue

                result = dict(table)

                if document_source:
                    result.setdefault(
                        "source",
                        document_source,
                    )

                result.setdefault(
                    "table_index",
                    table_index,
                )

                result["similarity_score"] = (
                    score / max(
                        len(words),
                        1,
                    )
                )

                result["source_type"] = "table"
                result["search_type"] = "table_search"

                result["content"] = json.dumps(
                    table_data,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

                results.append(
                    (
                        score,
                        result,
                    )
                )

        results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            result
            for _, result in results[: self.top_k]
        ]

    # ============================================================
    # OPTIONAL WEB SEARCH
    # ============================================================

    def _web_search(self, query):
        """
        model.py does not currently expose web_search().

        If llm.py/GeminiLLM is supplied instead, this method can
        still use it automatically.
        """

        if not query or not query.strip():
            return []

        web_method = getattr(
            self.llm,
            "web_search",
            None,
        )

        if not callable(web_method):
            return []

        try:
            results = web_method(
                query=query,
                max_results=self.top_k,
            )

            return self._normalize_results(
                results,
                "web_search",
            )

        except Exception as exc:
            print(
                f"Web search failed: {exc}"
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
            return self._vector_search(query)

        if action == "document_search":
            return self._document_search(query)

        if action == "table_search":
            return self._table_search(query)

        if action == "web_search":
            return self._web_search(query)

        return []

    # ============================================================
    # NORMALIZE RESULTS
    # ============================================================

    def _normalize_results(
        self,
        results,
        action,
    ):
        if not isinstance(results, list):
            return []

        normalized = []

        for result in results:

            if not isinstance(result, dict):
                continue

            item = dict(result)

            item.setdefault(
                "search_type",
                action,
            )

            item.setdefault(
                "source_type",
                (
                    "web"
                    if action == "web_search"
                    else "document"
                ),
            )

            if not item.get("content"):
                item["content"] = (
                    item.get("text")
                    or item.get("page_content")
                    or ""
                )

            normalized.append(item)

        return normalized

    # ============================================================
    # RESULT KEY
    # ============================================================

    def _result_key(self, result):
        if not isinstance(result, dict):
            return None

        source = (
            result.get("source")
            or result.get("filename")
            or result.get("file_name")
            or result.get("url")
            or ""
        )

        chunk_id = (
            result.get("chunk_id")
            or result.get("id")
            or result.get("table_index")
            or ""
        )

        content = str(
            result.get(
                "content",
                result.get(
                    "text",
                    "",
                ),
            )
        )

        return (
            str(source),
            str(chunk_id),
            content[:500],
        )

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    def _deduplicate_results(self, results):
        unique = []
        seen = set()

        for result in results:

            key = self._result_key(result)

            if key is None:
                unique.append(result)
                continue

            if key in seen:
                continue

            seen.add(key)
            unique.append(result)

        return unique

    # ============================================================
    # FALLBACK ACTION
    # ============================================================

    def _fallback_action(
        self,
        previous_action=None,
    ):
        if previous_action == "vector_search":
            return "document_search"

        if previous_action == "document_search":
            return "table_search"

        if previous_action == "table_search":
            return "vector_search"

        if previous_action == "web_search":
            return "vector_search"

        if self.selected_sources:
            return "vector_search"

        return "vector_search"

    # ============================================================
    # DETERMINISTIC ROUTING
    # ============================================================

    def _deterministic_action(self, query):
        q = query.lower().strip()

        document_patterns = [
            "uploaded document",
            "uploaded file",
            "this document",
            "this file",
            "the document",
            "the file",
            "in the document",
            "in this document",
            "from the document",
            "from this document",
            "according to the document",
            "according to this document",
            "according to the paper",
            "in the paper",
            "from the paper",
            "this paper",
        ]

        if any(
            pattern in q
            for pattern in document_patterns
        ):
            return "vector_search"

        metadata_terms = [
            "document title",
            "title of the document",
            "document name",
            "file name",
            "filename",
            "document author",
            "author of the document",
            "document subject",
            "how many pages",
            "page count",
            "number of pages",
        ]

        if any(
            term in q
            for term in metadata_terms
        ):
            return "document_search"

        table_terms = [
            "table",
            "row",
            "rows",
            "column",
            "columns",
            "spreadsheet",
            "excel",
            "cell",
            "sales total",
            "total sales",
            "revenue total",
            "total revenue",
        ]

        if any(
            term in q
            for term in table_terms
        ):
            return "table_search"

        return None

    # ============================================================
    # PLANNING
    # ============================================================

    def _plan_action(
        self,
        query,
        previous_actions,
        previous_evaluations,
    ):
        """
        First use deterministic routing.

        If no deterministic rule matches, ask model.py to return
        a JSON retrieval plan.
        """

        deterministic = (
            self._deterministic_action(query)
        )

        if deterministic:
            return {
                "action": deterministic,
                "query": query,
                "reason": (
                    "Selected using deterministic "
                    "document/table routing."
                ),
            }

        prompt = f"""
You are the planner of an Agentic RAG system.

Available retrieval tools:

1. vector_search
   Search uploaded document CONTENT.

2. document_search
   Search uploaded document METADATA.

3. table_search
   Search uploaded document TABLES.

4. web_search
   Public internet search, only if the supplied model supports it.

IMPORTANT:
- Questions about uploaded documents -> vector_search.
- Document filename/title/author/page count -> document_search.
- Tables/spreadsheets/rows/columns -> table_search.
- Public/current information -> web_search only if available.
- Never invent evidence.
- Prefer uploaded-document retrieval when the user refers to
  an uploaded document.

Previous actions:
{json.dumps(previous_actions, indent=2, default=str)}

Previous evaluations:
{json.dumps(previous_evaluations, indent=2, default=str)}

User question:
{query}

Return ONLY valid JSON.

Required format:

{{
    "action": "vector_search",
    "query": "optimized retrieval query",
    "reason": "why this retrieval tool is appropriate"
}}
"""

        try:
            raw = self._model_generate(prompt)

            plan = json.loads(
                self._clean_json(raw)
            )

            if not isinstance(plan, dict):
                raise ValueError(
                    "Planner did not return an object."
                )

            action = plan.get(
                "action",
                "vector_search",
            )

            if action not in self.allowed_actions:
                action = "vector_search"

            # model.py has no web_search, so don't route to it
            # unless the supplied model actually supports it.
            if (
                action == "web_search"
                and not callable(
                    getattr(
                        self.llm,
                        "web_search",
                        None,
                    )
                )
            ):
                action = "vector_search"

            return {
                "action": action,
                "query": (
                    str(
                        plan.get(
                            "query",
                            query,
                        )
                    ).strip()
                    or query
                ),
                "reason": str(
                    plan.get(
                        "reason",
                        "Model-selected retrieval strategy.",
                    )
                ),
            }

        except Exception as exc:
            print(
                f"Planner failed: {exc}"
            )

            return {
                "action": self._fallback_action(
                    previous_actions[-1]
                    if previous_actions
                    else None
                ),
                "query": query,
                "reason": (
                    "Planner failed; using "
                    "deterministic fallback."
                ),
            }

    # ============================================================
    # EVIDENCE EVALUATION
    # ============================================================

    def _evaluate_evidence(
        self,
        query,
        results,
        action,
    ):
        if not results:
            return {
                "sufficient": False,
                "confidence": 0.0,
                "reason": "No retrieval evidence.",
                "recommended_action": (
                    self._fallback_action(action)
                ),
            }

        evidence = []

        for result in results:

            if not isinstance(result, dict):
                continue

            evidence.append(
                {
                    "title": str(
                        result.get(
                            "title",
                            "",
                        )
                    ),
                    "source": str(
                        result.get(
                            "source",
                            "Unknown",
                        )
                    ),
                    "content": str(
                        result.get(
                            "content",
                            result.get(
                                "text",
                                "",
                            ),
                        )
                    )[:4000],
                    "score": result.get(
                        "similarity_score",
                        result.get(
                            "score",
                            0,
                        ),
                    ),
                }
            )

        if not evidence:
            return {
                "sufficient": False,
                "confidence": 0.0,
                "reason": "No usable evidence.",
                "recommended_action": (
                    self._fallback_action(action)
                ),
            }

        prompt = f"""
You are the evidence evaluator of an Agentic RAG system.

User question:
{query}

Retrieval tool used:
{action}

Retrieved evidence:
{json.dumps(
    evidence,
    indent=2,
    ensure_ascii=False,
    default=str,
)}

Determine whether the retrieved evidence directly supports
an answer to the user's question.

Rules:
- sufficient=true only when evidence is relevant enough to answer.
- sufficient=false when evidence is irrelevant or incomplete.
- Never invent facts.
- confidence must be between 0 and 1.
- If evidence is insufficient, recommend another retrieval tool.
- For document content use vector_search.
- For metadata use document_search.
- For tables use table_search.
- For public/current information use web_search only if available.

Return ONLY valid JSON:

{{
    "sufficient": true,
    "confidence": 0.90,
    "reason": "Evidence directly supports the question.",
    "recommended_action": "vector_search"
}}
"""

        try:
            raw = self._model_generate(prompt)

            evaluation = json.loads(
                self._clean_json(raw)
            )

            if not isinstance(
                evaluation,
                dict,
            ):
                raise ValueError(
                    "Invalid evaluator response."
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

            confidence = max(
                0.0,
                min(
                    confidence,
                    1.0,
                ),
            )

            sufficient = bool(
                evaluation.get(
                    "sufficient",
                    False,
                )
            )

            recommended_action = (
                evaluation.get(
                    "recommended_action",
                    self._fallback_action(action),
                )
            )

            if recommended_action not in self.allowed_actions:
                recommended_action = (
                    self._fallback_action(action)
                )

            if (
                recommended_action == "web_search"
                and not callable(
                    getattr(
                        self.llm,
                        "web_search",
                        None,
                    )
                )
            ):
                recommended_action = "vector_search"

            return {
                "sufficient": sufficient,
                "confidence": confidence,
                "reason": str(
                    evaluation.get(
                        "reason",
                        "",
                    )
                ),
                "recommended_action": (
                    recommended_action
                ),
            }

        except Exception as exc:
            print(
                f"Evidence evaluation failed: {exc}"
            )

            return {
                "sufficient": False,
                "confidence": 0.0,
                "reason": (
                    "Evidence evaluator failed."
                ),
                "recommended_action": (
                    self._fallback_action(action)
                ),
            }

    # ============================================================
    # SOURCE TYPE
    # ============================================================

    def _determine_source_type(self, results):
        source_types = {
            str(
                result.get(
                    "source_type",
                    "",
                )
            ).lower()
            for result in results
            if isinstance(result, dict)
        }

        if "web" in source_types:
            return "web"

        if "table" in source_types:
            return "table"

        return "document"

    # ============================================================
    # FINAL ANSWER
    # ============================================================

    def _generate_answer(
        self,
        query,
        chunks,
        source_type,
    ):
        if not chunks:
            return (
                "I could not find sufficient "
                "evidence to answer that question."
            )

        context_parts = []
        sources = []

        for chunk in chunks:

            if not isinstance(chunk, dict):
                continue

            content = (
                chunk.get("content")
                or chunk.get("text")
                or ""
            )

            if content:
                context_parts.append(
                    str(content)
                )

            source = (
                chunk.get("source")
                or chunk.get("filename")
                or chunk.get("file_name")
            )

            if source:
                sources.append(
                    str(source)
                )

        context = "\n\n---\n\n".join(
            context_parts
        )

        if not context.strip():
            return (
                "I could not find sufficient "
                "evidence to answer that question."
            )

        source_text = ", ".join(
            sorted(set(sources))
        ) or "retrieved documents"

        if source_type == "table":

            instruction = """
Answer using ONLY the retrieved table evidence.

If the table does not contain enough information,
say that the information is not available in the
retrieved tables.

Do not invent numbers.
"""

        elif source_type == "web":

            instruction = """
Answer using ONLY the retrieved web evidence.

Do not invent facts.
Mention that the answer is based on web research
and include URLs when they are available.
"""

        else:

            instruction = """
Answer using ONLY the retrieved uploaded-document evidence.

Do not use outside knowledge.
Do not invent facts.

If the evidence does not contain the answer, say:

"I could not find that information in the uploaded documents."

Mention the document/source when possible.
"""

        prompt = f"""
You are the final answer generator of an Enterprise RAG system.

USER QUESTION:
{query}

SOURCE TYPE:
{source_type}

SOURCES:
{source_text}

RETRIEVED EVIDENCE:
{context}

{instruction}

Give a clear, direct answer.
"""

        try:
            return self._model_generate(prompt)

        except Exception as exc:
            print(
                f"Answer generation failed: {exc}"
            )

            return (
                "I could not generate an answer "
                "from the retrieved evidence."
            )

    # ============================================================
    # MAIN AGENT LOOP
    # ============================================================

    def run(self, query):
        if not query or not query.strip():
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "action": None,
                "success": False,
                "iterations": 0,
                "trace": [],
            }

        query = query.strip()

        state = self._create_state(
            query
        )

        working_query = query

        conversation_context = (
            self._conversation_context()
        )

        if conversation_context:
            working_query = (
                "Conversation context:\n"
                + conversation_context
                + "\n\nCurrent question:\n"
                + query
            )

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

        # ========================================================
        # ITERATIVE RETRIEVAL
        # ========================================================

        while (
            state["iterations"]
            < self.max_iterations
        ):

            state["iterations"] += 1

            iteration = state["iterations"]

            # ----------------------------------------------------
            # PLAN
            # ----------------------------------------------------

            plan = self._plan_action(
                query=working_query,
                previous_actions=(
                    state["previous_actions"]
                ),
                previous_evaluations=(
                    state["previous_evaluations"]
                ),
            )

            action = plan.get(
                "action",
                "vector_search",
            )

            if action not in self.allowed_actions:
                action = "vector_search"

            search_query = (
                str(
                    plan.get(
                        "query",
                        working_query,
                    )
                ).strip()
                or working_query
            )

            reason = str(
                plan.get(
                    "reason",
                    "",
                )
            )

            state["action"] = action

            state["previous_actions"].append(
                action
            )

            if action == "web_search":
                state["web_used"] = True

            elif action in (
                "vector_search",
                "document_search",
            ):
                state["document_used"] = True

            elif action == "table_search":
                state["table_used"] = True

            self._trace(
                state,
                "agent_plan",
                {
                    "iteration": iteration,
                    "tool": action,
                    "query": search_query,
                    "reason": reason,
                    "selected_sources": (
                        self.selected_sources
                    ),
                },
            )

            print(
                f"\nIteration {iteration}"
            )

            print(
                f"Agent selected: {action}"
            )

            print(
                f"Search query: {search_query}"
            )

            # ----------------------------------------------------
            # EXECUTE
            # ----------------------------------------------------

            try:
                results = self.execute_tool(
                    action,
                    search_query,
                )

            except Exception as exc:
                print(
                    f"Tool execution failed: {exc}"
                )
                results = []

            results = self._normalize_results(
                results,
                action,
            )

            state["results"] = results

            state["all_results"].extend(
                results
            )

            state["all_results"] = (
                self._deduplicate_results(
                    state["all_results"]
                )
            )

            self._trace(
                state,
                "tool_result",
                {
                    "tool": action,
                    "results": len(results),
                    "total_unique_results": len(
                        state["all_results"]
                    ),
                },
            )

            print(
                f"Retrieved: {len(results)} results"
            )

            # ----------------------------------------------------
            # EVALUATE
            # ----------------------------------------------------

            evaluation = self._evaluate_evidence(
                query=query,
                results=results,
                action=action,
            )

            state["previous_evaluations"].append(
                evaluation
            )

            self._trace(
                state,
                "evidence_evaluation",
                evaluation,
            )

            print(
                "Evidence sufficient:",
                evaluation["sufficient"],
            )

            print(
                "Confidence:",
                f"{evaluation['confidence']:.2f}",
            )

            # ----------------------------------------------------
            # SUCCESS
            # ----------------------------------------------------

            if (
                evaluation["sufficient"]
                and results
            ):
                break

            # ----------------------------------------------------
            # REPLAN
            # ----------------------------------------------------

            if (
                iteration
                >= self.max_iterations
            ):
                break

            recommended_action = (
                evaluation.get(
                    "recommended_action"
                )
            )

            if (
                recommended_action
                not in self.allowed_actions
            ):
                recommended_action = (
                    self._fallback_action(action)
                )

            self._trace(
                state,
                "replan",
                {
                    "next_action": (
                        recommended_action
                    ),
                    "reason": evaluation.get(
                        "reason",
                        "",
                    ),
                },
            )

            print(
                "Replanning ->",
                recommended_action,
            )

            # Keep original user intent instead of replacing it
            # permanently with the planner's query.
            working_query = query

        # ========================================================
        # FINAL EVIDENCE
        # ========================================================

        final_results = (
            self._deduplicate_results(
                state["all_results"]
            )
        )

        if not final_results:
            state["success"] = False

            state["answer"] = (
                "I could not find sufficient "
                "evidence to answer that question."
            )

            self._trace(
                state,
                "answer_failed",
                {
                    "reason": (
                        "No retrieval evidence "
                        "was found."
                    ),
                },
            )

            return {
                "answer": state["answer"],
                "sources": [],
                "action": state["action"],
                "success": False,
                "iterations": state["iterations"],
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
        # GENERATE FINAL ANSWER
        # ========================================================

        answer = self._generate_answer(
            query=query,
            chunks=final_results,
            source_type=source_type,
        )

        if not answer:
            answer = (
                "Evidence was retrieved, "
                "but no final answer was generated."
            )

        state["answer"] = str(
            answer
        ).strip()

        state["success"] = True

        self._trace(
            state,
            "answer_generated",
            {
                "source_type": source_type,
                "evidence_count": len(
                    final_results
                ),
                "selected_sources": (
                    self.selected_sources
                ),
            },
        )

        return {
            "answer": state["answer"],
            "sources": final_results,
            "action": state["action"],
            "success": True,
            "iterations": state["iterations"],
            "trace": state["trace"],
        }