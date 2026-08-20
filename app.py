import json


class EnterpriseRAGAgent:
    """
    Enterprise Agentic RAG controller.

    Retrieval tools:
        1. vector_search
        2. document_search
        3. table_search
        4. web_search

    The agent:
        - plans which retrieval tool to use
        - executes retrieval
        - evaluates retrieved evidence
        - replans when evidence is insufficient
        - generates a grounded final answer

    This class is intentionally compatible with the GeminiLLM
    implementation provided by llm.py.
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
        """
        Normalize source names.

        Accepts:
            ["a.pdf", "b.pdf"]
            "a.pdf"
            ["a.pdf,b.pdf"]
        """

        if not sources:
            return []

        if isinstance(
            sources,
            str,
        ):
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

        return sorted(
            normalized
        )

    # ============================================================
    # SOURCE MATCHING
    # ============================================================

    def _source_matches_selection(
        self,
        result,
    ):
        """
        Determines whether a result belongs to one of the
        currently selected documents.

        If no sources are selected, everything is allowed.
        """

        if not self.selected_sources:
            return True

        if not isinstance(
            result,
            dict,
        ):
            return False

        possible_sources = []

        # --------------------------------------------------------
        # Top-level fields
        # --------------------------------------------------------

        for key in (
            "source",
            "filename",
            "file_name",
            "document",
            "document_name",
        ):

            value = result.get(
                key
            )

            if value:
                possible_sources.append(
                    str(value)
                )

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        metadata = result.get(
            "metadata",
            {},
        )

        if isinstance(
            metadata,
            dict,
        ):

            for key in (
                "uploaded_file_name",
                "file_name",
                "filename",
                "source",
                "document",
                "document_name",
            ):

                value = metadata.get(
                    key
                )

                if value:
                    possible_sources.append(
                        str(value)
                    )

        if not possible_sources:
            return False

        normalized_result_sources = (
            self._normalize_sources(
                possible_sources
            )
        )

        selected = set(
            self.selected_sources
        )

        result_sources = set(
            normalized_result_sources
        )

        # Exact match
        if selected & result_sources:
            return True

        # --------------------------------------------------------
        # Basename compatibility
        # --------------------------------------------------------

        selected_basenames = {
            self._basename(
                source
            )
            for source in selected
        }

        result_basenames = {
            self._basename(
                source
            )
            for source in result_sources
        }

        return bool(
            selected_basenames
            & result_basenames
        )

    # ============================================================
    # BASENAME
    # ============================================================

    @staticmethod
    def _basename(value):
        if not value:
            return ""

        value = str(value)

        # Handle Windows and Unix paths.
        value = value.replace(
            "\\",
            "/",
        )

        return value.rstrip(
            "/"
        ).split(
            "/"
        )[-1]

    # ============================================================
    # STATE
    # ============================================================

    def _create_state(
        self,
        query,
    ):
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

    def _vector_search(
        self,
        query,
    ):
        """
        Search the persistent vector database.

        Compatible with both:

            search(query, top_k=...)
        and

            search(query, top_k=..., sources=...)
        """

        if not query or not query.strip():
            return []

        if self.vector_db is None:
            return []

        try:

            # ----------------------------------------------------
            # Load persistent database if necessary.
            # ----------------------------------------------------

            index = getattr(
                self.vector_db,
                "index",
                None,
            )

            if index is None:

                exists = False

                try:
                    exists = bool(
                        self.vector_db.exists()
                    )
                except Exception:
                    exists = False

                if exists:

                    try:
                        self.vector_db.load()
                    except Exception as e:
                        print(
                            f"Vector database load failed: {e}"
                        )
                        return []

                else:
                    return []

            if getattr(
                self.vector_db,
                "index",
                None,
            ) is None:

                return []

            # ----------------------------------------------------
            # Search with source filtering if supported.
            # ----------------------------------------------------

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

                # Older VectorDatabase implementation.
                results = self.vector_db.search(
                    query,
                    top_k=self.top_k,
                )

            # ----------------------------------------------------
            # Normalize.
            # ----------------------------------------------------

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

                item = dict(
                    result
                )

                item["source_type"] = (
                    "document"
                )

                item["search_type"] = (
                    "vector_search"
                )

                normalized.append(
                    item
                )

            return normalized[
                : self.top_k
            ]

        except Exception as e:

            print(
                f"Vector search failed: {e}"
            )

            return []

    # ============================================================
    # DOCUMENT SEARCH
    # ============================================================

    def _document_search(
        self,
        query,
    ):
        """
        Search document text and metadata supplied to the agent.

        This is primarily useful for:
            - filename
            - title
            - author
            - page count
            - metadata
            - exact document properties
        """

        if not query:
            return []

        if not self.documents:
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

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            text = str(
                document.get(
                    "text",
                    "",
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
                    "",
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

            score = len(
                matched_words
            )

            if score <= 0:
                continue

            result = dict(
                document
            )

            result["similarity_score"] = (
                score / max(
                    len(words),
                    1,
                )
            )

            result["source_type"] = (
                "document"
            )

            result["search_type"] = (
                "document_search"
            )

            # Make metadata searchable by the final
            # answer generator.
            result["content"] = (
                result.get(
                    "content"
                )
                or result.get(
                    "text"
                )
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
            for _, result in results[
                : self.top_k
            ]
        ]

    # ============================================================
    # TABLE SEARCH
    # ============================================================

    def _table_search(
        self,
        query,
    ):
        """
        Search extracted document tables.
        """

        if not query:
            return []

        if not self.documents:
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

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            document_source = (
                document.get(
                    "source"
                )
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

                if not isinstance(
                    table,
                    dict,
                ):
                    continue

                table_data = table.get(
                    "table",
                    table.get(
                        "data",
                        table.get(
                            "rows",
                            [],
                        ),
                    ),
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

                score = len(
                    matched_words
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

                result["source_type"] = (
                    "table"
                )

                result["search_type"] = (
                    "table_search"
                )

                result["content"] = (
                    json.dumps(
                        table_data,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
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
            for _, result in results[
                : self.top_k
            ]
        ]

    # ============================================================
    # WEB SEARCH
    # ============================================================

    def _web_search(
        self,
        query,
    ):
        """
        Use GeminiLLM.web_search(), which performs the actual
        OpenRouter web search.
        """

        if not query or not query.strip():
            return []

        try:

            results = self.llm.web_search(
                query=query,
                max_results=self.top_k,
            )

            return self._normalize_results(
                results,
                "web_search",
            )

        except Exception as e:

            print(
                f"Web search failed: {e}"
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

        if action == "web_search":
            return self._web_search(
                query
            )

        print(
            f"Unknown agent action: {action}"
        )

        return []

    # ============================================================
    # NORMALIZE RESULTS
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

            item.setdefault(
                "source_type",
                (
                    "web"
                    if action == "web_search"
                    else "document"
                ),
            )

            # Web results use content, while document
            # results may use text.
            if not item.get(
                "content"
            ):

                if item.get(
                    "text"
                ):

                    item["content"] = (
                        item.get(
                            "text"
                        )
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
                "url"
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
                "table_index"
            )
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

            seen.add(
                key
            )

            unique.append(
                result
            )

        return unique

    # ============================================================
    # CONVERSATION CONTEXT
    # ============================================================

    def _conversation_context(
        self,
    ):
        if not self.conversation_history:
            return ""

        recent = (
            self.conversation_history[
                -8:
            ]
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
            )

            content = str(
                message.get(
                    "content",
                    "",
                )
            )

            if not content.strip():
                continue

            lines.append(
                f"{role.upper()}: {content}"
            )

        return "\n".join(
            lines
        )

    # ============================================================
    # FALLBACK ACTION
    # ============================================================

    def _fallback_action(
        self,
        previous_action=None,
    ):
        """
        Choose a sensible fallback when the planner/evaluator
        cannot determine the next tool.
        """

        if previous_action == "web_search":

            if self.selected_sources:
                return "vector_search"

            return "web_search"

        if previous_action in (
            "vector_search",
            "document_search",
            "table_search",
        ):

            return "web_search"

        if self.selected_sources:
            return "vector_search"

        return "web_search"

    # ============================================================
    # QUERY CLEANING
    # ============================================================

    @staticmethod
    def _clean_query(
        query,
        fallback,
    ):
        if not isinstance(
            query,
            str,
        ):
            return fallback

        query = query.strip()

        if not query:
            return fallback

        return query

    # ============================================================
    # SOURCE TYPE
    # ============================================================

    def _determine_source_type(
        self,
        results,
    ):
        source_types = {
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

        if "web" in source_types:
            return "web"

        if "table" in source_types:
            return "table"

        return "document"

    # ============================================================
    # AGENT LOOP
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

        conversation_context = (
            self._conversation_context()
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
        # ITERATIVE AGENT LOOP
        # ========================================================

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

                action = forced_action

                search_query = (
                    working_query
                )

                reason = (
                    "Replanned after "
                    "insufficient evidence."
                )

                forced_action = None

            else:

                try:

                    planning_query = (
                        working_query
                    )

                    if conversation_context:

                        planning_query = (
                            "Conversation context:\n"
                            + conversation_context
                            + "\n\n"
                            + "Current question:\n"
                            + working_query
                        )

                    plan = (
                        self.llm.plan_action(
                            query=planning_query,
                            previous_actions=(
                                state[
                                    "previous_actions"
                                ]
                            ),
                            previous_evaluations=(
                                state[
                                    "previous_evaluations"
                                ]
                            ),
                        )
                    )

                except Exception as e:

                    print(
                        f"Planner failed: {e}"
                    )

                    plan = {
                        "action": (
                            self._fallback_action(
                                state[
                                    "previous_actions"
                                ][-1]
                                if state[
                                    "previous_actions"
                                ]
                                else None
                            )
                        ),
                        "query": working_query,
                        "reason": (
                            "Planner failure "
                            "fallback."
                        ),
                    }

                if not isinstance(
                    plan,
                    dict,
                ):

                    plan = {
                        "action": (
                            self._fallback_action()
                        ),
                        "query": working_query,
                        "reason": (
                            "Invalid planner "
                            "response."
                        ),
                    }

                action = plan.get(
                    "action",
                    "web_search",
                )

                if action not in (
                    self.allowed_actions
                ):

                    action = (
                        self._fallback_action()
                    )

                search_query = (
                    self._clean_query(
                        plan.get(
                            "query",
                            working_query,
                        ),
                        working_query,
                    )
                )

                reason = str(
                    plan.get(
                        "reason",
                        "",
                    )
                )

            state["action"] = action

            state[
                "previous_actions"
            ].append(
                action
            )

            # ----------------------------------------------------
            # TRACK SOURCE USAGE
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # TRACE PLAN
            # ----------------------------------------------------

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

                results = (
                    self.execute_tool(
                        action,
                        search_query,
                    )
                )

            except Exception as e:

                print(
                    f"Tool execution failed: {e}"
                )

                results = []

            results = (
                self._normalize_results(
                    results,
                    action,
                )
            )

            state[
                "results"
            ] = results

            state[
                "all_results"
            ].extend(
                results
            )

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
                f"Retrieved: {len(results)} results"
            )

            # ----------------------------------------------------
            # EVALUATE CURRENT RETRIEVAL
            # ----------------------------------------------------

            try:

                evaluation = (
                    self.llm.evaluate_evidence(
                        query=(
                            state[
                                "original_query"
                            ]
                        ),
                        results=results,
                        action=action,
                    )
                )

            except Exception as e:

                print(
                    f"Evaluator failed: {e}"
                )

                evaluation = {
                    "sufficient": False,
                    "confidence": 0.0,
                    "reason": (
                        "Evidence evaluator "
                        "failed."
                    ),
                    "recommended_action": (
                        self._fallback_action(
                            action
                        )
                    ),
                }

            if not isinstance(
                evaluation,
                dict,
            ):

                evaluation = {
                    "sufficient": False,
                    "confidence": 0.0,
                    "reason": (
                        "Invalid evaluator "
                        "response."
                    ),
                    "recommended_action": (
                        self._fallback_action(
                            action
                        )
                    ),
                }

            # ----------------------------------------------------
            # NORMALIZE EVALUATION
            # ----------------------------------------------------

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
                    "recommended_action"
                )
            )

            if (
                recommended_action
                not in self.allowed_actions
            ):

                recommended_action = (
                    self._fallback_action(
                        action
                    )
                )

            evaluation[
                "confidence"
            ] = confidence

            evaluation[
                "sufficient"
            ] = sufficient

            evaluation[
                "recommended_action"
            ] = recommended_action

            state[
                "previous_evaluations"
            ].append(
                evaluation
            )

            self._trace(
                state,
                "evidence_evaluation",
                evaluation,
            )

            print(
                f"Evidence sufficient: {sufficient}"
            )

            print(
                f"Confidence: {confidence:.2f}"
            )

            # ----------------------------------------------------
            # SUCCESS
            # ----------------------------------------------------

            if sufficient and results:

                break

            # ----------------------------------------------------
            # REPLAN
            # ----------------------------------------------------

            working_query = (
                search_query
            )

            if (
                iteration
                < self.max_iterations
            ):

                forced_action = (
                    recommended_action
                )

                self._trace(
                    state,
                    "replan",
                    {
                        "next_action": (
                            recommended_action
                        ),
                        "reason": (
                            evaluation.get(
                                "reason",
                                "",
                            )
                        ),
                    },
                )

                print(
                    "Replanning → "
                    + str(
                        recommended_action
                    )
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

        # --------------------------------------------------------
        # If every retrieval attempt failed.
        # --------------------------------------------------------

        if not final_results:

            state[
                "success"
            ] = False

            state[
                "answer"
            ] = (
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
                    )
                },
            )

            return {
                "answer": state[
                    "answer"
                ],
                "sources": [],
                "action": state[
                    "action"
                ],
                "success": False,
                "iterations": state[
                    "iterations"
                ],
                "trace": state[
                    "trace"
                ],
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

        try:

            answer = (
                self.llm.generate_answer(
                    query=(
                        state[
                            "original_query"
                        ]
                    ),
                    chunks=final_results,
                    source_type=source_type,
                )
            )

        except Exception as e:

            print(
                f"Answer generation failed: {e}"
            )

            answer = (
                "Evidence was retrieved, "
                "but I could not generate "
                "the final answer."
            )

        if not answer or not str(
            answer
        ).strip():

            answer = (
                "Evidence was retrieved, "
                "but no final answer was generated."
            )

        state[
            "answer"
        ] = str(
            answer
        ).strip()

        state[
            "success"
        ] = True

        # ========================================================
        # FINAL TRACE
        # ========================================================

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
            "success": True,
            "iterations": state[
                "iterations"
            ],
            "trace": state[
                "trace"
            ],
        }