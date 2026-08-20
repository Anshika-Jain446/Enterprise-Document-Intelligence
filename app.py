import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from model import GeminiModel


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENTERPRISE RAG AGENT
# ============================================================

class EnterpriseRAGAgent:
    """
    Enterprise Agentic RAG controller.

    Compatible with the current model.py GeminiModel.

    Retrieval tools:
        1. vector_search
        2. document_search
        3. table_search
        4. web_search

    The agent:
        - plans retrieval
        - executes retrieval
        - evaluates evidence
        - replans when evidence is insufficient
        - generates a grounded final answer
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

    # ========================================================
    # SOURCE NORMALIZATION
    # ========================================================

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

    # ========================================================
    # BASENAME
    # ========================================================

    @staticmethod
    def _basename(value):
        if not value:
            return ""

        value = str(value).replace("\\", "/")

        return value.rstrip("/").split("/")[-1]

    # ========================================================
    # SOURCE MATCHING
    # ========================================================

    def _source_matches_selection(self, result):

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
                possible_sources.append(
                    str(value)
                )

        metadata = result.get(
            "metadata",
            {},
        )

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
                    possible_sources.append(
                        str(value)
                    )

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
            selected_basenames
            & result_basenames
        )

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

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

                item["source_type"] = "document"
                item["search_type"] = "vector_search"

                if not item.get("content"):
                    item["content"] = (
                        item.get("text")
                        or item.get("page_content")
                        or ""
                    )

                normalized.append(item)

            return normalized[:self.top_k]

        except Exception as exc:

            print(
                f"Vector search failed: {exc}"
            )

            return []

    # ========================================================
    # DOCUMENT SEARCH
    # ========================================================

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

            score = len(
                matched_words
            )

            if score <= 0:
                continue

            result = dict(document)

            result["similarity_score"] = (
                score
                / max(
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
            for _, result
            in results[:self.top_k]
        ]

    # ========================================================
    # TABLE SEARCH
    # ========================================================

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

                score = len(
                    matched_words
                )

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
                    score
                    / max(
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
            for _, result
            in results[:self.top_k]
        ]

    # ========================================================
    # WEB SEARCH
    # ========================================================

    def _web_search(self, query):

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

    # ========================================================
    # TOOL EXECUTION
    # ========================================================

    def execute_tool(
        self,
        action,
        query,
    ):

        if action == "vector_search":
            return self._vector_search(query)

        if action == "document_search":
            return self._document_search(query)

        if action == "table_search":
            return self._table_search(query)

        if action == "web_search":
            return self._web_search(query)

        return []

    # ========================================================
    # NORMALIZE RESULTS
    # ========================================================

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

    # ========================================================
    # RESULT KEY
    # ========================================================

    def _result_key(self, result):

        if not isinstance(
            result,
            dict,
        ):
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

    # ========================================================
    # DEDUPLICATION
    # ========================================================

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
                unique.append(result)
                continue

            if key in seen:
                continue

            seen.add(key)
            unique.append(result)

        return unique

    # ========================================================
    # FALLBACK
    # ========================================================

    def _fallback_action(
        self,
        previous_action=None,
    ):

        web_available = callable(
            getattr(
                self.llm,
                "web_search",
                None,
            )
        )

        if previous_action == "vector_search":
            return "document_search"

        if previous_action == "document_search":
            return "table_search"

        if previous_action == "table_search":

            if web_available:
                return "web_search"

            return "vector_search"

        if previous_action == "web_search":
            return "vector_search"

        return "vector_search"

    # ========================================================
    # DETERMINISTIC ROUTING
    # ========================================================

    def _deterministic_action(
        self,
        query,
    ):

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

    # ========================================================
    # PLANNING
    # ========================================================

    def _plan_action(
        self,
        query,
        previous_actions,
        previous_evaluations,
    ):

        deterministic = (
            self._deterministic_action(
                query
            )
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

        planner = getattr(
            self.llm,
            "plan_action",
            None,
        )

        if callable(planner):

            try:

                plan = planner(
                    query=query,
                    previous_actions=(
                        previous_actions
                    ),
                    previous_evaluations=(
                        previous_evaluations
                    ),
                )

                if isinstance(
                    plan,
                    dict,
                ):

                    action = plan.get(
                        "action",
                        "vector_search",
                    )

                    if action not in self.allowed_actions:
                        action = "vector_search"

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
                    f"Model planner failed: {exc}"
                )

        return {
            "action": self._fallback_action(
                previous_actions[-1]
                if previous_actions
                else None
            ),
            "query": query,
            "reason": (
                "Planner unavailable; "
                "using deterministic fallback."
            ),
        }

    # ========================================================
    # EVALUATE EVIDENCE
    # ========================================================

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
                    self._fallback_action(
                        action
                    )
                ),
            }

        evaluator = getattr(
            self.llm,
            "evaluate_evidence",
            None,
        )

        if callable(evaluator):

            try:

                evaluation = evaluator(
                    query=query,
                    results=results,
                    action=action,
                )

                if isinstance(
                    evaluation,
                    dict,
                ):

                    recommended_action = (
                        evaluation.get(
                            "recommended_action",
                            self._fallback_action(
                                action
                            ),
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
                        recommended_action = (
                            "vector_search"
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

                    return {
                        "sufficient": bool(
                            evaluation.get(
                                "sufficient",
                                False,
                            )
                        ),
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
                    f"Evidence evaluator failed: {exc}"
                )

        return {
            "sufficient": bool(results),
            "confidence": 0.5,
            "reason": (
                "Evidence was retrieved."
            ),
            "recommended_action": (
                self._fallback_action(
                    action
                )
            ),
        }

    # ========================================================
    # SOURCE TYPE
    # ========================================================

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

    # ========================================================
    # FINAL ANSWER
    # ========================================================

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

        generator = getattr(
            self.llm,
            "generate_answer",
            None,
        )

        if callable(generator):

            try:

                return generator(
                    query=query,
                    chunks=chunks,
                    source_type=source_type,
                )

            except Exception as exc:

                print(
                    f"Model answer generation failed: {exc}"
                )

        # Safe fallback

        for chunk in chunks:

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            content = (
                chunk.get("content")
                or chunk.get("text")
                or ""
            )

            if content:
                return str(
                    content
                )

        return (
            "I could not generate an answer "
            "from the retrieved evidence."
        )

    # ========================================================
    # TRACE
    # ========================================================

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

    # ========================================================
    # CONVERSATION
    # ========================================================

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

    # ========================================================
    # CREATE STATE
    # ========================================================

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

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(
        self,
        query,
    ):

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

        while (
            state["iterations"]
            < self.max_iterations
        ):

            state["iterations"] += 1

            iteration = state[
                "iterations"
            ]

            # ------------------------------------------------
            # PLAN
            # ------------------------------------------------

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

            # ------------------------------------------------
            # EXECUTE
            # ------------------------------------------------

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

            # ------------------------------------------------
            # EVALUATE
            # ------------------------------------------------

            evaluation = (
                self._evaluate_evidence(
                    query=query,
                    results=results,
                    action=action,
                )
            )

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

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if (
                evaluation["sufficient"]
                and results
            ):
                break

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
                    self._fallback_action(
                        action
                    )
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

            working_query = query

        # ====================================================
        # FINAL EVIDENCE
        # ====================================================

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

            return {
                "answer": state["answer"],
                "sources": [],
                "action": state["action"],
                "success": False,
                "iterations": state[
                    "iterations"
                ],
                "trace": state[
                    "trace"
                ],
            }

        # ====================================================
        # SOURCE TYPE
        # ====================================================

        source_type = (
            self._determine_source_type(
                final_results
            )
        )

        # ====================================================
        # ANSWER
        # ====================================================

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


# ============================================================
# VECTOR DATABASE LOADING
# ============================================================

def load_vector_database():
    """
    Try to load the project's existing vector database.

    This function intentionally checks common implementations
    rather than replacing your existing vector database.
    """

    candidates = [
        ("vector_db", "VectorDatabase"),
        ("vector_database", "VectorDatabase"),
        ("database", "VectorDatabase"),
        ("vectordb", "VectorDatabase"),
    ]

    for module_name, class_name in candidates:

        try:

            module = __import__(
                module_name,
                fromlist=[class_name],
            )

            cls = getattr(
                module,
                class_name,
                None,
            )

            if cls is None:
                continue

            try:
                db = cls()
            except TypeError:
                continue

            return db

        except Exception:
            continue

    return None


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_uploaded_document(
    uploaded_file,
):
    """
    Generic document representation.

    Existing project-specific processing should remain
    responsible for creating embeddings/tables.

    This fallback makes uploaded files visible/searchable
    where possible without changing the original features.
    """

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    data = {
        "source": uploaded_file.name,
        "filename": uploaded_file.name,
        "file_name": uploaded_file.name,
        "metadata": {
            "uploaded_file_name": uploaded_file.name,
            "filename": uploaded_file.name,
        },
        "text": "",
        "content": "",
        "tables": [],
    }

    # --------------------------------------------------------
    # TXT / MD
    # --------------------------------------------------------

    if suffix in {
        ".txt",
        ".md",
        ".csv",
    }:

        try:

            raw = uploaded_file.getvalue()

            text = raw.decode(
                "utf-8",
                errors="ignore",
            )

            data["text"] = text
            data["content"] = text

        except Exception:
            pass

    return data


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    if "llm" not in st.session_state:
        st.session_state.llm = None

    if "agent" not in st.session_state:
        st.session_state.agent = None


initialize_session_state()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📄 Enterprise Document Intelligence"
)

st.caption(
    "Agentic RAG • Document Search • Table Search • Web Search"
)


# ============================================================
# INITIALIZE LLM
# ============================================================

if st.session_state.llm is None:

    try:

        st.session_state.llm = (
            GeminiModel()
        )

    except Exception as exc:

        st.error(
            f"LLM initialization failed: {exc}"
        )

        st.info(
            "Check OPENROUTER_API_KEY and LLM_MODEL in your config.py/.env."
        )

        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Configuration"
    )

    top_k = st.slider(
        "Results per search",
        min_value=1,
        max_value=10,
        value=5,
    )

    max_iterations = st.slider(
        "Maximum agent iterations",
        min_value=1,
        max_value=6,
        value=4,
    )

    st.divider()

    st.header(
        "📁 Documents"
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "txt",
            "md",
            "csv",
            "docx",
            "xlsx",
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:

        existing_names = {
            item.get("filename")
            for item in st.session_state.documents
            if isinstance(
                item,
                dict,
            )
        }

        for uploaded_file in uploaded_files:

            if uploaded_file.name in existing_names:
                continue

            document = (
                extract_uploaded_document(
                    uploaded_file
                )
            )

            st.session_state.documents.append(
                document
            )

            st.session_state.uploaded_files.append(
                uploaded_file.name
            )

    if st.session_state.documents:

        st.success(
            f"{len(st.session_state.documents)} document(s) loaded."
        )

        source_names = [
            doc.get(
                "filename",
                doc.get(
                    "source",
                    "Unknown",
                ),
            )
            for doc in st.session_state.documents
            if isinstance(
                doc,
                dict,
            )
        ]

        selected_sources = st.multiselect(
            "Search only selected sources",
            options=source_names,
            default=source_names,
        )

    else:

        selected_sources = []

        st.info(
            "Upload documents to enable document retrieval."
        )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "🔄 Reset documents",
        use_container_width=True,
    ):

        st.session_state.documents = []
        st.session_state.uploaded_files = []

        st.rerun()


# ============================================================
# VECTOR DATABASE
# ============================================================

if st.session_state.vector_db is None:

    st.session_state.vector_db = (
        load_vector_database()
    )


# ============================================================
# CREATE AGENT
# ============================================================

st.session_state.agent = (
    EnterpriseRAGAgent(
        vector_db=(
            st.session_state.vector_db
        ),
        llm=(
            st.session_state.llm
        ),
        documents=(
            st.session_state.documents
        ),
        selected_sources=(
            selected_sources
        ),
        conversation_history=(
            st.session_state.messages
        ),
        top_k=top_k,
        max_iterations=max_iterations,
    )
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role"
    )

    content = message.get(
        "content",
        "",
    )

    if role not in {
        "user",
        "assistant",
    }:
        continue

    with st.chat_message(
        role
    ):
        st.markdown(
            content
        )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about your documents..."
)


if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            query
        )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Agent is searching and evaluating evidence..."
        ):

            result = (
                st.session_state.agent.run(
                    query
                )
            )

        answer = result.get(
            "answer",
            "No answer generated.",
        )

        st.markdown(
            answer
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.metric(
                "Retrieval tool",
                result.get(
                    "action"
                )
                or "N/A",
            )

        with col2:

            st.metric(
                "Iterations",
                result.get(
                    "iterations",
                    0,
                ),
            )

        with col3:

            st.metric(
                "Evidence",
                len(
                    result.get(
                        "sources",
                        [],
                    )
                ),
            )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )

        if sources:

            with st.expander(
                "📚 Retrieved Sources"
            ):

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    if not isinstance(
                        source,
                        dict,
                    ):
                        continue

                    source_name = (
                        source.get(
                            "source"
                        )
                        or source.get(
                            "filename"
                        )
                        or source.get(
                            "file_name"
                        )
                        or source.get(
                            "url"
                        )
                        or "Unknown source"
                    )

                    search_type = source.get(
                        "search_type",
                        "retrieval",
                    )

                    score = source.get(
                        "similarity_score",
                        source.get(
                            "score",
                            "",
                        ),
                    )

                    st.markdown(
                        f"**{index}. {source_name}**"
                    )

                    st.caption(
                        f"Search type: {search_type}"
                    )

                    if score != "":
                        st.caption(
                            f"Score: {score}"
                        )

                    content = (
                        source.get(
                            "content"
                        )
                        or source.get(
                            "text"
                        )
                        or ""
                    )

                    if content:

                        st.text(
                            str(
                                content
                            )[:2000]
                        )

                    st.divider()

        # ----------------------------------------------------
        # AGENT TRACE
        # ----------------------------------------------------

        trace = result.get(
            "trace",
            [],
        )

        if trace:

            with st.expander(
                "🔎 Agent Execution Trace"
            ):

                for step in trace:

                    st.markdown(
                        f"**Step {step.get('step')} — "
                        f"{step.get('action')}**"
                    )

                    details = step.get(
                        "details",
                        {},
                    )

                    if details:

                        st.json(
                            details
                        )

    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        ### 👋 Welcome

        Ask questions about your enterprise documents.

        **Examples**

        - What is the main objective of this document?
        - What are the key findings?
        - What does the table show?
        - What is the total revenue?
        - Who is the author of the document?
        - Compare the information across the uploaded documents.
        - Search the web for the latest information about this topic.
        """
    )

    if not st.session_state.documents:

        st.info(
            "👈 Upload a document from the sidebar to get started."
        )