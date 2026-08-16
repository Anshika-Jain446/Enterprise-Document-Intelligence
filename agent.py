import json


class EnterpriseRAGAgent:

    def __init__(
        self,
        vector_db,
        llm,
        documents=None,
        top_k=5,
        max_iterations=4
    ):
        self.vector_db = vector_db
        self.llm = llm
        self.documents = documents or []

        self.top_k = top_k
        self.max_iterations = max_iterations

        self.allowed_actions = {
            "vector_search",
            "document_search",
            "table_search",
            "web_search"
        }

    # ========================================================
    # STATE
    # ========================================================

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
            "table_used": False
        }

    # ========================================================
    # TRACE
    # ========================================================

    def _trace(self, state, step, action, details=None):

        state["trace"].append({
            "step": step,
            "action": action,
            "details": details or {}
        })

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def _vector_search(self, query):

        if not query or not query.strip():
            return []

        if self.vector_db is None:
            return []

        try:

            if self.vector_db.index is None:

                if self.vector_db.exists():
                    self.vector_db.load()
                else:
                    return []

            if self.vector_db.index is None:
                return []

            results = self.vector_db.search(
                query,
                top_k=self.top_k
            )

            normalized = []

            for result in results:

                if not isinstance(result, dict):
                    continue

                item = dict(result)

                item["source_type"] = "document"
                item["search_type"] = "vector_search"

                normalized.append(item)

            return normalized

        except Exception as e:

            print(f"Vector search failed: {e}")
            return []

    # ========================================================
    # DOCUMENT SEARCH
    # ========================================================

    def _document_search(self, query):

        if not query or not self.documents:
            return []

        words = {
            word.lower()
            for word in query.split()
            if len(word) > 2
        }

        results = []

        for document in self.documents:

            if not isinstance(document, dict):
                continue

            metadata = document.get("metadata", {})
            text = str(document.get("text", ""))

            searchable = (
                text
                + " "
                + json.dumps(
                    metadata,
                    ensure_ascii=False
                )
            ).lower()

            score = sum(
                1
                for word in words
                if word in searchable
            )

            if score == 0:
                continue

            result = dict(document)

            result["similarity_score"] = (
                score / len(words)
                if words
                else 0
            )

            result["source_type"] = "document"
            result["search_type"] = "document_search"

            results.append((score, result))

        results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            result
            for _, result in results[:self.top_k]
        ]

    # ========================================================
    # TABLE SEARCH
    # ========================================================

    def _table_search(self, query):

        if not query or not self.documents:
            return []

        words = {
            word.lower()
            for word in query.split()
            if len(word) > 2
        }

        results = []

        for document in self.documents:

            tables = document.get(
                "tables",
                []
            )

            if not isinstance(tables, list):
                continue

            for table in tables:

                if not isinstance(table, dict):
                    continue

                table_data = table.get(
                    "table",
                    []
                )

                searchable = json.dumps(
                    table_data,
                    ensure_ascii=False
                ).lower()

                score = sum(
                    1
                    for word in words
                    if word in searchable
                )

                if score == 0:
                    continue

                result = dict(table)

                result["similarity_score"] = (
                    score / len(words)
                    if words
                    else 0
                )

                result["source_type"] = "table"
                result["search_type"] = "table_search"

                results.append((score, result))

        results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            result
            for _, result in results[:self.top_k]
        ]

    # ========================================================
    # TOOL EXECUTION
    # ========================================================

    def execute_tool(self, action, query):

        print(f"Executing tool: {action}")

        if action == "vector_search":
            return self._vector_search(query)

        if action == "document_search":
            return self._document_search(query)

        if action == "table_search":
            return self._table_search(query)

        if action == "web_search":

            try:

                return self.llm.web_search(
                    query=query,
                    max_results=self.top_k
                )

            except Exception as e:

                print(f"Web search failed: {e}")
                return []

        return []

    # ========================================================
    # NORMALIZE
    # ========================================================

    def _normalize_results(self, results, action):

        if not isinstance(results, list):
            return []

        normalized = []

        for result in results:

            if not isinstance(result, dict):
                continue

            item = dict(result)

            item.setdefault(
                "search_type",
                action
            )

            item.setdefault(
                "source_type",
                "web"
                if action == "web_search"
                else "document"
            )

            normalized.append(item)

        return normalized

    # ========================================================
    # AGENT LOOP
    # ========================================================

    def run(self, query):

        if not query or not query.strip():

            return {
                "answer": "Please enter a question.",
                "sources": [],
                "action": None,
                "success": False,
                "iterations": 0,
                "trace": []
            }

        state = self._create_state(query)

        working_query = query
        forced_action = None

        print("\n" + "=" * 70)
        print("ENTERPRISE AGENTIC RAG")
        print("=" * 70)

        while state["iterations"] < self.max_iterations:

            state["iterations"] += 1

            iteration = state["iterations"]

            # =================================================
            # PLAN
            # =================================================

            if forced_action:

                plan = {
                    "action": forced_action,
                    "query": working_query,
                    "reason": "Replanned after insufficient evidence."
                }

                forced_action = None

            else:

                try:

                    plan = self.llm.plan_action(
                        query=working_query,
                        previous_actions=state[
                            "previous_actions"
                        ],
                        previous_evaluations=state[
                            "previous_evaluations"
                        ]
                    )

                except Exception as e:

                    print(f"Planner failed: {e}")

                    plan = {
                        "action": "vector_search",
                        "query": working_query,
                        "reason": "Planner failure fallback."
                    }

            action = plan.get(
                "action",
                "vector_search"
            )

            search_query = plan.get(
                "query",
                working_query
            )

            if action not in self.allowed_actions:
                action = "vector_search"

            state["action"] = action

            state["previous_actions"].append(action)

            # =================================================
            # SOURCE TRACKING
            # =================================================

            if action == "web_search":
                state["web_used"] = True

            elif action == "vector_search":
                state["document_used"] = True

            elif action == "document_search":
                state["document_used"] = True

            elif action == "table_search":
                state["table_used"] = True

            # =================================================
            # TRACE PLAN
            # =================================================

            self._trace(
                state,
                len(state["trace"]) + 1,
                "agent_plan",
                {
                    "iteration": iteration,
                    "tool": action,
                    "query": search_query,
                    "reason": plan.get("reason", "")
                }
            )

            print(
                f"\nIteration {iteration}"
            )

            print(
                f"Agent selected: {action}"
            )

            # =================================================
            # EXECUTE
            # =================================================

            try:

                results = self.execute_tool(
                    action,
                    search_query
                )

            except Exception as e:

                print(
                    f"Tool execution failed: {e}"
                )

                results = []

            results = self._normalize_results(
                results,
                action
            )

            state["results"] = results

            state["all_results"].extend(
                results
            )

            self._trace(
                state,
                len(state["trace"]) + 1,
                "tool_result",
                {
                    "tool": action,
                    "results": len(results)
                }
            )

            print(
                f"Retrieved: {len(results)} results"
            )

            # =================================================
            # EVALUATE
            # =================================================

            try:

                evaluation = self.llm.evaluate_evidence(
                    query=state["original_query"],
                    results=results,
                    action=action
                )

            except Exception as e:

                print(
                    f"Evaluator failed: {e}"
                )

                evaluation = {
                    "sufficient": False,
                    "confidence": 0.0,
                    "reason": str(e),
                    "recommended_action": (
                        "web_search"
                    )
                }

            state[
                "previous_evaluations"
            ].append(evaluation)

            sufficient = bool(
                evaluation.get(
                    "sufficient",
                    False
                )
            )

            confidence = float(
                evaluation.get(
                    "confidence",
                    0
                )
            )

            self._trace(
                state,
                len(state["trace"]) + 1,
                "evidence_evaluation",
                evaluation
            )

            print(
                f"Evidence sufficient: {sufficient}"
            )

            print(
                f"Confidence: {confidence:.2f}"
            )

            # =================================================
            # SUCCESS
            # =================================================

            if sufficient and results:
                break

            # =================================================
            # REPLAN
            # =================================================

            recommended = evaluation.get(
                "recommended_action"
            )

            if recommended not in self.allowed_actions:

                recommended = (
                    "web_search"
                    if action != "web_search"
                    else "vector_search"
                )

            working_query = search_query

            if iteration < self.max_iterations:

                forced_action = recommended

                self._trace(
                    state,
                    len(state["trace"]) + 1,
                    "replan",
                    {
                        "next_action": recommended
                    }
                )

                print(
                    f"Replanning → {recommended}"
                )

        # =====================================================
        # FINAL EVIDENCE
        # =====================================================

        if not state["results"]:

            return {
                "answer": (
                    "I could not find sufficient "
                    "evidence to answer that question."
                ),
                "sources": state["all_results"],
                "action": state["action"],
                "success": False,
                "iterations": state["iterations"],
                "trace": state["trace"]
            }

        # =====================================================
        # SOURCE TYPE
        # =====================================================

        if state["web_used"]:
            source_type = "web"

        elif state["table_used"]:
            source_type = "table"

        else:
            source_type = "document"

        # =====================================================
        # FINAL ANSWER
        # =====================================================

        try:

            answer = self.llm.generate_answer(
                query=state["original_query"],
                chunks=state["all_results"],
                source_type=source_type
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

        state["answer"] = answer
        state["success"] = True

        self._trace(
            state,
            len(state["trace"]) + 1,
            "answer_generated",
            {
                "source_type": source_type,
                "evidence_count": len(
                    state["all_results"]
                )
            }
        )

        return {
            "answer": state["answer"],
            "sources": state["all_results"],
            "action": state["action"],
            "success": True,
            "iterations": state["iterations"],
            "trace": state["trace"]
        }