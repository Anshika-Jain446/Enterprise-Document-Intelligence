import json


# ========================================================
# DETERMINISTIC ROUTING
# ========================================================

def _deterministic_action(
    self,
    query,
):

    q = str(query or "").lower().strip()

    if not q:
        return None

    # ====================================================
    # EXPLICIT UPLOADED DOCUMENT REFERENCES
    # ====================================================

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
        "my document",
        "my file",
        "my uploaded",
    ]

    if any(
        pattern in q
        for pattern in document_patterns
    ):
        return "vector_search"

    # ====================================================
    # DOCUMENT CONTENT QUESTIONS
    # ====================================================

    content_patterns = [
        "what does the document say",
        "what does the paper say",
        "explain the document",
        "summarize the document",
        "summarise the document",
        "according to the pdf",
        "according to the file",
        "according to my file",
        "according to my document",
        "find in the document",
        "find in the file",
        "mentioned in the document",
        "mentioned in the file",
        "contained in the document",
        "contained in the file",
    ]

    if any(
        pattern in q
        for pattern in content_patterns
    ):
        return "vector_search"

    # ====================================================
    # DOCUMENT METADATA
    # ====================================================

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
        "pages does the document have",
    ]

    if any(
        term in q
        for term in metadata_terms
    ):
        return "document_search"

    # ====================================================
    # TABLE / SPREADSHEET QUESTIONS
    # ====================================================

    table_terms = [
        "table",
        "tables",
        "row",
        "rows",
        "column",
        "columns",
        "spreadsheet",
        "excel",
        "xlsx",
        "csv",
        "cell",
        "cells",
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

    # ====================================================
    # PUBLIC / CURRENT INFORMATION
    # ====================================================

    web_patterns = [
        "latest",
        "current",
        "today",
        "news",
        "weather",
        "stock price",
        "share price",
        "exchange rate",
        "current price",
        "recent news",
        "who is",
        "who was",
        "who are",
        "where is",
        "where was",
        "when was",
        "when did",
        "president",
        "prime minister",
        "actor",
        "actress",
        "celebrity",
        "footballer",
        "cricketer",
        "athlete",
        "elon musk",
        "shah rukh khan",
        "shahrukh khan",
        "srk",
    ]

    if any(
        pattern in q
        for pattern in web_patterns
    ):
        return "web_search"

    return None


# ========================================================
# AGENT PLANNER
# ========================================================

def plan_action(
    self,
    query,
    previous_actions=None,
    previous_evaluations=None,
):

    previous_actions = (
        previous_actions or []
    )

    previous_evaluations = (
        previous_evaluations or []
    )

    query = str(query or "").strip()

    if not query:

        return {
            "action": "web_search",
            "query": "",
            "reason": "Empty query; using safe fallback.",
        }

    # ====================================================
    # FIRST: DETERMINISTIC ROUTING
    # ====================================================

    deterministic_action = (
        self._deterministic_action(
            query
        )
    )

    if deterministic_action:

        return {
            "action": deterministic_action,

            "query": query,

            "reason": (
                "Selected using deterministic "
                "routing rules."
            ),
        }

    # ====================================================
    # CHECK PREVIOUS FAILED RETRIEVAL
    # ====================================================

    failed_actions = []

    for evaluation in previous_evaluations:

        if not isinstance(
            evaluation,
            dict,
        ):
            continue

        if not evaluation.get(
            "sufficient",
            False,
        ):

            recommended = evaluation.get(
                "recommended_action"
            )

            if recommended:
                failed_actions.append(
                    recommended
                )

    planning_prompt = f"""
You are the planner of a REAL Agentic RAG system.

Available retrieval tools:

1. vector_search
   Searches uploaded document CONTENT.

2. document_search
   Searches uploaded document METADATA.

3. table_search
   Searches uploaded TABLES and spreadsheets.

4. web_search
   Searches PUBLIC INTERNET information.

ROUTING RULES:

- Questions explicitly about uploaded documents:
  vector_search

- Questions asking for document title, filename,
  author, page count, metadata:
  document_search

- Questions about tables, rows, columns,
  spreadsheets, Excel, CSV:
  table_search

- Public people, companies, current information,
  latest information, news, weather:
  web_search

- General knowledge questions:
  web_search

IMPORTANT:

Do NOT use vector_search for general knowledge.

Do NOT assume uploaded documents contain
general public knowledge.

Do NOT use document_search for document CONTENT.

Do NOT use table_search unless the question
actually requires table/spreadsheet data.

Look at previous retrieval attempts.

If the previous retrieval failed, select a
different appropriate retrieval strategy when
possible.

Return ONLY valid JSON.

Required format:

{{
    "action": "web_search",
    "query": "optimized search query",
    "reason": "why this retrieval tool should be used"
}}

Allowed actions:

vector_search
document_search
table_search
web_search

USER QUESTION:

{query}

PREVIOUS ACTIONS:

{json.dumps(
    previous_actions,
    indent=2,
    default=str,
)}

PREVIOUS EVALUATIONS:

{json.dumps(
    previous_evaluations,
    indent=2,
    default=str,
)}

FAILED / RECOMMENDED ACTIONS:

{json.dumps(
    failed_actions,
    indent=2,
    default=str,
)}

Do not answer the user.

Only plan retrieval.
"""

    try:

        raw = self._call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a precise "
                        "Agentic RAG planner. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": planning_prompt,
                },
            ]
        )

        cleaned = self._clean_json(
            raw
        )

        plan = json.loads(
            cleaned
        )

        allowed_actions = {
            "vector_search",
            "document_search",
            "table_search",
            "web_search",
        }

        action = str(
            plan.get(
                "action",
                "web_search",
            )
        ).strip()

        if action not in allowed_actions:
            action = "web_search"

        optimized_query = str(
            plan.get(
                "query",
                query,
            )
            or query
        ).strip()

        reason = str(
            plan.get(
                "reason",
                "Selected retrieval strategy.",
            )
            or "Selected retrieval strategy."
        ).strip()

        return {
            "action": action,
            "query": optimized_query,
            "reason": reason,
        }

    except Exception as e:

        print(
            f"Agent planning failed: {e}"
        )

        return {
            "action": "web_search",

            "query": query,

            "reason": (
                "Planner failed; using "
                "web search fallback."
            ),
        }


# ========================================================
# EVIDENCE EVALUATOR
# ========================================================

def evaluate_evidence(
    self,
    query,
    results,
    action=None,
):

    action = (
        action
        or "web_search"
    )

    # ====================================================
    # NO RESULTS
    # ====================================================

    if not results:

        if action == "web_search":

            recommended_action = "web_search"

        elif action == "vector_search":

            recommended_action = "vector_search"

        elif action == "document_search":

            recommended_action = "document_search"

        elif action == "table_search":

            recommended_action = "table_search"

        else:

            recommended_action = "web_search"

        return {
            "sufficient": False,

            "confidence": 0.0,

            "reason": (
                "No usable retrieval evidence."
            ),

            "recommended_action":
                recommended_action,
        }

    # ====================================================
    # NORMALIZE EVIDENCE
    # ====================================================

    evidence = []

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        content = str(
            result.get(
                "content",
                "",
            )
            or ""
        ).strip()

        title = str(
            result.get(
                "title",
                "",
            )
            or ""
        ).strip()

        url = str(
            result.get(
                "url",
                "",
            )
            or ""
        ).strip()

        source = str(
            result.get(
                "source",
                "Unknown",
            )
            or "Unknown"
        ).strip()

        score = result.get(
            "similarity_score",
            result.get(
                "score",
                0,
            ),
        )

        if not content and not title:
            continue

        evidence.append(
            {
                "title": title,
                "url": url,
                "content": content[:4000],
                "source": source,
                "score": score,
            }
        )

    if not evidence:

        return {
            "sufficient": False,

            "confidence": 0.0,

            "reason": (
                "No usable evidence."
            ),

            "recommended_action":
                action,
        }

    # ====================================================
    # EVALUATION PROMPT
    # ====================================================

    evaluation_prompt = f"""
You are the evidence evaluator of a
REAL Agentic RAG system.

USER QUESTION:

{query}

RETRIEVAL TOOL:

{action}

RETRIEVED EVIDENCE:

{json.dumps(
    evidence,
    indent=2,
    default=str,
)}

Determine whether the retrieved evidence
actually answers the user's question.

Return ONLY valid JSON:

{{
    "sufficient": true,
    "confidence": 0.95,
    "reason": "Evidence directly answers the question.",
    "recommended_action": "web_search"
}}

RULES:

- If evidence directly supports the answer:
  sufficient=true.

- If evidence is irrelevant:
  sufficient=false.

- If evidence is incomplete:
  sufficient=false.

- Do not invent facts.

- If web evidence is insufficient:
  recommend web_search.

- If document content is insufficient:
  recommend vector_search.

- If metadata is insufficient:
  recommend document_search.

- If table evidence is insufficient:
  recommend table_search.

- Confidence must be between 0 and 1.

- Return JSON only.
"""

    try:

        raw = self._call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You evaluate retrieval "
                        "quality. Return valid "
                        "JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": evaluation_prompt,
                },
            ]
        )

        evaluation = json.loads(
            self._clean_json(
                raw
            )
        )

        confidence = float(
            evaluation.get(
                "confidence",
                0.0,
            )
        )

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

        allowed_actions = {
            "vector_search",
            "document_search",
            "table_search",
            "web_search",
        }

        recommended_action = str(
            evaluation.get(
                "recommended_action",
                action,
            )
            or action
        ).strip()

        if (
            recommended_action
            not in allowed_actions
        ):

            recommended_action = action

        return {
            "sufficient": sufficient,

            "confidence": confidence,

            "reason": str(
                evaluation.get(
                    "reason",
                    "",
                )
                or ""
            ).strip(),

            "recommended_action":
                recommended_action,
        }

    except Exception as e:

        print(
            f"Evidence evaluation failed: {e}"
        )

        return {
            "sufficient": False,

            "confidence": 0.0,

            "reason": (
                "Evidence evaluator failed."
            ),

            "recommended_action":
                action,
        }


# ========================================================
# ANSWER GENERATION
# ========================================================

def generate_answer(
    self,
    query,
    chunks,
    source_type="document",
):

    # ====================================================
    # NO EVIDENCE
    # ====================================================

    if not chunks:

        if source_type == "web":

            return (
                "I could not find reliable "
                "web evidence to answer that question."
            )

        return (
            "I could not find that information "
            "in the uploaded documents."
        )

    # ====================================================
    # BUILD CONTEXT
    # ====================================================

    context_parts = []

    sources = set()

    web_sources = []

    for chunk in chunks:

        if not isinstance(
            chunk,
            dict,
        ):
            continue

        content = str(
            chunk.get(
                "content",
                "",
            )
            or ""
        ).strip()

        if content:

            context_parts.append(
                content
            )

        source = chunk.get(
            "source"
        )

        if source:

            sources.add(
                str(source)
            )

        url = chunk.get(
            "url"
        )

        if url:

            web_sources.append(
                str(url)
            )

    # ====================================================
    # NO USABLE CONTEXT
    # ====================================================

    context = "\n\n".join(
        context_parts
    )

    if not context.strip():

        if source_type == "web":

            return (
                "I could not find sufficient "
                "web evidence to answer that question."
            )

        return (
            "I could not find sufficient "
            "evidence in the uploaded documents."
        )

    # ====================================================
    # SOURCE INFORMATION
    # ====================================================

    sources_text = (
        ", ".join(
            sorted(sources)
        )
        if sources
        else "Retrieved sources"
    )

    # Remove duplicate URLs while preserving order.

    unique_web_sources = list(
        dict.fromkeys(
            web_sources
        )
    )

    # ====================================================
    # WEB ANSWER
    # ====================================================

    if source_type == "web":

        prompt = f"""
You are the final answer generator of a
REAL Agentic RAG system.

The user asked:

{query}

The system performed a REAL PUBLIC WEB SEARCH.

Use ONLY the retrieved web evidence below.

WEB EVIDENCE:

{context}

RULES:

1. Answer the user's question directly.

2. Use only the retrieved evidence.

3. Do not invent facts.

4. If the evidence is incomplete,
   clearly say so.

5. If sources disagree,
   explain the disagreement.

6. Prefer authoritative evidence
   when multiple sources are available.

7. Do not claim that uploaded documents
   contain this information.

8. Do not claim that you personally
   browsed the internet.

9. Keep the answer concise and useful.

WEB SOURCES:

{chr(10).join(unique_web_sources)}
"""

    # ====================================================
    # DOCUMENT ANSWER
    # ====================================================

    else:

        prompt = f"""
You are the final answer generator of an
Enterprise Document RAG system.

USER QUESTION:

{query}

UPLOADED DOCUMENT EVIDENCE:

{context}

DOCUMENTS / SOURCES USED:

{sources_text}

RULES:

1. Use ONLY the uploaded-document evidence.

2. Do not use outside knowledge.

3. Do not invent facts.

4. Answer the question directly.

5. If the evidence is insufficient, say:

"I could not find that information in
the uploaded documents."

6. Mention the document/source when useful.

7. Do not claim information that is not
supported by the retrieved evidence.
"""

    # ====================================================
    # CALL LLM
    # ====================================================

    try:

        answer = self._call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a grounded "
                        "Agentic RAG answer generator. "
                        "Never invent evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )

        answer = str(
            answer or ""
        ).strip()

        if not answer:

            return (
                "I could not generate an answer "
                "from the retrieved evidence."
            )

        return answer

    except Exception as e:

        print(
            f"Answer generation failed: {e}"
        )

        return (
            "I could not generate an answer "
            "from the retrieved evidence."
        )