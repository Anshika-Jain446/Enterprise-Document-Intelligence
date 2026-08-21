import json
from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    LLM_MODEL,
)


class GeminiLLM:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    # ========================================================
    # GENERIC LLM CALL
    # ========================================================

    def _call_llm(self, messages):

        response = (
            self.client
            .chat
            .completions
            .create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0
            )
        )

        if not response.choices:
            raise RuntimeError(
                "LLM returned no choices."
            )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "LLM returned empty content."
            )

        return content.strip()

    # ========================================================
    # CLEAN JSON
    # ========================================================

    def _clean_json(self, raw):

        raw = raw.strip()

        if raw.startswith("```json"):
            raw = raw[7:]

        elif raw.startswith("```"):
            raw = raw[3:]

        if raw.endswith("```"):
            raw = raw[:-3]

        return raw.strip()

    # ========================================================
    # REAL WEB SEARCH
    # ========================================================

    def web_search(
        self,
        query,
        max_results=5
    ):

        if not query or not query.strip():
            return []

        try:

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=LLM_MODEL,

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are the web research "
                                "tool of an Agentic RAG system. "
                                "Use the web search tool to retrieve "
                                "real public internet evidence. "
                                "Do not answer from memory when "
                                "web evidence is required."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                "Search the public web for:\n\n"
                                f"{query}\n\n"
                                "Retrieve reliable sources."
                            )
                        }
                    ],

                    tools=[
                        {
                            "type": "openrouter:web_search",
                            "parameters": {
                                "engine": "auto",
                                "max_results": max_results,
                                "max_total_results": max_results * 2,
                                "search_context_size": "medium"
                            }
                        }
                    ],

                    max_tokens=1106,
                    temperature=0
                )
            )

        except Exception as e:

            print(
                f"REAL WEB SEARCH FAILED: {e}"
            )

            return []

        if not response.choices:
            return []

        message = response.choices[0].message

        content = (
            getattr(
                message,
                "content",
                None
            )
            or ""
        )

        results = []

        annotations = getattr(
            message,
            "annotations",
            None
        )

        # ----------------------------------------------------
        # PARSE WEB CITATIONS
        # ----------------------------------------------------

        if annotations:

            for annotation in annotations:

                try:

                    if isinstance(
                        annotation,
                        dict
                    ):

                        citation = annotation.get(
                            "url_citation",
                            {}
                        )

                    else:

                        citation = getattr(
                            annotation,
                            "url_citation",
                            None
                        )

                    if not citation:
                        continue

                    if isinstance(
                        citation,
                        dict
                    ):

                        url = citation.get(
                            "url",
                            ""
                        )

                        title = citation.get(
                            "title",
                            ""
                        )

                        citation_content = citation.get(
                            "content",
                            ""
                        )

                    else:

                        url = getattr(
                            citation,
                            "url",
                            ""
                        )

                        title = getattr(
                            citation,
                            "title",
                            ""
                        )

                        citation_content = getattr(
                            citation,
                            "content",
                            ""
                        )

                    if not url:
                        continue

                    results.append(
                        {
                            "title": (
                                title
                                or url
                            ),

                            "url": url,

                            "content": (
                                citation_content
                                or content
                            ),

                            "source": url,

                            "source_type": "web",

                            "search_type": "web_search"
                        }
                    )

                except Exception as e:

                    print(
                        "Web citation parsing failed:",
                        e
                    )

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if (
            not results
            and content.strip()
        ):

            results.append(
                {
                    "title": "Web Search Result",

                    "url": "",

                    "content": content,

                    "source": "Web Search",

                    "source_type": "web",

                    "search_type": "web_search"
                }
            )

        return results

    # ========================================================
    # DETERMINISTIC ROUTING
    # ========================================================

    def _deterministic_action(
        self,
        query
    ):

        q = query.lower().strip()

        # ====================================================
        # EXPLICIT DOCUMENT REFERENCES
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
        ]

        if any(
            pattern in q
            for pattern in document_patterns
        ):
            return "vector_search"

        # ====================================================
        # METADATA
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
        ]

        if any(
            term in q
            for term in metadata_terms
        ):
            return "document_search"

        # ====================================================
        # TABLES
        # ====================================================

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

        # ====================================================
        # PUBLIC / CURRENT INFORMATION
        # ====================================================

        web_patterns = [
            "who is ",
            "who was ",
            "who are ",
            "where is ",
            "where was ",
            "when was ",
            "when did ",
            "latest ",
            "current ",
            "today ",
            "news ",
            "weather ",
            "stock price",
            "share price",
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

        # Let the LLM planner decide for
        # questions that don't match a
        # deterministic rule.

        return None

    # ========================================================
    # AGENT PLANNER
    # ========================================================

    def plan_action(
        self,
        query,
        previous_actions=None,
        previous_evaluations=None
    ):

        previous_actions = (
            previous_actions or []
        )

        previous_evaluations = (
            previous_evaluations or []
        )

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
                    "Selected using "
                    "deterministic routing."
                )
            }

        planning_prompt = f"""
You are the planner of a REAL Agentic RAG system.

You have FOUR tools.

1. vector_search

Search uploaded document CONTENT.

2. document_search

Search uploaded document METADATA.

3. table_search

Search uploaded TABLES.

4. web_search

Perform REAL PUBLIC INTERNET SEARCH.

---

## ROUTING RULES

Uploaded document content:
vector_search

Document metadata:
document_search

Tables / spreadsheets:
table_search

Public person:
web_search

General knowledge:
web_search

Current/latest information:
web_search

News:
web_search

Weather:
web_search

Public companies:
web_search

Questions unrelated to uploaded documents:
web_search

IMPORTANT:

Do NOT use vector_search for general knowledge.

Do NOT assume uploaded documents contain
public knowledge.

If the user explicitly refers to the uploaded
document, use document tools.

Otherwise, public/general questions should
use web_search.

---

## AGENTIC BEHAVIOR

Look at previous actions and evaluations.

If the previous retrieval failed, choose a
different appropriate tool.

Do not repeatedly select the same failed tool
unless there is a good reason.

Return ONLY JSON.

Required:

{{
    "action": "web_search",
    "query": "optimized search query",
    "reason": "why this tool should be used"
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
    default=str
)}

PREVIOUS EVALUATIONS:

{json.dumps(
    previous_evaluations,
    indent=2,
    default=str
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
                            "Return JSON only."
                        )
                    },
                    {
                        "role": "user",
                        "content": planning_prompt
                    }
                ]
            )

            raw = self._clean_json(raw)

            plan = json.loads(raw)

            allowed_actions = {
                "vector_search",
                "document_search",
                "table_search",
                "web_search"
            }

            action = plan.get(
                "action",
                "web_search"
            )

            if action not in allowed_actions:
                action = "web_search"

            return {
                "action": action,

                "query": plan.get(
                    "query",
                    query
                ),

                "reason": plan.get(
                    "reason",
                    "Selected retrieval strategy."
                )
            }

        except Exception as e:

            print(
                f"Agent planning failed: {e}"
            )

            # Safer fallback for non-document
            # general questions.

            return {
                "action": "web_search",

                "query": query,

                "reason": (
                    "Planner failed; "
                    "using web search fallback."
                )
            }

    # ========================================================
    # EVIDENCE EVALUATOR
    # ========================================================

    def evaluate_evidence(
        self,
        query,
        results,
        action=None
    ):

        if not results:

            # IMPORTANT:
            # A failed web search should remain
            # a failed retrieval, not become success.

            return {
                "sufficient": False,

                "confidence": 0.0,

                "reason": (
                    "No usable retrieval evidence."
                ),

                "recommended_action": (
                    "web_search"
                    if action == "web_search"
                    else action or "vector_search"
                )
            }

        evidence = []

        for result in results:

            if not isinstance(
                result,
                dict
            ):
                continue

            evidence.append(
                {
                    "title": str(
                        result.get(
                            "title",
                            ""
                        )
                    ),

                    "url": str(
                        result.get(
                            "url",
                            ""
                        )
                    ),

                    "content": str(
                        result.get(
                            "content",
                            ""
                        )
                    )[:4000],

                    "source": str(
                        result.get(
                            "source",
                            "Unknown"
                        )
                    ),

                    "score": result.get(
                        "similarity_score",
                        result.get(
                            "score",
                            0
                        )
                    )
                }
            )

        if not evidence:

            return {
                "sufficient": False,
                "confidence": 0.0,
                "reason": (
                    "No usable evidence."
                ),
                "recommended_action": (
                    action or "web_search"
                )
            }

        # ----------------------------------------------------
        # ALWAYS LLM-EVALUATE WEB EVIDENCE
        # ----------------------------------------------------

        evaluation_prompt = f"""
You are the evidence evaluator of a REAL Agentic RAG system.

USER QUESTION:

{query}

RETRIEVAL TOOL:

{action}

EVIDENCE:

{json.dumps(
    evidence,
    indent=2,
    default=str
)}

Determine whether the evidence actually answers
the user's question.

Return ONLY JSON:

{{
    "sufficient": true,
    "confidence": 0.95,
    "reason": "Evidence directly answers the question.",
    "recommended_action": "web_search"
}}

Rules:

- If evidence directly supports the answer:
  sufficient=true.

- If evidence is irrelevant:
  sufficient=false.

- If evidence is incomplete:
  sufficient=false.

- If web evidence is insufficient:
  recommend web_search.

- If document content is insufficient:
  recommend vector_search.

- If metadata is insufficient:
  recommend document_search.

- If table evidence is insufficient:
  recommend table_search.

- Do not invent facts.
"""

        try:

            raw = self._call_llm(
                [
                    {
                        "role": "system",
                        "content": (
                            "You evaluate retrieval "
                            "quality. Return JSON only."
                        )
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ]
            )

            evaluation = json.loads(
                self._clean_json(raw)
            )

            confidence = float(
                evaluation.get(
                    "confidence",
                    0.0
                )
            )

            confidence = max(
                0.0,
                min(
                    confidence,
                    1.0
                )
            )

            allowed_actions = {
                "vector_search",
                "document_search",
                "table_search",
                "web_search"
            }

            recommended_action = (
                evaluation.get(
                    "recommended_action",
                    action or "web_search"
                )
            )

            if (
                recommended_action
                not in allowed_actions
            ):

                recommended_action = (
                    action or "web_search"
                )

            return {
                "sufficient": bool(
                    evaluation.get(
                        "sufficient",
                        False
                    )
                ),

                "confidence": confidence,

                "reason": evaluation.get(
                    "reason",
                    ""
                ),

                "recommended_action":
                    recommended_action
            }

        except Exception as e:

            print(
                f"Evidence evaluation failed: {e}"
            )

            # Conservative fallback.

            return {
                "sufficient": False,

                "confidence": 0.0,

                "reason": (
                    "Evidence evaluator failed."
                ),

                "recommended_action": (
                    action or "web_search"
                )
            }

    # ========================================================
    # ANSWER GENERATION
    # ========================================================

    def generate_answer(
        self,
        query,
        chunks,
        source_type="document"
    ):

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

        context_parts = []

        sources = set()

        web_sources = []

        for chunk in chunks:

            if not isinstance(
                chunk,
                dict
            ):
                continue

            content = chunk.get(
                "content",
                ""
            )

            if content:

                context_parts.append(
                    str(content)
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

        context = "\n\n".join(
            context_parts
        )

        if not context.strip():

            return (
                "I could not find sufficient "
                "evidence to answer that question."
            )

        sources_text = (
            ", ".join(
                sorted(sources)
            )
            if sources
            else "Retrieved sources"
        )

        if source_type == "web":

            prompt = f"""
You are the final answer generator of a
REAL Agentic RAG system.

The user asked:

{query}

The agent performed a REAL PUBLIC WEB SEARCH.

Use ONLY the retrieved web evidence below.

WEB EVIDENCE:

{context}

RULES:

1. Answer directly.
2. Use only retrieved evidence.
3. Do not invent facts.
4. If sources disagree, explain it.
5. Prefer authoritative sources.
6. Clearly state that the answer is based on web research.
7. Include source URLs when available.
8. Do not claim uploaded documents contain this information.

WEB SOURCES:

{chr(10).join(web_sources)}
"""

        else:

            prompt = f"""
You are the final answer generator of an
Enterprise Document RAG system.

USER QUESTION:

{query}

DOCUMENT EVIDENCE:

{context}

DOCUMENTS USED:

{sources_text}

RULES:

1. Use ONLY uploaded-document evidence.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Answer directly.
5. If evidence is insufficient, say:

"I could not find that information in the uploaded documents."

6. Mention the document used.
"""

        try:

            return self._call_llm(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a grounded "
                            "Agentic RAG answer generator."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        except Exception as e:

            print(
                f"Answer generation failed: {e}"
            )

            return (
                "I could not generate an answer "
                "from the retrieved evidence."
            )