import json
from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    LLM_MODEL,
)


class GeminiModel:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

        self.model_name = LLM_MODEL

    # ========================================================
    # BASIC RESPONSE
    # ========================================================

    def _extract_content(self, response):

        if not response or not response.choices:
            raise RuntimeError(
                "OpenRouter returned no choices."
            )

        message = response.choices[0].message

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:
            raise RuntimeError(
                "OpenRouter returned empty content."
            )

        return str(content).strip()

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(self, prompt):

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )
        )

        return self._extract_content(
            response
        )

    # ========================================================
    # CHAT
    # ========================================================

    def chat(self, messages):

        if not messages:
            raise ValueError(
                "Messages cannot be empty."
            )

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model_name,
                messages=messages,
                temperature=0,
            )
        )

        return self._extract_content(
            response
        )

    # ========================================================
    # JSON GENERATION
    # ========================================================

    def _generate_json(
        self,
        system_prompt,
        user_prompt,
        fallback=None,
    ):

        try:

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model_name,

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],

                    temperature=0,

                    response_format={
                        "type": "json_object"
                    },
                )
            )

            content = self._extract_content(
                response
            )

            # ------------------------------------------------
            # Remove accidental markdown fences.
            # ------------------------------------------------

            if content.startswith(
                "```"
            ):

                content = (
                    content
                    .replace(
                        "```json",
                        "",
                    )
                    .replace(
                        "```",
                        "",
                    )
                    .strip()
                )

            return json.loads(
                content
            )

        except Exception as e:

            print(
                f"JSON generation failed: {e}"
            )

            return (
                fallback
                if fallback is not None
                else {}
            )

    # ========================================================
    # PLAN ACTION
    # ========================================================

    def plan_action(
        self,
        query,
        previous_actions=None,
        previous_evaluations=None,
    ):
        """
        Decide which retrieval tool the agent should use.

        Compatible with app.py:

            plan_action(
                query=...,
                previous_actions=...,
                previous_evaluations=...
            )
        """

        previous_actions = (
            previous_actions or []
        )

        previous_evaluations = (
            previous_evaluations or []
        )

        system_prompt = """
You are the planning component of an enterprise
document intelligence agent.

Your job is to select the BEST retrieval action
for the user's question.

Allowed actions:

1. vector_search
   - semantic search over uploaded documents
   - use for questions about document content

2. document_search
   - lexical/document metadata search
   - use for filenames, titles, authors, metadata,
     exact document properties, or exact text

3. table_search
   - search extracted tables
   - use for numerical/tabular information

4. web_search
   - current or external information
   - use when the answer requires information
     outside the uploaded documents
   - use for current/latest/news/recent information

Important:
- Do NOT invent an action.
- Return ONLY valid JSON.
- "query" must be a useful search query.
- "reason" should briefly explain the choice.

Return exactly:

{
  "action": "vector_search | document_search | table_search | web_search",
  "query": "search query",
  "reason": "brief reason"
}
"""

        user_prompt = f"""
Current user question:

{query}

Previous actions:

{json.dumps(
    previous_actions,
    ensure_ascii=False,
)}

Previous evidence evaluations:

{json.dumps(
    previous_evaluations,
    ensure_ascii=False,
)}

Choose the next retrieval action.
"""

        fallback_action = (
            "vector_search"
            if not previous_actions
            else "web_search"
        )

        fallback = {
            "action": fallback_action,
            "query": query,
            "reason": "Fallback retrieval action.",
        }

        result = self._generate_json(
            system_prompt,
            user_prompt,
            fallback=fallback,
        )

        allowed_actions = {
            "vector_search",
            "document_search",
            "table_search",
            "web_search",
        }

        action = result.get(
            "action"
        )

        if action not in allowed_actions:

            action = fallback_action

        search_query = result.get(
            "query"
        )

        if not isinstance(
            search_query,
            str,
        ) or not search_query.strip():

            search_query = query

        reason = result.get(
            "reason",
            "Selected retrieval tool.",
        )

        return {
            "action": action,
            "query": search_query.strip(),
            "reason": str(reason),
        }

    # ========================================================
    # EVALUATE EVIDENCE
    # ========================================================

    def evaluate_evidence(
        self,
        query,
        results,
        action,
    ):
        """
        Evaluate whether retrieved evidence is sufficient
        to answer the user's question.

        Compatible with app.py.
        """

        if not results:

            return {
                "sufficient": False,
                "confidence": 0.0,
                "reason": "No evidence was retrieved.",
                "recommended_action": (
                    "web_search"
                    if action != "web_search"
                    else "vector_search"
                ),
            }

        # ----------------------------------------------------
        # Keep evaluator context bounded.
        # ----------------------------------------------------

        evidence = []

        for index, result in enumerate(
            results[:8],
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
                or ""
            )

            evidence.append(
                {
                    "id": index,
                    "source": (
                        result.get(
                            "source"
                        )
                        or result.get(
                            "filename"
                        )
                        or result.get(
                            "url"
                        )
                        or ""
                    ),
                    "content": str(
                        content
                    )[:5000],
                }
            )

        system_prompt = """
You are an evidence evaluator for an enterprise
RAG system.

Determine whether the retrieved evidence is sufficient
to answer the user's question accurately.

Rules:

- sufficient=true ONLY when the evidence directly
  supports an answer.
- Do not assume missing facts.
- If evidence is weak, irrelevant, contradictory,
  or insufficient, return sufficient=false.
- confidence must be between 0 and 1.
- recommended_action must be one of:
    vector_search
    document_search
    table_search
    web_search

Return ONLY valid JSON:

{
  "sufficient": true,
  "confidence": 0.0,
  "reason": "brief explanation",
  "recommended_action": "vector_search"
}
"""

        user_prompt = f"""
USER QUESTION:

{query}

RETRIEVAL ACTION:

{action}

RETRIEVED EVIDENCE:

{json.dumps(
    evidence,
    ensure_ascii=False,
    indent=2,
)}

Evaluate the evidence.
"""

        fallback_action = (
            "web_search"
            if action != "web_search"
            else "vector_search"
        )

        fallback = {
            "sufficient": False,
            "confidence": 0.0,
            "reason": "Evidence evaluation failed.",
            "recommended_action": fallback_action,
        }

        result = self._generate_json(
            system_prompt,
            user_prompt,
            fallback=fallback,
        )

        try:

            confidence = float(
                result.get(
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
            result.get(
                "sufficient",
                False,
            )
        )

        recommended_action = result.get(
            "recommended_action"
        )

        allowed_actions = {
            "vector_search",
            "document_search",
            "table_search",
            "web_search",
        }

        if (
            recommended_action
            not in allowed_actions
        ):

            recommended_action = (
                fallback_action
            )

        return {
            "sufficient": sufficient,
            "confidence": confidence,
            "reason": str(
                result.get(
                    "reason",
                    "",
                )
            ),
            "recommended_action": (
                recommended_action
            ),
        }

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    def generate_answer(
        self,
        query,
        chunks,
        source_type="document",
    ):
        """
        Generate a grounded final answer from retrieved
        evidence.

        Compatible with app.py.
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        if not chunks:

            return (
                "I could not find sufficient "
                "evidence to answer that question."
            )

        evidence = []

        for index, chunk in enumerate(
            chunks[:12],
            start=1,
        ):

            if not isinstance(
                chunk,
                dict,
            ):
                continue

            content = (
                chunk.get(
                    "content"
                )
                or chunk.get(
                    "text"
                )
                or ""
            )

            if not str(
                content
            ).strip():

                continue

            source = (
                chunk.get(
                    "source"
                )
                or chunk.get(
                    "filename"
                )
                or chunk.get(
                    "file_name"
                )
                or chunk.get(
                    "url"
                )
                or "Unknown source"
            )

            evidence.append(
                {
                    "id": index,
                    "source": str(
                        source
                    ),
                    "content": str(
                        content
                    )[:7000],
                }
            )

        if not evidence:

            return (
                "I could not find usable evidence "
                "to answer that question."
            )

        system_prompt = """
You are the final answer generator for an enterprise
document intelligence and agentic RAG system.

Answer the user's question using ONLY the supplied
retrieved evidence.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge unless it is explicitly
   present in the evidence.
3. If the evidence does not fully answer the question,
   clearly say what is missing.
4. Prefer precise, direct answers.
5. When evidence comes from uploaded documents,
   identify the relevant source when useful.
6. When web evidence is supplied, preserve source URLs
   when they are available.
7. Do not claim that you searched the web unless web
   evidence is actually supplied.
8. If sources disagree, explicitly mention the conflict.
9. Never fabricate citations, URLs, page numbers, or
   document metadata.

Return a normal natural-language answer.
"""

        user_prompt = f"""
SOURCE TYPE:

{source_type}

USER QUESTION:

{query}

RETRIEVED EVIDENCE:

{json.dumps(
    evidence,
    ensure_ascii=False,
    indent=2,
)}

Write the final grounded answer.
"""

        try:

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model_name,

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],

                    temperature=0,
                )
            )

            return self._extract_content(
                response
            )

        except Exception as e:

            print(
                f"Final answer generation failed: {e}"
            )

            # ------------------------------------------------
            # Safe fallback: return evidence instead of
            # inventing an answer.
            # ------------------------------------------------

            first = evidence[0]

            return (
                "I found relevant evidence, but "
                "the final answer could not be generated.\n\n"
                f"Source: {first['source']}\n"
                f"{first['content']}"
            )

    # ========================================================
    # WEB SEARCH
    # ========================================================

    def web_search(
        self,
        query,
        max_results=5,
    ):
        """
        Perform web search through OpenRouter's server-side
        web search tool.

        app.py calls:

            llm.web_search(
                query=query,
                max_results=self.top_k
            )
        """

        if not query or not query.strip():

            return []

        try:

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model_name,

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Search the web for the user's "
                                "question. Return factual search "
                                "results with source URLs. "
                                "Do not invent sources."
                            ),
                        },
                        {
                            "role": "user",
                            "content": query.strip(),
                        },
                    ],

                    temperature=0,

                    tools=[
                        {
                            "type": "openrouter:web_search",
                            "parameters": {
                                "max_results": max(
                                    1,
                                    int(
                                        max_results
                                    ),
                                ),
                            },
                        }
                    ],
                )
            )

            content = self._extract_content(
                response
            )

            # ------------------------------------------------
            # IMPORTANT:
            # OpenRouter server-side web search returns
            # search context to the model, and the model
            # returns the synthesized result in message.content.
            #
            # Convert that into the result structure expected
            # by app.py.
            # ------------------------------------------------

            return [
                {
                    "content": content,
                    "text": content,
                    "source": "OpenRouter Web Search",
                    "source_type": "web",
                    "search_type": "web_search",
                }
            ]

        except Exception as e:

            print(
                f"OpenRouter web search failed: {e}"
            )

            return []

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def test_connection(self):

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model_name,

                messages=[
                    {
                        "role": "user",
                        "content": "Reply with OK.",
                    }
                ],

                temperature=0,
            )
        )

        return self._extract_content(
            response
        )