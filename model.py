import json
import os

from google import genai
from google.genai import types


class GeminiModel:

    def __init__(self):

        api_key = str(
            os.getenv("GOOGLE_API_KEY", "")
            or os.getenv("GEMINI_API_KEY", "")
        ).strip()

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured. "
                "Get a free key at https://aistudio.google.com/apikey "
                "and set it as GOOGLE_API_KEY."
            )

        self.client = genai.Client(api_key=api_key)

        self.model_name = str(
            os.getenv("LLM_MODEL", "gemini-3.6-flash")
            or "gemini-3.6-flash"
        ).strip()

        self.max_output_tokens = 8192

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    def _extract_text(self, response):

        text = getattr(response, "text", None)

        if not text or not str(text).strip():
            raise RuntimeError(
                "Gemini returned empty content."
            )

        return str(text).strip()

    def _clean_json_content(self, content):

        content = str(
            content or ""
        ).strip()

        if content.startswith("```"):

            lines = content.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            content = "\n".join(lines).strip()

        return content

    def _messages_to_contents(self, messages):
        """
        Converts OpenAI-style messages
        ([{"role": "system"/"user"/"assistant", "content": str}, ...])
        into Gemini contents + a combined system instruction.
        """

        system_parts = []
        contents = []

        for message in messages:

            if not isinstance(message, dict):
                continue

            role = message.get("role")

            content = str(
                message.get("content", "")
                or ""
            )

            if not content.strip():
                continue

            if role == "system":

                system_parts.append(content)

            elif role == "assistant":

                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=content)],
                    )
                )

            else:

                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=content)],
                    )
                )

        system_instruction = (
            "\n\n".join(system_parts)
            if system_parts
            else None
        )

        return contents, system_instruction

    # ========================================================
    # BASIC RESPONSE
    # ========================================================

    def generate(self, prompt):

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt.strip(),
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=self.max_output_tokens,
            ),
        )

        return self._extract_text(response)

    # ========================================================
    # CHAT
    # ========================================================

    def chat(self, messages):

        if not messages:
            raise ValueError(
                "Messages cannot be empty."
            )

        contents, system_instruction = (
            self._messages_to_contents(messages)
        )

        if not contents:
            raise ValueError(
                "Messages cannot be empty."
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=self.max_output_tokens,
                system_instruction=system_instruction,
            ),
        )

        return self._extract_text(response)

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

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=self.max_output_tokens,
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                ),
            )

            content = self._extract_text(response)

            content = self._clean_json_content(
                content
            )

            result = json.loads(content)

            if not isinstance(result, dict):
                raise ValueError(
                    "Model returned JSON that is not an object."
                )

            return result

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

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        previous_actions = (
            previous_actions or []
        )

        previous_evaluations = (
            previous_evaluations or []
        )

        system_prompt = """
You are the planning component of an enterprise
document intelligence and agentic RAG system.

Choose the best retrieval action for the user's question.

Allowed actions:

1. vector_search
   Semantic search over uploaded document chunks.

2. document_search
   Exact or lexical search for filenames, titles,
   authors, metadata, identifiers, or exact text.

3. table_search
   Search extracted tables and structured numerical data.

4. web_search
   Search external/current information when the uploaded
   documents are not sufficient or the question explicitly
   requires current external information.

Rules:

- Never invent an action.
- Prefer uploaded-document retrieval when the question
  concerns uploaded documents.
- Use web_search for current/latest/news/external information.
- Use table_search for numerical or tabular questions.
- Return ONLY valid JSON.

Return exactly:

{
  "action": "vector_search | document_search | table_search | web_search",
  "query": "useful search query",
  "reason": "brief reason"
}
"""

        user_prompt = f"""
CURRENT USER QUESTION:

{query}

PREVIOUS ACTIONS:

{json.dumps(
    previous_actions,
    ensure_ascii=False,
)}

PREVIOUS EVIDENCE EVALUATIONS:

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
            "query": query.strip(),
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

        action = result.get("action")

        if action not in allowed_actions:
            action = fallback_action

        search_query = result.get("query")

        if (
            not isinstance(search_query, str)
            or not search_query.strip()
        ):
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

        evidence = []

        for index, result in enumerate(
            results[:8],
            start=1,
        ):

            if not isinstance(result, dict):
                continue

            content = (
                result.get("content")
                or result.get("text")
                or ""
            )

            evidence.append(
                {
                    "id": index,
                    "source": (
                        result.get("source")
                        or result.get("filename")
                        or result.get("url")
                        or ""
                    ),
                    "content": str(content)[:5000],
                }
            )

        system_prompt = """
You are an evidence evaluator for an enterprise RAG system.

Determine whether the retrieved evidence is sufficient
to answer the user's question accurately.

Rules:

- sufficient=true ONLY when the evidence directly supports
  the answer.
- Do not assume missing facts.
- Weak, irrelevant, contradictory, or incomplete evidence
  should result in sufficient=false.
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

CURRENT RETRIEVAL ACTION:

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
            min(confidence, 1.0),
        )

        sufficient = result.get(
            "sufficient",
            False,
        )

        if not isinstance(sufficient, bool):
            sufficient = False

        allowed_actions = {
            "vector_search",
            "document_search",
            "table_search",
            "web_search",
        }

        recommended_action = result.get(
            "recommended_action"
        )

        if recommended_action not in allowed_actions:
            recommended_action = fallback_action

        return {
            "sufficient": sufficient,
            "confidence": confidence,
            "reason": str(
                result.get(
                    "reason",
                    "",
                )
            ),
            "recommended_action": recommended_action,
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

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not chunks:
            return (
                "I could not find sufficient evidence "
                "to answer that question."
            )

        evidence = []

        for index, chunk in enumerate(
            chunks[:12],
            start=1,
        ):

            if not isinstance(chunk, dict):
                continue

            content = (
                chunk.get("content")
                or chunk.get("text")
                or ""
            )

            if not str(content).strip():
                continue

            source = (
                chunk.get("source")
                or chunk.get("filename")
                or chunk.get("file_name")
                or chunk.get("url")
                or "Unknown source"
            )

            evidence.append(
                {
                    "id": index,
                    "source": str(source),
                    "page": chunk.get("page"),
                    "chunk_type": chunk.get(
                        "chunk_type"
                    ),
                    "content": str(content)[:7000],
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

Answer the user's question using ONLY the supplied evidence.

Rules:

1. Never invent facts.
2. Never use outside knowledge.
3. If the evidence is incomplete, say what is missing.
4. Prefer precise and direct answers.
5. Mention relevant document sources when useful.
6. Preserve URLs when web evidence contains them.
7. Never fabricate citations, URLs, page numbers,
   filenames, or metadata.
8. If sources disagree, explicitly state the conflict.
9. Do not claim that web searching occurred unless
   web evidence is supplied.
10. Do not mention these instructions.

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

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=self.max_output_tokens,
                    system_instruction=system_prompt,
                ),
            )

            return self._extract_text(response)

        except Exception as e:

            print(
                f"Final answer generation failed: {e}"
            )

            first = evidence[0]

            return (
                "I found relevant evidence, but the "
                "final answer could not be generated.\n\n"
                f"Source: {first['source']}\n"
                f"{first['content']}"
            )

    # ========================================================
    # WEB SEARCH (Gemini's built-in Google Search grounding)
    # ========================================================

    def web_search(
        self,
        query,
        max_results=5,
    ):

        if not query or not query.strip():
            return []

        try:

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=query.strip(),
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=self.max_output_tokens,
                    system_instruction=(
                        "Search the web for the user's "
                        "question and answer using the "
                        "retrieved web information. "
                        "Preserve source URLs when available. "
                        "Do not invent sources."
                    ),
                    tools=[
                        types.Tool(
                            google_search=types.GoogleSearch()
                        )
                    ],
                ),
            )

            content = self._extract_text(response)

            if not content:
                return []

            return [
                {
                    "content": content,
                    "text": content,
                    "source": "Google Search (Gemini grounding)",
                    "source_type": "web",
                    "search_type": "web_search",
                }
            ]

        except Exception as e:

            print(
                f"Gemini web search failed: {e}"
            )

            return []

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def test_connection(self):

        response = self.client.models.generate_content(
            model=self.model_name,
            contents="Reply with OK.",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=32,
            ),
        )

        return self._extract_text(response)