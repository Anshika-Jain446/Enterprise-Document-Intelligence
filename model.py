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
            base_url="https://openrouter.ai/api/v1"
        )

        self.model_name = LLM_MODEL

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        prompt
    ):

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
                        "content": prompt
                    }
                ],

                temperature=0
            )
        )

        if not response.choices:
            raise RuntimeError(
                "OpenRouter returned no choices."
            )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "OpenRouter returned empty content."
            )

        return content.strip()

    # ========================================================
    # CHAT
    # ========================================================

    def chat(
        self,
        messages
    ):

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
                temperature=0
            )
        )

        if not response.choices:
            raise RuntimeError(
                "OpenRouter returned no choices."
            )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "OpenRouter returned empty content."
            )

        return content.strip()