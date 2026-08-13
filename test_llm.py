from llm import GeminiLLM


llm = GeminiLLM()

response = llm._call_llm([
    {
        "role": "user",
        "content": "Say hello in one sentence."
    }
])

print(response)