const MAX_TOKENS = 1000;

export async function askLLM(messages, model, requestedTokens = 800) {
  const requested = Number(requestedTokens);

  // HARD STOP: never allow anything above 1000
  if (requested > MAX_TOKENS) {
    throw new Error(
      `LLM request blocked: ${requested} tokens requested. Maximum allowed is ${MAX_TOKENS}.`
    );
  }

  const max_tokens =
    Number.isFinite(requested) && requested > 0
      ? Math.floor(requested)
      : 800;

  const apiKey = process.env.OPENROUTER_API_KEY;

  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY is missing");
  }

  const body = {
    model: model || process.env.OPENROUTER_MODEL,
    messages,
    max_tokens,
    temperature: 0.2,
  };

  // FINAL SAFETY CHECK
  if (body.max_tokens > MAX_TOKENS) {
    throw new Error("LLM request blocked before sending to OpenRouter.");
  }

  const response = await fetch(
    "https://openrouter.ai/api/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data?.error?.message ||
        `OpenRouter request failed: ${response.status}`
    );
  }

  return data?.choices?.[0]?.message?.content || "";
}