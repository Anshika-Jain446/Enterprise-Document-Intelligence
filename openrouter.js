const MAX_TOKENS = 1000;

export async function askLLM(messages, model) {
  const apiKey = process.env.OPENROUTER_API_KEY;

  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY is missing");
  }

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: model || process.env.OPENROUTER_MODEL,
      messages,
      max_tokens: MAX_TOKENS,
      temperature: 0.2,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data?.error?.message ||
      `OpenRouter request failed: ${response.status}`
    );
  }

  return data?.choices?.[0]?.message?.content || "";
}