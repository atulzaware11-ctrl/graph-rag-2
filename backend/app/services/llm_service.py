from openai import OpenAI

from backend.app.core.config import settings


class LLMService:

    def __init__(self) -> None:

        if not settings.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    def generate(
        self,
        question: str,
        contexts: list[dict],
    ) -> str:

        context_text = "\n\n".join(
            [
                (
                    f"[Source {index + 1}] "
                    f"Page {item.get('page')}\n"
                    f"{item.get('text', '')}"
                )
                for index, item in enumerate(contexts)
            ]
        )

        prompt = f"""
You are GraphRAG-X, an AI research assistant.

Answer the user's question ONLY using the supplied evidence.

If the evidence is insufficient, clearly say:
"There is not enough evidence in the provided documents."

Do not invent facts.

USER QUESTION:
{question}

EVIDENCE:
{context_text}

Give a concise but technically accurate answer.

Use source numbers such as [Source 1] when
supporting important claims.
"""

        response = self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual research assistant "
                        "that answers using retrieved evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response.choices[0].message.content or ""
