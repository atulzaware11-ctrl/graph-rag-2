# Import the OpenAI client library to interact with LLMs
from openai import OpenAI

# Import project settings (contains API keys, model name, etc.)
from backend.app.core.config import settings


class LLMQuotaExceededError(RuntimeError):
    """Raised when the upstream provider rejects generation for quota/rate limits."""


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status


def _quota_error_text(error: Exception) -> str:
    parts = [str(error), repr(getattr(error, "body", ""))]
    response = getattr(error, "response", None)
    if response is not None:
        try:
            parts.append(response.text)
        except Exception:
            pass
    return " ".join(parts).lower()


def _is_quota_error(error: Exception) -> bool:
    status = _status_code(error)
    if status == 429:
        return True
    if status != 402:
        return False
    message = _quota_error_text(error)
    return any(
        marker in message
        for marker in ("quota", "insufficient", "rate limit", "free-models-per-day")
    )


class LLMService:
    """
    LLMService is a wrapper around the OpenAI client.
    It handles initialization and provides a method to generate answers
    using retrieved evidence and knowledge graph context.
    """

    def __init__(self) -> None:
        # Ensure the API key is configured; otherwise, raise an error
        if not settings.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        # Initialize the OpenAI client with the OpenRouter API key and base URL
        self.client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
        )

    def generate(self, question: str, contexts: list[dict]) -> str:
        """
        Generate an answer using retrieved documents and knowledge graph evidence.

        The model is explicitly instructed to cite only the supplied numbered sources.
        """

        # Collect formatted context blocks (documents + knowledge graph)
        context_blocks = []
        source_number = 1  # Track numbering for sources

        for item in contexts:
            # Extract text and page info from each context item
            text = item.get("text", "")
            page = item.get("page")

            # If the chunk is from the knowledge graph, mark it separately
            if item.get("chunk_id") == "knowledge_graph":
                context_blocks.append("KNOWLEDGE GRAPH:\n" + text)
                continue

            # Otherwise, format as a numbered source with page info
            context_blocks.append(
                f"[Source {source_number}] Page {page}\n{text}"
            )
            source_number += 1

        # Join all context blocks into one string with spacing
        context_text = "\n\n".join(context_blocks)

        # Build the prompt with strict rules for factual answering
        prompt = f"""
You are GraphRAG-X, an AI research assistant.

Answer the user's question ONLY using the supplied evidence.

IMPORTANT RULES:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If evidence is insufficient, say:
   "There is not enough evidence in the provided documents."
4. Cite important factual statements using:
   [Source 1], [Source 2], etc.
5. Only use source numbers that actually exist below.
6. Knowledge graph information may be used to connect entities
   and relationships, but document sources should support factual
   claims whenever possible.
7. Keep the answer concise and technically accurate.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context_text}

Provide the final answer with source citations.
"""

        # Send the prompt to the LLM via OpenRouter
        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,  # Model name from settings
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
        except Exception as exc:
            if _is_quota_error(exc):
                raise LLMQuotaExceededError(
                    "The AI generation service has reached its request limit."
                ) from None
            raise

        # Return the model's answer text, or empty string if missing
        return response.choices[0].message.content or ""
