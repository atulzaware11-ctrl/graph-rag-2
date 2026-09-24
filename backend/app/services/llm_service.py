# Import the OpenAI client library to interact with LLMs
from openai import OpenAI

# Import project settings (contains API keys, model name, etc.)
from backend.app.core.config import settings


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

        # Return the model's answer text, or empty string if missing
        return response.choices[0].message.content or ""
