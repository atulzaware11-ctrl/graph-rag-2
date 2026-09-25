import json
import re

from openai import OpenAI

from backend.app.core.config import settings
from backend.app.services.llm_service import LLMQuotaExceededError


class GraphExtractionService:

    def __init__(self) -> None:

        if not settings.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
        )

    def extract(
        self,
        text: str,
        page: int,
        chunk_id: str,
    ) -> dict:

        prompt = f"""
You are an information extraction system for a research
knowledge graph.

Extract important entities and relationships from the
following research text.

ENTITY TYPES:
- Paper
- Author
- Method
- Model
- Dataset
- Metric
- Concept
- Technology
- Organization
- Task

RELATION TYPES:
- USES
- PROPOSES
- EVALUATES
- EVALUATED_ON
- ACHIEVES
- IMPROVES
- COMPARES_WITH
- RELATED_TO
- AUTHORED_BY
- USES_DATASET
- USES_MODEL

Return ONLY valid JSON.

Required format:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type"
    }}
  ],
  "relationships": [
    {{
      "source": "entity name",
      "relation": "RELATION_TYPE",
      "target": "entity name"
    }}
  ]
}}

Rules:

1. Only extract information explicitly supported by the text.
2. Do not invent entities.
3. Do not invent relationships.
4. Keep entity names concise.
5. Avoid duplicate entities.
6. Use the exact entity names consistently.

PAGE:
{page}

CHUNK ID:
{chunk_id}

TEXT:
{text}
"""

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract factual knowledge from "
                            "research documents."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429:
                raise LLMQuotaExceededError(
                    "Knowledge graph generation has reached the current free-model request limit."
                ) from None
            raise

        content = (
            response.choices[0].message.content
            or "{}"
        )

        return self._parse_json(content)

    @staticmethod
    def _parse_json(content: str) -> dict:

        content = content.strip()

        # Remove markdown code fences if the model adds them.
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
        )

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {
                "entities": [],
                "relationships": [],
            }

        return {
            "entities": data.get(
                "entities",
                [],
            ),
            "relationships": data.get(
                "relationships",
                [],
            ),
        }