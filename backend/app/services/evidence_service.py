import re


class EvidenceService:
    """
    Builds structured evidence from reranked retrieval results.

    Responsibilities:

    1. Normalize retrieval scores.
    2. Assign stable source numbers.
    3. Prepare citation-ready evidence.
    4. Calculate retrieval/evidence confidence.
    5. Detect source references used by the generated answer.

    Important:
    The confidence calculated here is NOT a probability that
    the answer is factually correct.

    It represents the quality/strength of the retrieved evidence.
    """

    # ---------------------------------------------------------
    # SCORE NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def normalize_scores(
        documents: list[dict],
    ) -> list[dict]:
        """
        Convert reranker scores into a 0-1 relative score.

        CrossEncoder scores can be negative or positive.
        Therefore, the raw score should not be presented as
        a percentage.

        We use min-max normalization across the current
        retrieved candidates.
        """

        if not documents:
            return []

        scores = [
            float(
                document.get(
                    "rerank_score",
                    0.0,
                )
            )
            for document in documents
        ]

        minimum = min(scores)
        maximum = max(scores)

        # If every score is identical, give every source
        # the same normalized relevance.
        if maximum == minimum:

            normalized = [
                1.0
                for _ in documents
            ]

        else:

            normalized = [
                (score - minimum)
                / (maximum - minimum)
                for score in scores
            ]

        results = []

        for document, normalized_score in zip(
            documents,
            normalized,
        ):

            item = document.copy()

            item["evidence_score"] = round(
                float(normalized_score),
                4,
            )

            results.append(item)

        return results

    # ---------------------------------------------------------
    # SOURCE BUILDING
    # ---------------------------------------------------------

    def build_sources(
        self,
        documents: list[dict],
    ) -> list[dict]:
        """
        Convert retrieved documents into citation-ready
        source objects.
        """

        normalized_documents = self.normalize_scores(
            documents
        )

        sources = []

        for index, document in enumerate(
            normalized_documents,
            start=1,
        ):

            sources.append(
                {
                    "source": index,
                    "chunk_id": document.get(
                        "chunk_id"
                    ),
                    "page": document.get(
                        "page"
                    ),
                    "text": document.get(
                        "text",
                        "",
                    ),
                    "evidence_score": document.get(
                        "evidence_score",
                        0.0,
                    ),
                    "rerank_score": document.get(
                        "rerank_score",
                    ),
                    "vector_score": document.get(
                        "vector_score",
                    ),
                    "bm25_score": document.get(
                        "bm25_score",
                    ),
                }
            )

        return sources

    # ---------------------------------------------------------
    # EVIDENCE CONFIDENCE
    # ---------------------------------------------------------

    def calculate_confidence(
        self,
        sources: list[dict],
        graph_entities: int = 0,
    ) -> dict:
        """
        Calculate a simple retrieval evidence confidence.

        Factors:

        - Strength of retrieved evidence.
        - Number of supporting sources.
        - Presence of graph evidence.

        This is deliberately NOT called factual accuracy.
        """

        if not sources:
            return {
                "score": 0.0,
                "label": "Low",
                "reason": "No document evidence was retrieved.",
            }

        evidence_scores = [
            float(
                source.get(
                    "evidence_score",
                    0.0,
                )
            )
            for source in sources
        ]

        # Strongest evidence.
        strongest = max(evidence_scores)

        # Average evidence strength.
        average = sum(
            evidence_scores
        ) / len(evidence_scores)

        # More supporting sources improve coverage,
        # but we cap the contribution.
        source_factor = min(
            len(sources) / 5.0,
            1.0,
        )

        # Graph evidence gives an additional supporting signal.
        graph_factor = (
            0.10
            if graph_entities > 0
            else 0.0
        )

        confidence = (
            (0.50 * strongest)
            + (0.25 * average)
            + (0.15 * source_factor)
            + graph_factor
        )

        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

        percentage = round(
            confidence * 100,
            1,
        )

        if percentage >= 75:
            label = "High"
        elif percentage >= 45:
            label = "Medium"
        else:
            label = "Low"

        return {
            "score": percentage,
            "label": label,
            "reason": (
                "Based on retrieved evidence strength, "
                "source coverage, and graph support."
            ),
        }

    # ---------------------------------------------------------
    # CITATION EXTRACTION
    # ---------------------------------------------------------

    @staticmethod
    def extract_citations(
        answer: str,
    ) -> list[int]:
        """
        Extract references such as:

            [Source 1]
            [Source 3]

        from the generated answer.
        """

        matches = re.findall(
            r"\[Source\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )

        return sorted(
            {
                int(match)
                for match in matches
            }
        )

    # ---------------------------------------------------------
    # CITATION VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def validate_citations(
        answer: str,
        sources: list[dict],
    ) -> dict:
        """
        Check whether the LLM cited sources that actually exist
        in the backend response.
        """

        cited_sources = EvidenceService.extract_citations(
            answer
        )

        available_sources = {
            source["source"]
            for source in sources
        }

        valid = [
            source
            for source in cited_sources
            if source in available_sources
        ]

        invalid = [
            source
            for source in cited_sources
            if source not in available_sources
        ]

        return {
            "cited_sources": cited_sources,
            "valid_sources": valid,
            "invalid_sources": invalid,
            "citation_count": len(valid),
            "citation_valid": len(invalid) == 0,
        }