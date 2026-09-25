import json
import os
import time
from pathlib import Path

import requests


API_URL = "http://localhost:8000/api/query"
DOCUMENT_ID = os.getenv("GRAPHRAG_DOCUMENT_ID", "")

BASE_DIR = Path(__file__).resolve().parents[2]

QUESTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "datasets"
    / "questions.json"
)

RESULTS_FILE = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "benchmark_results.json"
)


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_hit_at_k(
    retrieved_chunks: list[str],
    expected_chunks: list[str],
    k: int,
) -> int:
    retrieved_top_k = retrieved_chunks[:k]

    return int(
        any(
            chunk in expected_chunks
            for chunk in retrieved_top_k
        )
    )


def calculate_recall_at_k(
    retrieved_chunks: list[str],
    expected_chunks: list[str],
    k: int,
) -> float:
    if not expected_chunks:
        return 0.0

    retrieved_top_k = set(retrieved_chunks[:k])
    expected = set(expected_chunks)

    relevant_retrieved = retrieved_top_k.intersection(expected)

    return len(relevant_retrieved) / len(expected)


def calculate_mrr(
    retrieved_chunks: list[str],
    expected_chunks: list[str],
) -> float:
    expected = set(expected_chunks)

    for rank, chunk_id in enumerate(
        retrieved_chunks,
        start=1,
    ):
        if chunk_id in expected:
            return 1.0 / rank

    return 0.0


def calculate_keyword_coverage(
    answer: str,
    expected_keywords: list[str],
) -> float:
    if not expected_keywords:
        return 0.0

    answer_lower = answer.lower()

    matched = sum(
        1
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    )

    return matched / len(expected_keywords)


def main():
    if not DOCUMENT_ID:
        raise RuntimeError(
            "Set GRAPHRAG_DOCUMENT_ID to the ready document UUID before running the benchmark."
        )
    questions = load_questions()

    results = []

    for question in questions:
        question_id = question["id"]
        query = question["question"]
        expected_chunks = question.get(
            "expected_chunks",
            [],
        )
        expected_keywords = question.get(
            "expected_keywords",
            [],
        )

        print("\n" + "=" * 80)
        print(f"{question_id}: {query}")
        print("=" * 80)

        start_time = time.perf_counter()

        try:
            response = requests.post(
                API_URL,
                json={
    "document_id": DOCUMENT_ID,
    "question": query,
    "top_k": 5,
},
                timeout=180,
            )

            response.raise_for_status()

            data = response.json()

            elapsed = time.perf_counter() - start_time

            answer = data.get(
                "answer",
                "",
            )

            # evidence = data.get(
                # "evidence",
                # [],
            # )

            # retrieved_chunks = [
                # item.get("chunk_id")
                # for item in evidence
                # if item.get("chunk_id")
            # ]
            evidence = data.get("evidence", [])

            retrieved_chunks = [
                item.get("chunk_id")
                for item in evidence
                    if item.get("chunk_id")
                ]

                # Fallback: some API responses may expose sources instead of evidence
            if not retrieved_chunks:
                sources = data.get("sources", [])

                retrieved_chunks = [
                item.get("chunk_id")
        for item in sources
        if item.get("chunk_id")
    ]
                print(
    f"[BENCHMARK] Retrieved chunks: {retrieved_chunks}"
)



            hit_at_1 = calculate_hit_at_k(
                retrieved_chunks,
                expected_chunks,
                1,
            )

            hit_at_3 = calculate_hit_at_k(
                retrieved_chunks,
                expected_chunks,
                3,
            )

            hit_at_5 = calculate_hit_at_k(
                retrieved_chunks,
                expected_chunks,
                5,
            )

            recall_at_3 = calculate_recall_at_k(
                retrieved_chunks,
                expected_chunks,
                3,
            )

            recall_at_5 = calculate_recall_at_k(
                retrieved_chunks,
                expected_chunks,
                5,
            )

            mrr = calculate_mrr(
                retrieved_chunks,
                expected_chunks,
            )

            keyword_coverage = calculate_keyword_coverage(
                answer,
                expected_keywords,
            )

            citation_validation = data.get(
                "citation_validation",
                {},
            )

            result = {
                "id": question_id,
                "question": query,
                "expected_chunks": expected_chunks,
                "retrieved_chunks": retrieved_chunks,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "recall_at_3": round(
                    recall_at_3,
                    4,
                ),
                "recall_at_5": round(
                    recall_at_5,
                    4,
                ),
                "mrr": round(
                    mrr,
                    4,
                ),
                "keyword_coverage": round(
                    keyword_coverage,
                    4,
                ),
                "citation_valid": citation_validation.get(
                    "citation_valid",
                    False,
                ),
                "citation_count": citation_validation.get(
                    "citation_count",
                    0,
                ),
                "latency_seconds": round(
                    elapsed,
                    3,
                ),
                "answer": answer,
            }

            results.append(result)

            print(f"Expected chunks : {expected_chunks}")
            print(f"Retrieved       : {retrieved_chunks}")
            print(f"Hit@1           : {hit_at_1}")
            print(f"Hit@3           : {hit_at_3}")
            print(f"Hit@5           : {hit_at_5}")
            print(f"Recall@3        : {recall_at_3:.4f}")
            print(f"Recall@5        : {recall_at_5:.4f}")
            print(f"MRR             : {mrr:.4f}")
            print(f"Keyword coverage: {keyword_coverage:.4f}")
            print(
                f"Citations valid : "
                f"{citation_validation.get('citation_valid', False)}"
            )
            print(
                f"Latency         : "
                f"{elapsed:.3f}s"
            )

        except Exception as error:
            elapsed = time.perf_counter() - start_time

            print(f"ERROR: {error}")

            results.append(
                {
                    "id": question_id,
                    "question": query,
                    "expected_chunks": expected_chunks,
                    "retrieved_chunks": [],
                    "hit_at_1": 0,
                    "hit_at_3": 0,
                    "hit_at_5": 0,
                    "recall_at_3": 0.0,
                    "recall_at_5": 0.0,
                    "mrr": 0.0,
                    "keyword_coverage": 0.0,
                    "citation_valid": False,
                    "citation_count": 0,
                    "latency_seconds": round(
                        elapsed,
                        3,
                    ),
                    "error": str(error),
                }
            )

    successful = [
        result
        for result in results
        if "error" not in result
    ]

    total = len(successful)

    if total:
        summary = {
            "questions": total,
            "average_latency_seconds": round(
                sum(
                    result["latency_seconds"]
                    for result in successful
                )
                / total,
                3,
            ),
            "hit_at_1": round(
                sum(
                    result["hit_at_1"]
                    for result in successful
                )
                / total,
                4,
            ),
            "hit_at_3": round(
                sum(
                    result["hit_at_3"]
                    for result in successful
                )
                / total,
                4,
            ),
            "hit_at_5": round(
                sum(
                    result["hit_at_5"]
                    for result in successful
                )
                / total,
                4,
            ),
            "recall_at_3": round(
                sum(
                    result["recall_at_3"]
                    for result in successful
                )
                / total,
                4,
            ),
            "recall_at_5": round(
                sum(
                    result["recall_at_5"]
                    for result in successful
                )
                / total,
                4,
            ),
            "mrr": round(
                sum(
                    result["mrr"]
                    for result in successful
                )
                / total,
                4,
            ),
            "average_keyword_coverage": round(
                sum(
                    result["keyword_coverage"]
                    for result in successful
                )
                / total,
                4,
            ),
            "citation_validity_rate": round(
                sum(
                    bool(result["citation_valid"])
                    for result in successful
                )
                / total,
                4,
            ),
        }
    else:
        summary = {
            "questions": 0,
            "average_latency_seconds": 0.0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "average_keyword_coverage": 0.0,
            "citation_validity_rate": 0.0,
        }

    output = {
        "document_id": DOCUMENT_ID,
        "legacy_data": False,
        "summary": summary,
        "results": results,
    }

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("GRAPHRAG-X RETRIEVAL BENCHMARK")
    print("=" * 80)

    print(
        f"Questions              : "
        f"{summary['questions']}"
    )

    print(
        f"Average latency        : "
        f"{summary['average_latency_seconds']}s"
    )

    print(
        f"Hit@1                  : "
        f"{summary['hit_at_1']}"
    )

    print(
        f"Hit@3                  : "
        f"{summary['hit_at_3']}"
    )

    print(
        f"Hit@5                  : "
        f"{summary['hit_at_5']}"
    )

    print(
        f"Recall@3               : "
        f"{summary['recall_at_3']}"
    )

    print(
        f"Recall@5               : "
        f"{summary['recall_at_5']}"
    )

    print(
        f"MRR                    : "
        f"{summary['mrr']}"
    )

    print(
        f"Keyword coverage       : "
        f"{summary['average_keyword_coverage']}"
    )

    print(
        f"Citation validity rate: "
        f"{summary['citation_validity_rate']}"
    )

    print("=" * 80)

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()
