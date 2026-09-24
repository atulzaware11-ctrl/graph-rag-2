import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

QUESTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "datasets"
    / "questions.json"
)

BASELINE_FILE = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "baseline_hybrid.json"
)


def main():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    with open(
        BASELINE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        benchmark = json.load(file)

    results = benchmark.get(
        "results",
        [],
    )

    print("\n" + "=" * 80)
    print("GRAPHRAG-X BASELINE RETRIEVAL ANALYSIS")
    print("=" * 80)

    print(
        f"\nQuestions evaluated: "
        f"{len(questions)}"
    )

    print("\nQuestion-level retrieval results:")

    for result in results:
        print("\n" + "-" * 80)

        print(
            f"{result.get('id')}: "
            f"{result.get('question')}"
        )

        print(
            f"Expected : "
            f"{result.get('expected_chunks')}"
        )

        print(
            f"Retrieved: "
            f"{result.get('retrieved_chunks')}"
        )

        print(
            f"Hit@1={result.get('hit_at_1')} | "
            f"Hit@3={result.get('hit_at_3')} | "
            f"Hit@5={result.get('hit_at_5')}"
        )

        print(
            f"Recall@3={result.get('recall_at_3')} | "
            f"Recall@5={result.get('recall_at_5')}"
        )

        print(
            f"MRR={result.get('mrr')}"
        )

    summary = benchmark.get(
        "summary",
        {},
    )

    print("\n" + "=" * 80)
    print("BASELINE SUMMARY")
    print("=" * 80)

    metrics = [
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "average_keyword_coverage",
        "citation_validity_rate",
        "average_latency_seconds",
    ]

    for metric in metrics:
        print(
            f"{metric:<30}: "
            f"{summary.get(metric)}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()