import json
import sys
import time
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2])
)

from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.reranker_service import RerankerService


QUESTIONS_FILE = Path(
    "evaluation/datasets/questions.json"
)

RESULTS_FILE = Path(
    "evaluation/results/retrieval_comparison.json"
)


def hit_at_k(retrieved, expected, k):
    return int(
        bool(
            set(retrieved[:k])
            & set(expected)
        )
    )


def recall_at_k(retrieved, expected, k):
    expected_set = set(expected)

    if not expected_set:
        return 0.0

    retrieved_set = set(retrieved[:k])

    return len(
        retrieved_set & expected_set
    ) / len(expected_set)


def reciprocal_rank(retrieved, expected):
    expected_set = set(expected)

    for index, chunk_id in enumerate(
        retrieved,
        start=1,
    ):
        if chunk_id in expected_set:
            return 1.0 / index

    return 0.0


def evaluate_retriever(
    name,
    search_function,
    questions,
):
    results = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        expected = item["expected_chunks"]

        start = time.perf_counter()

        documents = search_function(
            question,
            5,
        )

        elapsed = time.perf_counter() - start

        retrieved = [
            document.get("chunk_id")
            for document in documents
            if document.get("chunk_id")
        ]

        result = {
            "id": question_id,
            "question": question,
            "expected": expected,
            "retrieved": retrieved,
            "hit_at_1": hit_at_k(
                retrieved,
                expected,
                1,
            ),
            "hit_at_3": hit_at_k(
                retrieved,
                expected,
                3,
            ),
            "hit_at_5": hit_at_k(
                retrieved,
                expected,
                5,
            ),
            "recall_at_3": recall_at_k(
                retrieved,
                expected,
                3,
            ),
            "recall_at_5": recall_at_k(
                retrieved,
                expected,
                5,
            ),
            "mrr": reciprocal_rank(
                retrieved,
                expected,
            ),
            "latency_seconds": round(
                elapsed,
                4,
            ),
        }

        results.append(result)

        print(
            f"{question_id}: "
            f"Hit@1={result['hit_at_1']} "
            f"Hit@3={result['hit_at_3']} "
            f"Recall@5={result['recall_at_5']:.2f} "
            f"MRR={result['mrr']:.2f} "
            f"{result['latency_seconds']:.3f}s"
        )

    summary = {
        "retriever": name,
        "questions": len(results),
        "hit_at_1": round(
            sum(r["hit_at_1"] for r in results)
            / len(results),
            4,
        ),
        "hit_at_3": round(
            sum(r["hit_at_3"] for r in results)
            / len(results),
            4,
        ),
        "hit_at_5": round(
            sum(r["hit_at_5"] for r in results)
            / len(results),
            4,
        ),
        "recall_at_3": round(
            sum(r["recall_at_3"] for r in results)
            / len(results),
            4,
        ),
        "recall_at_5": round(
            sum(r["recall_at_5"] for r in results)
            / len(results),
            4,
        ),
        "mrr": round(
            sum(r["mrr"] for r in results)
            / len(results),
            4,
        ),
        "average_latency_seconds": round(
            sum(
                r["latency_seconds"]
                for r in results
            ) / len(results),
            4,
        ),
        "details": results,
    }

    return summary


def main():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    retrieval = RetrievalService()
    reranker = RerankerService()

    all_results = []

    try:
        print("\n" + "=" * 80)
        print("GRAPHRAG-X RETRIEVAL COMPARISON")
        print("=" * 80)

        # --------------------------------------------------
        # 1. Vector
        # --------------------------------------------------

        print("\n[1/4] VECTOR SEARCH")

        vector_result = evaluate_retriever(
            "Vector",
            retrieval.vector_search,
            questions,
        )

        all_results.append(vector_result)

        # --------------------------------------------------
        # 2. BM25
        # --------------------------------------------------

        print("\n[2/4] BM25 SEARCH")

        bm25_result = evaluate_retriever(
            "BM25",
            retrieval.bm25_search,
            questions,
        )

        all_results.append(bm25_result)

        # --------------------------------------------------
        # 3. Hybrid
        # --------------------------------------------------

        print("\n[3/4] HYBRID SEARCH")

        hybrid_result = evaluate_retriever(
            "Hybrid",
            retrieval.hybrid_search,
            questions,
        )

        all_results.append(hybrid_result)

        # --------------------------------------------------
        # 4. Hybrid + Reranker
        # --------------------------------------------------

        print("\n[4/4] HYBRID + RERANKER")

        def hybrid_reranked(
            question,
            limit=5,
        ):
            candidates = retrieval.hybrid_search(
                question,
                limit=10,
            )

            return reranker.rerank(
                question,
                candidates,
                top_k=limit,
            )

        reranked_result = evaluate_retriever(
            "Hybrid + Reranker",
            hybrid_reranked,
            questions,
        )

        all_results.append(
            reranked_result
        )

    finally:
        retrieval.close()

    output = {
        "dataset": str(
            QUESTIONS_FILE
        ),
        "question_count": len(
            questions
        ),
        "results": all_results,
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
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("RETRIEVAL COMPARISON SUMMARY")
    print("=" * 80)

    print(
        f"{'Retriever':<22}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Recall@5':>12}"
        f"{'MRR':>10}"
        f"{'Latency':>12}"
    )

    print("-" * 80)

    for result in all_results:
        print(
            f"{result['retriever']:<22}"
            f"{result['hit_at_1']:>10.2f}"
            f"{result['hit_at_3']:>10.2f}"
            f"{result['recall_at_5']:>12.2f}"
            f"{result['mrr']:>10.2f}"
            f"{result['average_latency_seconds']:>12.3f}s"
        )

    print("=" * 80)

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()