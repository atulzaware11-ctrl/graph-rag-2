import json
from pathlib import Path


RESULTS_FILE = Path("evaluation/results/benchmark_results.json")


with RESULTS_FILE.open("r", encoding="utf-8") as file:
    data = json.load(file)


print("\n=== BENCHMARK RESULTS ===\n")

for result in data["results"]:
    print(
        f"{result['id']} | "
        f"{result['latency_seconds']:.3f}s | "
        f"coverage={result['keyword_coverage']} | "
        f"citations={result['citation_valid']} | "
        f"graph={result['graph_entities']} | "
        f"{result['question']}"
    )


print("\n=== SLOWEST QUERIES ===\n")

slowest = sorted(
    data["results"],
    key=lambda x: x["latency_seconds"],
    reverse=True,
)

for result in slowest:
    print(
        f"{result['id']} -> "
        f"{result['latency_seconds']:.3f}s -> "
        f"{result['question']}"
    )


print("\n=== SUMMARY ===\n")

summary = data.get("summary", {})

print(f"Questions: {summary.get('questions')}")
print(f"Successful: {summary.get('successful')}")
print(f"Average latency: {summary.get('average_latency_seconds')}s")
print(f"Average keyword coverage: {summary.get('average_keyword_coverage')}")
print(f"Citation validity rate: {summary.get('citation_validity_rate')}")