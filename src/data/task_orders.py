"""task_orders.py - Predefined task orderings for reproducibility."""

from typing import List

# RQ1 benchmark: 15-task sequence matching the thesis evaluation protocol.
# Each task: 2,000 balanced train / 1,000 balanced test, seed=42.
RQ1_ORDERS = {
    0: [
        "sst2", "ag_news", "mnli", "trec", "imdb",
        "dbpedia", "snli", "cola", "yelp", "yahoo_answers",
        "qnli", "mrpc", "amazon_reviews", "20news", "rte",
    ],
}

BENCHMARK_ORDERS = {
    "rq1": RQ1_ORDERS,
}


def get_task_order(benchmark: str, order_id: int) -> List[str]:
    orders = BENCHMARK_ORDERS.get(benchmark)
    if orders is None:
        raise ValueError(f"Unknown benchmark '{benchmark}'")
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found for benchmark '{benchmark}'")
    return orders[order_id]
