"""task_orders.py - Predefined task orderings for reproducibility."""

from typing import List

# RQ1 benchmark: 14-task sequence (CL14, trec dropped).
# Task names match the cl14_balanced_*_cleaned.csv `task_name` column.
# Each task: 2,000 balanced train / 1,000 balanced test, seed=42.
RQ1_ORDERS = {
    0: [
        "sst2", "ag_news", "mnli", "imdb", "dbpedia",
        "snli", "cola", "yelp", "yahoo", "qnli",
        "mrpc", "amazon", "newsgroup20", "rte",
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
