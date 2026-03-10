"""task_orders.py - Predefined task orderings for reproducibility."""

from typing import List

STANDARD_CL_ORDERS = {
    0: ["ag_news", "amazon_reviews", "dbpedia", "yahoo_answers"],
    1: ["dbpedia", "yahoo_answers", "ag_news", "amazon_reviews"],
    2: ["amazon_reviews", "dbpedia", "yahoo_answers", "ag_news"],
}

LNT_ORDERS = {
    0: ["ag_news", "amazon_reviews", "dbpedia", "yahoo_answers", "mnli", "qqp",
        "rte", "sst2", "wic", "cb", "copa", "boolq", "multirc", "imdb", "sst2_v2"],
    1: ["mnli", "dbpedia", "copa", "sst2", "ag_news", "boolq", "qqp",
        "yahoo_answers", "wic", "amazon_reviews", "rte", "cb", "imdb", "multirc", "sst2_v2"],
    2: ["yahoo_answers", "wic", "imdb", "ag_news", "cb", "qqp", "amazon_reviews",
        "rte", "boolq", "mnli", "sst2", "multirc", "dbpedia", "copa", "sst2_v2"],
}

BENCHMARK_ORDERS = {
    "standard_cl": STANDARD_CL_ORDERS,
    "lnt": LNT_ORDERS,
}


def get_task_order(benchmark: str, order_id: int) -> List[str]:
    orders = BENCHMARK_ORDERS.get(benchmark)
    if orders is None:
        raise ValueError(f"Unknown benchmark '{benchmark}'")
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found for benchmark '{benchmark}'")
    return orders[order_id]
