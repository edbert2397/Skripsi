from .base import BaseSelector
from .orthogonal import OrthogonalSelector
from .surprise import SurpriseSelector
from .reservoir import ReservoirSelector
from .hybrid import HybridSelector

SELECTOR_REGISTRY = {
    "orthogonal": OrthogonalSelector,
    "surprise": SurpriseSelector,
    "reservoir": ReservoirSelector,
    "hybrid": HybridSelector,
}


def build_selector(config) -> BaseSelector:
    method = config.selection_method
    if method not in SELECTOR_REGISTRY:
        raise ValueError(f"Unknown selector '{method}'. Choose from {list(SELECTOR_REGISTRY)}")

    kwargs = dict(use_fp16=config.fp16)

    if method in ("orthogonal", "hybrid"):
        kwargs["subspace_rank_k"] = config.subspace_rank_k
        kwargs["n_estimation_batches"] = config.n_estimation_batches
        kwargs["grad_batch_size"] = config.grad_batch_size

    if method == "hybrid":
        kwargs["alpha"] = config.hybrid_alpha

    return SELECTOR_REGISTRY[method](**kwargs)
