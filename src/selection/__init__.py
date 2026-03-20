from .orthogonal import OrthogonalSelector
from .feature import FeatureSelector
from .conflict import ConflictSelector
from .surprise import SurpriseSelector
from .reservoir import ReservoirSelector
from .hybrid import HybridSelector


def build_selector(cfg):
    method = getattr(cfg, "selection_method", "orthogonal")
    use_fp16 = getattr(cfg, "fp16", True)
    k = getattr(cfg, "subspace_rank_k", 10)
    n_est = getattr(cfg, "n_estimation_batches", 10)
    grad_bs = getattr(cfg, "grad_batch_size", 8)
    alpha = getattr(cfg, "hybrid_alpha", 0.5)

    if method == "orthogonal":
        return OrthogonalSelector(
            subspace_rank_k=k,
            n_estimation_batches=n_est,
            grad_batch_size=grad_bs,
            use_fp16=use_fp16,
        )
    elif method == "feature":
        return FeatureSelector(
            subspace_rank_k=k,
            n_estimation_batches=n_est,
            score_batch_size=32,
            use_fp16=use_fp16,
        )
    elif method == "conflict":
        return ConflictSelector(
            n_estimation_batches=n_est,
            grad_batch_size=grad_bs,
            use_fp16=use_fp16,
        )
    elif method == "surprise":
        return SurpriseSelector(use_fp16=use_fp16)
    elif method == "reservoir":
        return ReservoirSelector()
    elif method == "hybrid":
        return HybridSelector(
            alpha=alpha,
            subspace_rank_k=k,
            n_estimation_batches=n_est,
            grad_batch_size=grad_bs,
            use_fp16=use_fp16,
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")
