"""ema.py - EMA consolidation step (standalone for clarity)."""


def ema_step(model, beta: float = None):
    """
    Perform one EMA update: θ_slow ← β·θ_slow + (1-β)·θ_fast.

    If beta is provided, temporarily overrides model.beta.
    """
    if beta is not None:
        old_beta = model.beta
        model.beta = beta
    model.ema_update()
    if beta is not None:
        model.beta = old_beta
