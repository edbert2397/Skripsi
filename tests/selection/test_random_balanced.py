"""Tests for RandomBalancedSelector (SuRe Random-A baseline)."""

from types import SimpleNamespace

from src.selection import build_selector
from src.selection.reservoir import (
    RandomBalancedSelector,
    ReservoirSelector,
)


def _candidates(n: int):
    return [{"id": i} for i in range(n)]


def test_quota_respected():
    sel = RandomBalancedSelector(seed=0)
    out, scores = sel.select_for_buffer(model=None, candidates=_candidates(1000), quota=100, device=None)
    assert len(out) == 100
    assert len({c["id"] for c in out}) == 100
    assert scores is None


def test_seed_reproducibility():
    cands = _candidates(1000)
    a, _ = RandomBalancedSelector(seed=42).select_for_buffer(None, cands, 100, None)
    b, _ = RandomBalancedSelector(seed=42).select_for_buffer(None, cands, 100, None)
    assert [c["id"] for c in a] == [c["id"] for c in b]


def test_different_seeds_differ():
    cands = _candidates(1000)
    a, _ = RandomBalancedSelector(seed=1).select_for_buffer(None, cands, 100, None)
    b, _ = RandomBalancedSelector(seed=2).select_for_buffer(None, cands, 100, None)
    assert [c["id"] for c in a] != [c["id"] for c in b]


def test_alias_works():
    assert ReservoirSelector is RandomBalancedSelector
    sel = ReservoirSelector(seed=7)
    out, _ = sel.select_for_buffer(None, _candidates(50), 10, None)
    assert len(out) == 10


def test_build_selector_both_keys():
    cfg_a = SimpleNamespace(selection_method="reservoir", seed=42)
    cfg_b = SimpleNamespace(selection_method="random_balanced", seed=42)
    assert isinstance(build_selector(cfg_a), RandomBalancedSelector)
    assert isinstance(build_selector(cfg_b), RandomBalancedSelector)


def test_quota_larger_than_candidates():
    sel = RandomBalancedSelector(seed=0)
    out, _ = sel.select_for_buffer(None, _candidates(5), quota=100, device=None)
    assert len(out) == 5
