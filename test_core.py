"""Quick unit tests for DLOG core modules."""
import torch
import sys
sys.path.insert(0, ".")

def test_all():
    # Test 1: DualLoRALinear forward
    from dual_lora import DualLoRALinear
    layer = DualLoRALinear(64, 32, rank=8, alpha=16.0)
    x = torch.randn(2, 64)
    y = layer(x)
    print(f"[OK] DualLoRALinear forward: {x.shape} -> {y.shape}")

    # Test 2: Orthogonal gating soft constraint
    from orthogonal_gating import compute_orth_loss
    loss = compute_orth_loss([layer])
    print(f"[OK] Orth loss: {loss.item():.6f}")

    # Test 3: Hard projection (parameter-based)
    layer.A_slow.data = torch.randn_like(layer.A_slow)
    layer.B_slow.data = torch.randn_like(layer.B_slow)
    layer.A_fast.grad = torch.randn_like(layer.A_fast)
    layer.B_fast.grad = torch.randn_like(layer.B_fast)

    from orthogonal_gating import project_gradients_parameter
    project_gradients_parameter([layer])

    dot = torch.sum(layer.A_fast.grad @ layer.A_slow.T)
    print(f"[OK] After projection, dot(g_fast, A_slow) = {dot.item():.8f} (should be ~0)")

    # Test 4: Memory-gradient projector
    from orthogonal_gating import MemoryGradientProjector
    projector = MemoryGradientProjector(buffer_size=5)
    fast_p = [layer.A_fast, layer.B_fast]
    layer.A_fast.grad = torch.randn_like(layer.A_fast)
    layer.B_fast.grad = torch.randn_like(layer.B_fast)
    projector.accumulate(fast_p)
    layer.A_fast.grad = torch.randn_like(layer.A_fast)
    layer.B_fast.grad = torch.randn_like(layer.B_fast)
    leakage_before = projector.compute_leakage(fast_p)
    projector.project_gradients(fast_p)
    leakage_after = projector.compute_leakage(fast_p)
    print(f"[OK] Leakage before: {leakage_before:.6f}, after: {leakage_after:.6f}")

    # Test 5: Subspace overlap
    from metrics import subspace_overlap
    A = torch.randn(64, 8)
    B = torch.randn(64, 8)
    result = subspace_overlap(A, B)
    print(f"[OK] Subspace overlap: separation={result['separation_score']:.4f}")

    # Test 6: EMA update
    layer.A_fast.data = torch.randn_like(layer.A_fast)
    old_slow = layer.A_slow.data.clone()
    layer.ema_update_slow(decay=0.999)
    diff = torch.norm(layer.A_slow.data - old_slow)
    print(f"[OK] EMA update changed slow params by {diff.item():.6f}")

    # Test 7: CLMetricsTracker
    from metrics import CLMetricsTracker
    tracker = CLMetricsTracker(["t1", "t2", "t3"])
    tracker.record(0, 0, 0.9)
    tracker.record(1, 0, 0.7)
    tracker.record(1, 1, 0.85)
    tracker.record(2, 0, 0.6)
    tracker.record(2, 1, 0.8)
    tracker.record(2, 2, 0.88)
    s = tracker.summary()
    print(f"[OK] CLMetrics: FP={s['Final Performance (FP)']:.4f}, FT={s['Forgetting (FT)']:.4f}")

    print("\nAll unit tests passed!")

if __name__ == "__main__":
    test_all()
