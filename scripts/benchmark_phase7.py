"""Phase 7 GPU benchmark script.

Run this ONLY on a machine with:
  1. An NVIDIA CUDA GPU.
  2. The `diff_gaussian_rasterization` Python package installed. Install it with:
       pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
     (requires CUDA toolkit + nvcc; see the repo's README for CUDA version compatibility.)

What it does:
  1. Verifies grassmann.fast_rasterizer.is_available() returns True.
  2. Creates a moderately-sized model (e.g. 10,000 Gaussians).
  3. Renders the scene from one camera, TIMING both the toy rasterizer and
     the CUDA rasterizer.
  4. Checks that the outputs are numerically close (same image up to small
     differences from EWA / tile-based rendering differences).
  5. Prints speedup factor.

Typical expected result: the CUDA path is 100-500x faster than the toy
rasterizer for thousands of Gaussians at modest resolution.
"""
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from grassmann import quaternion as Q
from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize as toy_rasterize
from grassmann.initialization import init_gaussians_from_points
from grassmann.synthetic import cameras_on_ring
from grassmann.trainable import trainable_from_params
from grassmann.fast_rasterizer import fast_rasterize, is_available


DEVICE = "cuda"
DTYPE = torch.float32
H, W = 480, 640
# H, W = 200, 200
N_GAUSSIANS = 100 # 10_000


def build_large_model(n_gaussians: int, cams):
    """Synthetic model with random positions + random colors, all at t=1.0."""
    torch.manual_seed(0)
    points = torch.randn(n_gaussians, 3, dtype=torch.float64) * 1.0 + torch.tensor([0.0, 0.0, 5.0])
    times_t = torch.ones(n_gaussians, dtype=torch.float64)
    colors = torch.rand(n_gaussians, 3, dtype=torch.float64)
    params_init = init_gaussians_from_points(
        points, times_t, cams, colors=colors,
        sigma_aa=0.005, sigma_bb=0.01, opacity=0.5, sigma_k_pixel=1.0, sigma_k_temporal=1.0,
    )
    return trainable_from_params(params_init, dtype=DTYPE, device=DEVICE)


def time_render(render_fn, n_warmup=3, n_trials=10):
    """Call render_fn() n_warmup + n_trials times; return median time (seconds)."""
    # Warmup
    for _ in range(n_warmup):
        _ = render_fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    trials = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = render_fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        trials.append(time.perf_counter() - start)
    trials.sort()
    return trials[len(trials) // 2]


def main():
    print("=" * 60)
    print("Phase 7 GPU Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nERROR: No CUDA GPU detected. Cannot run this benchmark.")
        return
    if not is_available():
        print("\nERROR: diff_gaussian_rasterization not importable.")
        print("Install with:")
        print("  pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git")
        return

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Model:  {N_GAUSSIANS} Gaussians")
    print(f"Image:  {H} x {W}")
    print()

    # Build scene + model.
    cams = cameras_on_ring(K=1, radius=6.0, image_w=W, image_h=H, fx=400, fy=400)
    model = build_large_model(N_GAUSSIANS, cams)
    background = torch.zeros(3, dtype=DTYPE, device=DEVICE)

    # Wrap renders as closures for timing.
    def toy_render():
        params = model.forward()
        derived = compute_derived(params)
        tc = condition_on_time(params, derived, 1.0)
        sg = project_to_screen(params, tc, cams[0])
        return toy_rasterize(sg, H=H, W=W, background=background)

    def fast_render():
        params = model.forward()
        return fast_rasterize(params, t_0=1.0, cam=cams[0], H=H, W=W,
                               background=background)

    # Numerical sanity: the two renders should be close (but not pixel-identical
    # -- tile-based rasterization has small differences from pure accumulation).
    print("Numerical check...")
    with torch.no_grad():
        img_toy = toy_render()
        img_fast = fast_render()
    print(f"  toy image shape:  {tuple(img_toy.shape)}  min={img_toy.min():.3f} max={img_toy.max():.3f}")
    print(f"  fast image shape: {tuple(img_fast.shape)} min={img_fast.min():.3f} max={img_fast.max():.3f}")

    diff = (img_toy - img_fast).abs()
    print(f"  L1 diff:   mean={diff.mean():.4f} max={diff.max():.4f}")
    if diff.mean() > 0.05:
        print("  WARNING: mean L1 difference is large. The renders may be visually similar")
        print("           but quantitatively off. Inspect images if this bothers you.")

    # Timing.
    print("\nTiming...")
    t_toy = time_render(toy_render)
    t_fast = time_render(fast_render)
    speedup = t_toy / t_fast

    print(f"\n  Toy rasterizer:  {t_toy * 1000:.1f} ms / render")
    print(f"  CUDA rasterizer: {t_fast * 1000:.1f} ms / render")
    print(f"  Speedup:         {speedup:.1f}x")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
