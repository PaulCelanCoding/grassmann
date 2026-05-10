# A1 anchor drift investigation — code paths are byte-identical

**Date:** 2026-05-10
**Branch:** monocular-init
**TL;DR:** Between `b958b68` (24.41 dB val, ~249s) and `e514bc9` (23.50 dB val, 413s), the **default code paths for the gaussian rasterizer are byte-identical**. The 0.91 dB val drift cannot be explained by source-code review; it is an environmental / Modal-image effect.

## Method

`git diff b958b68..e514bc9` covering every file the A1 default path touches:

| file | lines diff | default-path effect |
|---|---|---|
| `grassmann/training.py` | +113 | additive: surfel rasterizer + 2DGS losses, `mu_lr_split`, `lambda_mu_penalty`. **Default `rasterizer="gaussian"`, `mu_constraint="free"`, `mu_lr_split=False` keep all new code dormant.** |
| `grassmann/trainable.py` | +65 | additive: `mu_constraint`, `mu_lr_split`, `clamp_mode`, `eps_schur`. Defaults are legacy. |
| `grassmann/gaussian.py` | +57 | adds `mu_constraint="free"` and `clamp_mode="hard"` defaults. The hard-clamp branch reproduces the legacy `clamp_min(1e-20)` numerically. |
| `grassmann/initialization.py` | +32 | adds `spatial_slice` strategy. `random` (default) is unchanged. |
| `grassmann/density_control.py` | **0** | identical |
| `grassmann/fast_rasterizer.py` | **0** | identical |
| `grassmann/losses.py` | +73 | adds 2DGS regularizers; only called when `aux is not None`, never in default path. |
| `grassmann/datasets/` | **0** | identical |
| `scripts/train_mono.py` | +89 | additive flags; no default value changes that affect the gaussian path. |
| `scripts/train_modal.py` | +88 | additive; pip install line for `diff_surfel_rasterization` added (does not invalidate the upstream `diff_gaussian_rasterization` layer). |

### Spot-checks
- `train_step` signature gained `iter_num` kwarg, but the 2DGS loss block (`if aux is not None`) is unreachable when `rasterizer="gaussian"`.
- `lr_mu_spatial`/`lr_mu_time` are passed through to `build_optimizer` but only consumed when `model.mu_lr_split=True` (default false). The legacy `mu` param group is added on the `else` branch.
- The hard-clamp branch in `condition_on_time` evaluates to `clamp_min(eps)` with `eps=1e-20` — bit-identical to the legacy `clamp_min(1e-20)`.
- `from .surfel_rasterizer import ...` at training.py L35 is a pure-Python import (the CUDA extension load is lazy via `_try_import()`), so it does not affect the CUDA context state.

## Conclusion

The 0.91 dB val regression and the 65% wallclock slowdown (249→413s) are **not in the code**. The remaining causes, ordered by likelihood:

1. **Modal image rebuild + CUDA-driver / kernel difference.** Adding the surfel pip-install line creates a new image hash; while the gaussian-rasterizer layer should still hit the cache, the surrounding torch/cuda layers may have re-pulled minor versions. This would explain both wallclock and bit-level non-determinism in densification.
2. **Densification chaos amplification.** Even with `seed=42`, the screen-space gradient threshold for split is on the boundary of float32 noise; a 1-iteration desync compounds over 14k iters.
3. **Modal scheduling / GPU lottery.** Different L4 SKU or driver between sessions.

## What this means for Wave A

The Combo-AA result (val=24.36 dB at `e514bc9`) **is already 0.86 dB above the e514bc9 anchor (23.50)** but only **0.05 dB below the b958b68 anchor (24.41)**. So Wave A has approximately recovered the historical baseline; the residual to D3DGS that Phase C left unresolved is still ~1.4 dB in metric terms even after Wave A.

## Action

No code fix exists — this drift is environmental. To unblock further claims about residual reduction vs. D3DGS, the only honest path is:

1. Pin `diff_gaussian_rasterization` git SHA in `train_modal.py` (currently `git+...gaussian-rasterization.git` with no ref).
2. Re-run A1 at b958b68 and at HEAD with identical pinned image; if drift persists, it's CUDA non-determinism + densification.

This is a stable-run hygiene issue, not a quality-knob issue. Filed for later; not blocking #4.1 MCMC or #9.1 depth.
