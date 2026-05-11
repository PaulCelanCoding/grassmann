"""Run 5-plot spectral / density / motion diagnostic on the Bug-D checkpoint.

Designed to run inside the Modal training image — uses CUDA when available
to evaluate val frames, but the spectral / spatial-distribution plots only
need the checkpoint and CPU.

Outputs PNGs to /checkpoints/<ckpt_dir>/diagnostics/ (committed to gs-checkpoints
volume so we can fetch them via `modal volume get`).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make grassmann importable when run from anywhere.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from grassmann.gaussian import GaussianParams, compute_derived, condition_on_time  # noqa: E402
from grassmann.trainable import TrainableGaussians  # noqa: E402


def _safe_log10(x: torch.Tensor, floor: float = 1e-20) -> torch.Tensor:
    return torch.log10(x.clamp_min(floor))


def diagnostic_main(ckpt_path: str, log_path: str | None, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    n_raw = sd["n_raw"].to(device).float()
    L_raw = sd["L_raw"].to(device).float()
    # mu may be split into mu_time + mu_spatial.
    if "mu" in sd:
        mu = sd["mu"].to(device).float()
    else:
        mu = torch.cat([sd["mu_time"], sd["mu_spatial"]], dim=-1).to(device).float()
    opacity_logit = sd["opacity_logit"].to(device).float()
    N = int(n_raw.shape[0])
    print(f"loaded ckpt: N={N}, device={device}")

    # n_unit (normalized to S^3) — exact same code as TrainableGaussians.forward().
    n_norm = n_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n_unit = n_raw / n_norm

    # Build GaussianParams for compute_derived. sigma_k_temporal default 1e-3.
    sigma_k_temporal = float(sd.get("sigma_k_temporal", torch.tensor(1e-3)))
    params = GaussianParams(
        n=n_unit, L_raw=L_raw, mu=mu,
        opacity=torch.sigmoid(opacity_logit),
        color=torch.zeros(N, 3, device=device, dtype=torch.float32),
        sigma_k_temporal=sigma_k_temporal,
        mu_constraint="free",
        clamp_mode="soft", eps_schur=1e-8,
    )
    d = compute_derived(params)
    Sigma_3D_pre = d.Sigma_3D                    # (N, 3, 3), pre-Schur
    c_world = d.c_world                          # (N, 3)
    Sigma_tt = d.Sigma_tt                        # (N,)
    sigma_tt_pure = getattr(d, "_sigma_tt_pure", Sigma_tt)

    # Σ_3D_t = post-Schur, what the rasterizer sees.
    inv_Stt = 1.0 / sigma_tt_pure.clamp_min(1e-8)
    outer = c_world.unsqueeze(-1) * c_world.unsqueeze(-2)
    Sigma_3D_t = Sigma_3D_pre - inv_Stt.unsqueeze(-1).unsqueeze(-1) * outer

    eigs_3Dt = torch.linalg.eigvalsh(Sigma_3D_t)  # (N, 3) ascending
    eigs_3Dt_desc = eigs_3Dt.flip(-1)             # λ_1 ≥ λ_2 ≥ λ_3

    # Σ_4D = L_plane @ L_plane^T (using compute_derived's internal projection).
    nL = torch.einsum("...i,...ij->...j", n_unit, L_raw)
    L_plane = L_raw - n_unit.unsqueeze(-1) * nL.unsqueeze(-2)
    Sigma_4D = L_plane @ L_plane.transpose(-1, -2)
    eigs_4D = torch.linalg.eigvalsh(Sigma_4D)     # (N, 4) ascending

    # Plane-constraint check: ||Σ_4D n||_2 / ||Σ_4D||_F should be ~machine eps.
    S4n = (Sigma_4D @ n_unit.unsqueeze(-1)).squeeze(-1)
    plane_resid = S4n.norm(dim=-1) / Sigma_4D.flatten(1).norm(dim=-1).clamp_min(1e-30)

    # |n_{1:}| = sin(angle from time-axis pole)
    n_spatial_norm = n_unit[:, 1:].norm(dim=-1)   # (N,)

    print("computing eigvals + plot...")

    # ----- Plot 2: spectral diagnostics -----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    # 2a: Σ_3D_t eigvals (sorted ↓), should be rank-2 (λ_3 ≈ 0)
    for i, (col, lbl) in enumerate(zip([0, 1, 2], ["λ_1 (max)", "λ_2 (mid)", "λ_3 (min, ≈0 expected)"])):
        axes[0, 0].hist(_safe_log10(eigs_3Dt_desc[:, col]).cpu().numpy(),
                        bins=80, alpha=0.55, label=lbl)
    axes[0, 0].set_title("Σ_3D_t (post-Schur) eigvals — log10")
    axes[0, 0].set_xlabel("log10(λ)"); axes[0, 0].legend(fontsize=8)

    # 2b: Σ_4D eigvals — smallest should be ~0 (n-kernel)
    for i in range(4):
        axes[0, 1].hist(_safe_log10(eigs_4D[:, i]).cpu().numpy(),
                        bins=80, alpha=0.5, label=f"eig[{i}]")
    axes[0, 1].set_title("Σ_4D eigvals — log10 (eig[0] is the n-kernel, ≈0)")
    axes[0, 1].set_xlabel("log10(λ)"); axes[0, 1].legend(fontsize=8)

    # 2c: anisotropy ratio λ_1/λ_2 of Σ_3D_t
    aniso = (eigs_3Dt_desc[:, 0] / eigs_3Dt_desc[:, 1].clamp_min(1e-12)).cpu().numpy()
    axes[0, 2].hist(np.log10(np.clip(aniso, 1e-3, 1e6)), bins=80, color="C3")
    axes[0, 2].axvline(np.log10(20), color="k", ls="--", label="threshold log10(20)")
    axes[0, 2].set_title(f"Anisotropy log10(λ_1/λ_2) — q90={np.quantile(aniso,0.9):.1f}")
    axes[0, 2].set_xlabel("log10(λ_1/λ_2)"); axes[0, 2].legend(fontsize=8)

    # 2d: plane-constraint residual
    pr = plane_resid.cpu().numpy()
    axes[1, 0].hist(np.log10(pr.clip(1e-20, 1e2)), bins=80, color="C4")
    axes[1, 0].set_title(f"plane constraint ||Σ_4D n||/||Σ_4D|| (log10) — q99={np.quantile(pr, 0.99):.2e}")
    axes[1, 0].set_xlabel("log10(residual)")

    # 2e: |n_{1:}| histogram on linear scale
    nsn = n_spatial_norm.cpu().numpy()
    axes[1, 1].hist(nsn, bins=80, color="C5")
    axes[1, 1].axvline(1e-2, color="k", ls="--", label="0.01 (static threshold)")
    axes[1, 1].set_title(f"|n_{{1:}}| (sin angle from e₀) — q50={np.median(nsn):.3f}, q99={np.quantile(nsn,0.99):.3f}")
    axes[1, 1].set_xlabel("|n_{1:}|"); axes[1, 1].legend(fontsize=8)

    # 2f: sqrt(Σ_tt) histogram
    sqrt_stt = Sigma_tt.clamp_min(0.0).sqrt().cpu().numpy()
    axes[1, 2].hist(np.log10(sqrt_stt.clip(1e-6, 1e2)), bins=80, color="C6")
    axes[1, 2].set_title(f"log10(√Σ_tt) — q5={np.quantile(sqrt_stt,0.05):.3g}, q50={np.median(sqrt_stt):.3g}, q95={np.quantile(sqrt_stt,0.95):.3g}")
    axes[1, 2].set_xlabel("log10(√Σ_tt)")

    plt.suptitle(f"Spectral diagnostic on Bug-D ckpt (N={N})")
    plt.tight_layout()
    out2 = os.path.join(out_dir, "diag_spectral.png")
    plt.savefig(out2, dpi=110); plt.close()
    print(f"  wrote {out2}")

    # ----- Plot 3: motion decomposition (normal-drift vs tangential-drift) -----
    # From Prop 2.1: rigid-plane velocity ∝ n_0 (time-component of n).
    # Tangential drift on disk: projection of c_world / Σ_tt onto n_{1:}^⊥.
    n_t = n_unit[:, 0].abs().cpu().numpy()                # rigid speed proxy
    # tangential drift magnitude
    tang_speed = (c_world / sigma_tt_pure.clamp_min(1e-8).unsqueeze(-1))
    # project off the spatial-normal direction (n_{1:})
    nsp = n_unit[:, 1:]
    nsp_norm = nsp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    nsp_unit = nsp / nsp_norm                              # (N, 3)
    proj = (tang_speed * nsp_unit).sum(-1, keepdim=True)
    tang_perp = tang_speed - proj * nsp_unit               # in spatial plane orthogonal to n_{1:}
    tang_mag = tang_perp.norm(dim=-1).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
    sc = ax.scatter(np.log10(np.clip(n_t, 1e-6, 1e2)),
                    np.log10(np.clip(tang_mag, 1e-6, 1e6)),
                    s=4, alpha=0.2, c=torch.sigmoid(opacity_logit).cpu().numpy(),
                    cmap="viridis")
    ax.set_xlabel("log10 |n_0|  (rigid plane velocity proxy)")
    ax.set_ylabel("log10 |tangential drift|")
    plt.colorbar(sc, ax=ax, label="opacity")
    ax.set_title("Motion decomposition: rigid vs tangential")
    plt.tight_layout()
    out3 = os.path.join(out_dir, "diag_motion.png")
    plt.savefig(out3, dpi=110); plt.close()
    print(f"  wrote {out3}")

    # ----- Density-control firing rates from training log -----
    if log_path and os.path.exists(log_path):
        events = []  # list of (iter, split, tsplit, pruned, N)
        pat = re.compile(
            r"\[density @ iter\s+(\d+)\]\s+split=\s*(\d+)\s+tsplit=\s*(\d+)\s+pruned=\s*(\d+)\s+N=(\d+)"
        )
        with open(log_path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    events.append(tuple(int(m.group(i)) for i in range(1, 6)))
        if events:
            arr = np.array(events)
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            ax[0].plot(arr[:, 0], arr[:, 1], "o-", label="split")
            ax[0].plot(arr[:, 0], arr[:, 2], "o-", label="tsplit")
            ax[0].plot(arr[:, 0], arr[:, 3], "o-", label="pruned")
            ax[0].set_xlabel("iter"); ax[0].set_ylabel("count"); ax[0].set_yscale("symlog")
            ax[0].grid(alpha=0.3); ax[0].legend()
            ax[0].set_title("Density-control fire rates per cycle (every 200 iters)")
            ax[1].plot(arr[:, 0], arr[:, 4], "o-", color="C3")
            ax[1].set_xlabel("iter"); ax[1].set_ylabel("N (population)"); ax[1].grid(alpha=0.3)
            ax[1].set_title("Population over time")
            plt.tight_layout()
            out_dc = os.path.join(out_dir, "diag_density_control.png")
            plt.savefig(out_dc, dpi=110); plt.close()
            print(f"  wrote {out_dc}")

    # ----- Per-frame val PSNR + spatial error map -----
    # Skip if dataset isn't pre-loaded; only do this when --scene-dir is given.
    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    diagnostic_main(args.ckpt, args.log, args.out_dir)
