#!/usr/bin/env python
"""
Trigger-vs-distribution audit for the Grassmann density-control loop.

Loads a trained checkpoint and reports, for every threshold-based
trigger, the trigger quantity's distribution alongside the threshold,
so we can spot mismatches like the φ-cascade-zombie + scale_min one.

Usage: python scripts/audit_triggers.py <ckpt.pt> [--logs /tmp/probe_<name>.log]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import torch


def derive(state: dict) -> dict:
    """Compute the same quantities the prune/split code uses."""
    n_raw = state["n_raw"]
    L_raw = state["L_raw"]
    mu = state["mu"]
    opacity_logit = state["opacity_logit"]

    n = n_raw / n_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    nL = (n.unsqueeze(-2) @ L_raw).squeeze(-2)
    L_plane = L_raw - n.unsqueeze(-1) * nL.unsqueeze(-2)
    Sigma_4D = L_plane @ L_plane.transpose(-1, -2)
    Sigma_3D = Sigma_4D[..., 1:, 1:]
    Sigma_tt = Sigma_4D[..., 0, 0]
    eigs_3D = torch.linalg.eigvalsh(Sigma_3D)
    return dict(
        opacity=torch.sigmoid(opacity_logit),
        opacity_logit=opacity_logit,
        Sigma_tt=Sigma_tt,
        lam_kernel=eigs_3D[..., 0],
        lam_mid=eigs_3D[..., 1],
        lam_max=eigs_3D[..., 2],
        L_F=L_raw.flatten(1).norm(dim=-1),
        N=opacity_logit.numel(),
        mu_t=mu[..., 0],
        mu_x=mu[..., 1:],
    )


def histogram(name: str, x: torch.Tensor, threshold: Optional[float] = None,
              direction: str = "below") -> None:
    """Print quantiles + bucket distribution. If threshold given, report
    fraction caught by `x DIR threshold`."""
    qs_levels = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    qs = torch.quantile(x.flatten().to(torch.float32),
                        torch.tensor(qs_levels, dtype=torch.float32))
    print(f"\n=== {name} ===")
    print(f"  N={x.numel()}  min={x.min().item():.5g}  max={x.max().item():.5g}  "
          f"mean={x.mean().item():.5g}")
    for q, v in zip(qs_levels, qs):
        marker = " <==" if (threshold is not None
                            and ((direction == "below" and v.item() < threshold)
                                 or (direction == "above" and v.item() > threshold))) else ""
        print(f"    q{q:5.3f} = {v.item():12.5g}{marker}")
    if threshold is not None:
        if direction == "below":
            n_caught = int((x < threshold).sum().item())
        else:
            n_caught = int((x > threshold).sum().item())
        print(f"  threshold {direction} {threshold:.5g}: caught={n_caught} ({100*n_caught/x.numel():.2f}%)")


def parse_density_logs(log_path: Path) -> dict:
    """Tally split/tsplit/prune fire counts from a probe log."""
    pat = re.compile(
        r"density @ iter\s+(\d+).*?"
        r"(?:split|hybrid split)=\s*(\d+)\s*"
        r"tsplit=\s*(\d+)\s*"
        r"(?:relocated=\s*\d+\s*)?"
        r"pruned=\s*(\d+)"
    )
    splits, tsplits, prunes, n_events = 0, 0, 0, 0
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if m:
            splits += int(m.group(2))
            tsplits += int(m.group(3))
            prunes += int(m.group(4))
            n_events += 1
    return dict(splits=splits, tsplits=tsplits, prunes=prunes, events=n_events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--logs", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    d = derive(state)

    print(f"\n{'='*60}\n  TRIGGER AUDIT — N={d['N']}\n{'='*60}")

    # -- Opacity-related triggers ----------------------------------------
    histogram("opacity (split / prune triggers @ 0.005, 1e-3 threshold)",
              d["opacity"], threshold=1e-3, direction="below")
    print(f"  ALSO: caught below 5e-3 = {int((d['opacity'] < 5e-3).sum())} "
          f"({100*(d['opacity'] < 5e-3).sum().item()/d['N']:.2f}%)")
    print(f"  reset target sigmoid(-5) = 0.0067; reset gives  N at 0.0067")

    # -- Σ_3D eigenvalues (split + collapsed/runaway) --------------------
    histogram("λ_kernel = eigs[0] (rank-2 kernel; ≈ 0)", d["lam_kernel"])
    histogram("λ_mid    = eigs[1] (smaller disk axis; collapsed-prune trigger)",
              d["lam_mid"], threshold=1e-6, direction="below")
    print(f"  ALSO: caught below 5e-3 = {int((d['lam_mid'] < 5e-3).sum())} "
          f"({100*(d['lam_mid'] < 5e-3).sum().item()/d['N']:.2f}%)")
    histogram("λ_max    = eigs[2] (split @ 0.5; runaway-prune @ 100)",
              d["lam_max"], threshold=0.5, direction="above")
    n_runaway = int((d["lam_max"] > 100.0).sum().item())
    print(f"  runaway @ 100: {n_runaway} ({100*n_runaway/d['N']:.4f}%)")
    print(f"  spatial_split @ 0.5 (size gate, AND'd with grad-threshold)")

    # -- Aspect ratio (relevant for lambda_aniso effect) ---------------
    aspect = (d["lam_max"] / d["lam_mid"].clamp_min(1e-12))
    histogram("aspect = λ_max / λ_mid (lambda_aniso pressure)", aspect,
              threshold=10.0, direction="above")

    # -- Σ_tt (temporal-split trigger) ---------------------------------
    histogram("Σ_tt (temporal-split trigger @ 0.1)", d["Sigma_tt"],
              threshold=0.1, direction="above")

    # -- |L_raw|_F (overall scale; relevant for split-shrink-cascade) --
    histogram("|L_raw|_F (overall geom scale; cascade victim)", d["L_F"])

    # -- Time-axis position (Σ_tt blur radius vs |t-spread|) -----------
    print(f"\nμ_t spread: min={d['mu_t'].min():.4f}  max={d['mu_t'].max():.4f}  "
          f"mean={d['mu_t'].mean():.4f}")
    print(f"  scene t in [0,1] (NeRFies normalization)")

    # -- Population grouping (alive / zombie / dead) -------------------
    alive = d["opacity"] > 0.5
    zombie = (d["opacity"] > 0.005) & (d["opacity"] <= 0.01)
    dead = d["opacity"] <= 0.005
    print(f"\nPopulations: alive={alive.sum().item()} ({100*alive.sum().item()/d['N']:.1f}%)  "
          f"zombie={zombie.sum().item()} ({100*zombie.sum().item()/d['N']:.1f}%)  "
          f"dead={dead.sum().item()} ({100*dead.sum().item()/d['N']:.1f}%)")
    if zombie.sum() > 0 and alive.sum() > 0:
        print(f"  zombie/alive  λ_mid q50:  "
              f"{d['lam_mid'][zombie].median().item():.5f} / "
              f"{d['lam_mid'][alive].median().item():.5f}  "
              f"(ratio = {d['lam_mid'][alive].median().item()/d['lam_mid'][zombie].median().item():.1f}×)")
        print(f"  zombie/alive  Σ_tt  q50:  "
              f"{d['Sigma_tt'][zombie].median().item():.5f} / "
              f"{d['Sigma_tt'][alive].median().item():.5f}")
        print(f"  zombie/alive  |L|_F q50:  "
              f"{d['L_F'][zombie].median().item():.5f} / "
              f"{d['L_F'][alive].median().item():.5f}  "
              f"(ratio = {d['L_F'][alive].median().item()/d['L_F'][zombie].median().item():.1f}×)")

    # -- Density-event log audit --------------------------------------
    if args.logs:
        print(f"\n{'='*60}\n  DENSITY-EVENT FIRE RATES (from logs)\n{'='*60}")
        print(f"  {'log':<50} {'events':>7} {'splits':>9} {'tsplits':>8} {'prunes':>8}")
        for lp in args.logs:
            if not lp.exists():
                continue
            r = parse_density_logs(lp)
            name = lp.name.replace("probe_", "").replace(".log", "")[:50]
            print(f"  {name:<50} {r['events']:>7} {r['splits']:>9} "
                  f"{r['tsplits']:>8} {r['prunes']:>8}")


if __name__ == "__main__":
    main()
