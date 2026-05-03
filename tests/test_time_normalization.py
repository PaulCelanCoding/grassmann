"""Tests for grassmann.time_normalization."""
from __future__ import annotations

import math

import torch

from grassmann.time_normalization import normalize_times


def test_frame_indices_map_to_unit_interval():
    out = normalize_times(range(10))
    assert out[0].item() == 0.0
    assert math.isclose(out[-1].item(), 1.0, abs_tol=1e-12)
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_arbitrary_seconds_normalize():
    out = normalize_times([1.5, 2.5, 3.5])
    assert math.isclose(out[0].item(), 0.0, abs_tol=1e-12)
    assert math.isclose(out[1].item(), 0.5, abs_tol=1e-12)
    assert math.isclose(out[2].item(), 1.0, abs_tol=1e-12)


def test_explicit_t_min_t_max():
    out = normalize_times([5.0, 10.0], t_min=0.0, t_max=20.0)
    assert math.isclose(out[0].item(), 0.25, abs_tol=1e-12)
    assert math.isclose(out[1].item(), 0.5, abs_tol=1e-12)


def test_single_frame_returns_zero():
    out = normalize_times([7.0])
    assert out.shape == (1,)
    assert out[0].item() == 0.0


def test_tensor_input_accepted():
    out = normalize_times(torch.arange(5.0))
    assert out[-1].item() == 1.0
    assert out.dtype == torch.float64


def test_default_sigma_bb_resolves_adjacent_frames():
    """RCA Bug C: with normalized times in [0,1] and sigma_bb=0.05, the adjacent-
    frame temporal weight should be ~1 (not collapse to 0 like with frame-index times)."""
    from grassmann.gaussian import GaussianParams, compute_derived, condition_on_time

    n_frames = 300
    times = normalize_times(range(n_frames))

    # Single Gaussian. Pick p = q so c = 1 -> e2_hat is purely temporal,
    # so beta_0 directly equals the temporal mean v_0.
    p_im = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    q_im = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    L = torch.tensor([[[math.sqrt(0.02), 0.0], [0.0, math.sqrt(0.05)]]], dtype=torch.float64)
    params = GaussianParams(
        p_im=p_im, q_im=q_im,
        alpha_0=torch.tensor([0.0], dtype=torch.float64),
        beta_0=torch.tensor([float(times[150])], dtype=torch.float64),
        L=L,
        opacity=torch.tensor([1.0], dtype=torch.float64),
        color=torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64),
        sigma_k_pixel=1.0,
        sigma_k_temporal=0.0,
    )
    derived = compute_derived(params)
    # At v_0 itself: w_t = 1.
    tc_on = condition_on_time(params, derived, t_0=float(times[150]))
    assert math.isclose(tc_on.w_t.item(), 1.0, abs_tol=1e-10)
    # At adjacent frame: dt = 1/299 ~ 0.0033. With c=1, Sigma_tt = sigma_bb = 0.05,
    # std ~ 0.224 -> w_t = exp(-0.5 * 0.0033^2 / 0.05) ~ 0.9998.
    tc_off = condition_on_time(params, derived, t_0=float(times[151]))
    assert tc_off.w_t.item() > 0.99, f"w_t at adjacent frame is {tc_off.w_t.item():.4f}; expected >0.99"
    # And at the OPPOSITE end of the timeline (~0.5 normalized away), it must
    # decay (else sigma_bb would be too large). exp(-0.5 * 0.5^2 / 0.05) ~ 0.082.
    tc_far = condition_on_time(params, derived, t_0=float(times[0]))
    assert tc_far.w_t.item() < 0.5, f"w_t far away is {tc_far.w_t.item():.4f}; expected <0.5"
