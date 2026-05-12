"""
Time normalization to [0, 1].

Convention used by all datasets and trainers in this repo: time is a scalar in
[0, 1], where 0 is the first frame and 1 is the last frame. The Grassmann
covariance parameter sigma_bb is then in (normalized-time)^2 units, so the
default sigma_bb=0.05 corresponds to a temporal std-dev of about
sqrt(0.05/2) = 0.16 of the timeline (~16% of the sequence). Without this
normalization, passing raw frame-index times (range [0, T-1]) leaves every
Gaussian invisible outside its own frame — see
`results/rca/streak_collapse.md` for the original investigation.

Datasets must emit times in [0, 1]. Use `normalize_times` if your raw data is
in some other unit (frame index, seconds, etc.).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import torch
from torch import Tensor


def normalize_times(
    times: Union[Sequence[float], Tensor, Iterable[float]],
    *,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Tensor:
    """Map an arbitrary time sequence to the canonical [0, 1] range.

    times:  raw times (e.g. frame indices 0..T-1, or seconds).
    t_min:  reference minimum time. Defaults to min(times).
    t_max:  reference maximum time. Defaults to max(times).

    Returns: torch.Tensor of float64, same length as `times`, in [0, 1].

    Edge case: if t_max == t_min (single-frame data), returns zeros. Logging a
    warning is the caller's responsibility.
    """
    if isinstance(times, Tensor):
        t = times.to(dtype=torch.float64).flatten()
    else:
        t = torch.tensor(list(times), dtype=torch.float64)

    if t_min is None:
        t_min = float(t.min().item())
    if t_max is None:
        t_max = float(t.max().item())

    span = t_max - t_min
    if span <= 0.0:
        return torch.zeros_like(t)
    return (t - t_min) / span
