"""
Quaternion arithmetic as batched PyTorch operations.

Convention: a quaternion x = x0 + x1*i + x2*j + x3*k is stored as a tensor
with shape (..., 4) in the order (x0, x1, x2, x3) = (real, i, j, k).

This matches the convention used in both papers:
  - x0 is the "time" / real component
  - (x1, x2, x3) is the "spatial" / imaginary component
"""
from __future__ import annotations

import torch
from torch import Tensor


# ---- Component accessors -----------------------------------------------------

def real(x: Tensor) -> Tensor:
    """Return the real (x0) component, shape (...,)."""
    return x[..., 0]


def imag(x: Tensor) -> Tensor:
    """Return the imaginary/spatial (x1, x2, x3) component, shape (..., 3)."""
    return x[..., 1:]


def from_real_imag(r: Tensor, v: Tensor) -> Tensor:
    """Assemble a quaternion from a scalar real part and 3-vector imaginary part.

    r: shape (...,)  or scalar
    v: shape (..., 3)
    returns: shape (..., 4)
    """
    if r.dim() == 0:
        r = r.expand(v.shape[:-1])
    return torch.cat([r.unsqueeze(-1), v], dim=-1)


def pure_imag(v: Tensor) -> Tensor:
    """Build a purely imaginary quaternion from a 3-vector: x0 = 0."""
    zero = torch.zeros(v.shape[:-1] + (1,), dtype=v.dtype, device=v.device)
    return torch.cat([zero, v], dim=-1)


# ---- Core arithmetic ---------------------------------------------------------

def mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product a * b for batched quaternions.

    Formula: if a = (a0, A) and b = (b0, B) with A, B in R^3, then
        a*b = (a0*b0 - A.B,  a0*B + b0*A + A x B)
    """
    a0, A = a[..., 0:1], a[..., 1:]
    b0, B = b[..., 0:1], b[..., 1:]
    real_part = a0 * b0 - (A * B).sum(dim=-1, keepdim=True)
    imag_part = a0 * B + b0 * A + torch.cross(A, B, dim=-1)
    return torch.cat([real_part, imag_part], dim=-1)


def conj(x: Tensor) -> Tensor:
    """Quaternion conjugate: (x0, X) -> (x0, -X)."""
    out = x.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def norm_sq(x: Tensor) -> Tensor:
    """Squared norm, shape (...,)."""
    return (x * x).sum(dim=-1)


def norm(x: Tensor) -> Tensor:
    """Euclidean norm, shape (...,)."""
    return torch.linalg.norm(x, dim=-1)


def inverse(x: Tensor) -> Tensor:
    """Quaternion inverse: conj(x) / |x|^2."""
    return conj(x) / norm_sq(x).unsqueeze(-1)


def normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize to unit quaternion."""
    n = norm(x).clamp_min(eps).unsqueeze(-1)
    return x / n


# ---- Convenience: unit imaginary quaternions from R^3 unit vectors ----------

def unit_imag(v: Tensor, eps: float = 1e-12) -> Tensor:
    """Given v in R^3, return the pure-imaginary unit quaternion (0, v/|v|).

    This is the form used for p, q in S^2 subset of Im(H).
    """
    v_n = v / v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return pure_imag(v_n)
