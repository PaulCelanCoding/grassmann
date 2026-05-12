"""
Grassmannian primitives: the canonical plane E_{p,q} and the correspondence
between oriented affine lines in R^3 and pairs (p, q) in S^2 x S^2.

References:
  - Grassmann paper, Section 2 (Preliminaries)
  - Jacobian paper, Proposition 1 (orthogonal basis) and Definition 2 (shorthands)

All p, q inputs are expected to be *unit imaginary* quaternions, shape (..., 4)
with p[..., 0] == 0 and |p[..., 1:]| == 1. Use quaternion.unit_imag() to build them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from . import quaternion as Q


# ---- The canonical plane E_{p,q} ---------------------------------------------

@dataclass
class CanonicalFrame:
    """Geometric shorthands for the plane E_{p,q}.

    All tensors are batched: shape (..., 3) for vectors, (...,) for scalars.

    Fields (matching Jacobian paper Definition 2):
      c = p . q                       (scalar, in [-1, 1])
      d = p + q                       (3-vector)
      s = p x q                       (3-vector, perpendicular to d)
      r = 1 / sqrt(2(1 + c))         (scalar, singular at p = -q)

    The orthogonal basis of E_{p,q} is:
      e1 = (0, d)                     (purely spatial)
      e2 = (1 + c, -s)                (mixes time and space)
    with |e1|^2 = |e2|^2 = 2(1 + c).
    """
    c: Tensor       # (...,)
    d: Tensor       # (..., 3)
    s: Tensor       # (..., 3)
    r: Tensor       # (...,)


def canonical_frame(p: Tensor, q: Tensor, eps: float = 1e-8) -> CanonicalFrame:
    """Compute (c, d, s, r) from unit imaginary quaternions p, q.

    p, q have shape (..., 4) with zero real part.
    The antidiagonal p = -q (c -> -1) gives r -> infinity; eps protects against that.
    """
    p_im = Q.imag(p)  # (..., 3)
    q_im = Q.imag(q)  # (..., 3)

    c = (p_im * q_im).sum(dim=-1)                       # (...,)
    d = p_im + q_im                                     # (..., 3)
    s = torch.cross(p_im, q_im, dim=-1)                 # (..., 3)
    two_one_plus_c = (2.0 * (1.0 + c)).clamp_min(eps)
    r = 1.0 / torch.sqrt(two_one_plus_c)                # (...,)
    return CanonicalFrame(c=c, d=d, s=s, r=r)


def basis_e1_e2(p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    """Return the (unnormalized) orthogonal basis (e1, e2) of E_{p,q}.

    e1 = p + q         (as a quaternion: real=0, imag=d)
    e2 = 1 - p*q       (as a quaternion: real=1+c, imag=-s)

    Shapes: each is (..., 4).
    """
    f = canonical_frame(p, q)
    e1 = Q.from_real_imag(torch.zeros_like(f.c), f.d)
    e2 = Q.from_real_imag(1.0 + f.c, -f.s)
    return e1, e2


def orthonormal_basis(p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    """Return the orthonormal basis (e1_hat, e2_hat) of E_{p,q}.

    Each has shape (..., 4), with norm 1 (up to the antidiagonal singularity).
    """
    e1, e2 = basis_e1_e2(p, q)
    f = canonical_frame(p, q)
    r = f.r.unsqueeze(-1)     # broadcast scalar over the last dim of the quaternion
    return r * e1, r * e2


# ---- Line <-> (p, q) correspondence -----------------------------------------
#
# A directed affine line in R^3 is L_{x, u} = { x + lambda*u : lambda in R }.
# At time t = 1, the Grassmann paper embeds L via
#     phi_1(L_{x,u}) = span_R{ (1, x),  (0, u) }   in H = R^4.
# (In general time t, we embed into span{ t(1, x), u }, but since this is a span,
#  scaling by t doesn't change the plane itself. Only the orientation depends on sign(t).)
#
# The embedding is a 2-plane; it corresponds to a pair (p, q) in S^2 x S^2 via
#     E_{p, q} = { z in H : p*z = z*q }.
#
# REMARK ON THE PAPER'S FORMULA.
# The Grassmann paper writes the formula
#     p = ytu^{-1} / | . |,   q = u^{-1}yt / | . |
# where y is x brought to standard form (y perp u). We tested this formula
# explicitly and found it produces p = -q (the antidiagonal) for *every* input,
# which contradicts Lemma 2.1 (image avoids the antidiagonal). We therefore use
# the following derivation instead, which we verified yields the correct plane.
#
# DERIVATION.
# Parameterize p = m + n, q = m - n where m = (p+q)/2, n = (p-q)/2.
# Unit norm + orthogonality give |m|^2 + |n|^2 = 1 and m . n = 0.
# A basis of E_{p,q} is { (0, p+q), (1+c, -(p x q)) } = { (0, 2m), (1+c, 2 m x n) }.
# For this to span { (1, y), (0, u) } with y perp u:
#   * (0, 2m) ∝ (0, u)              => m = lambda * u_hat, lambda = |m|
#   * real part 1+c comes from (1, y), so (1+c) * y - 2(m x n) must be in span{u}.
#     Since y perp u, we need 2(m x n) to match (1+c) y (no u-component survives
#     because y perp u and m x n perp m = lambda u_hat).
# This yields (by straightforward algebra):
#     lambda^2 = 1 / (1 + |y|^2),      c = (1 - |y|^2) / (1 + |y|^2)
#     m = lambda * u_hat
#     n = -lambda * (u_hat x y)
#     p = m + n = lambda (u_hat - u_hat x y)
#     q = m - n = lambda (u_hat + u_hat x y)
#
# We verified this sends { (1, y), (0, u) } into E_{p, q} to machine precision.


def line_standard_form(x: Tensor, u: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Project the line L_{x, u} to standard form L_{y, u_hat} with <y, u_hat> = 0.

    x, u have shape (..., 3). Returns (y, u_hat) with the same shapes.
    y is the foot of perpendicular from the origin onto the line.
    """
    u_hat = u / u.norm(dim=-1, keepdim=True).clamp_min(eps)
    proj = (x * u_hat).sum(dim=-1, keepdim=True) * u_hat
    y = x - proj
    return y, u_hat


def line_to_pq(x: Tensor, u: Tensor, t: float = 1.0) -> tuple[Tensor, Tensor]:
    """Map a directed line L_{x, u} in R^3 to a pair (p, q) of unit imaginary quaternions.

    The t parameter controls the orientation of the plane (its sign flips the
    ordered basis); since we produce unoriented (p, q) pairs, t only affects
    the output via its sign. For t < 0 we flip the roles of p and q, which
    reverses the orientation of E_{p, q}.

    x, u have shape (..., 3). Returns p, q each shape (..., 4), purely imaginary unit-norm.
    """
    assert t != 0.0, "t must be nonzero"
    y, u_hat = line_standard_form(x, u)

    y_norm_sq = (y * y).sum(dim=-1, keepdim=True)        # (..., 1)
    lam = 1.0 / torch.sqrt(1.0 + y_norm_sq)              # (..., 1)

    # p_im = lambda * (u_hat - u_hat x y),  q_im = lambda * (u_hat + u_hat x y)
    cross_uy = torch.cross(u_hat, y, dim=-1)             # (..., 3)
    p_im = lam * (u_hat - cross_uy)
    q_im = lam * (u_hat + cross_uy)

    p = Q.pure_imag(p_im)
    q = Q.pure_imag(q_im)

    # If t < 0, swap p and q (reversed orientation of the plane).
    if t < 0.0:
        p, q = q, p

    return p, q


def pq_to_line(p: Tensor, q: Tensor, t: float = 1.0) -> tuple[Tensor, Tensor]:
    """Recover a directed line (y, u_hat) from (p, q).

    Inverse of line_to_pq (standard form, so y is perpendicular to u_hat).

    From the derivation:
        u_hat = (p_im + q_im) / |p_im + q_im|   (= d / |d|)
        y = (q_im - p_im) x u_hat / lambda,     where lambda = |p_im + q_im| / 2
                                                             = 1 / sqrt(1 + |y|^2)

    Equivalently, since q_im - p_im = -2n = 2 lambda (u_hat x y):
        y = u_hat x (q_im - p_im) / lambda * (1/2) ... we derive below.

    The cleanest closed form:
        d = p_im + q_im = 2 m = 2 lambda u_hat
        |d|^2 = 4 lambda^2 = 4 / (1 + |y|^2)
        so  1 + |y|^2 = 4 / |d|^2,  |y|^2 = 4/|d|^2 - 1.
        Also  q_im - p_im = 2 lambda (u_hat x y).
        So  u_hat x y = (q_im - p_im) / (2 lambda) = (q_im - p_im) / |d|.
        Finally  y = u_hat x (u_hat x y) x u_hat ... easier: y = -(u_hat x ((q_im - p_im)/|d|)),
                 using  u x (u x v) = -v   when v perp u.

    Returns (y, u_hat) each shape (..., 3). For t < 0, this returns the same
    LINE but orientation is reversed externally by caller.
    """
    p_im = Q.imag(p)
    q_im = Q.imag(q)
    # If t < 0 was used on the forward map, p and q were swapped; swap back here.
    if t < 0.0:
        p_im, q_im = q_im, p_im

    d = p_im + q_im                                             # (..., 3)
    d_norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u_hat = d / d_norm                                          # (..., 3)

    # u_hat x y = (q_im - p_im) / |d|
    ucy = (q_im - p_im) / d_norm
    # y = -(u_hat x ucy)   because u_hat x (u_hat x y) = -y  when y perp u_hat.
    y = -torch.cross(u_hat, ucy, dim=-1)

    return y, u_hat
