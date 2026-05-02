"""Unit tests for the Grassmannian module.

These tests verify the math from:
  - Jacobian paper, Proposition 1: e1, e2 form an orthogonal basis of E_{p,q}.
  - Grassmann paper, Section 2: the line <-> (p, q) correspondence.
"""
import pytest
import torch

from grassmann import quaternion as Q
from grassmann import grassmann as G


torch.manual_seed(1234)


def rand_pq(n=8, min_sep=0.1):
    """Random pairs (p, q) of unit imaginary quaternions, avoiding p ~ -q.

    Reject pairs with c = p.q < -1 + min_sep (near antidiagonal).
    """
    ps = []
    qs = []
    while len(ps) < n:
        p_vec = torch.randn(3)
        q_vec = torch.randn(3)
        p_vec = p_vec / p_vec.norm()
        q_vec = q_vec / q_vec.norm()
        c = (p_vec * q_vec).sum().item()
        if c > -1 + min_sep:
            ps.append(Q.unit_imag(p_vec))
            qs.append(Q.unit_imag(q_vec))
    return torch.stack(ps), torch.stack(qs)


# ---- Proposition 1: e1, e2 form an orthogonal basis of E_{p,q} ---------------

def test_e1_e2_lie_in_Epq():
    """Points x in E_{p,q} satisfy p*x = x*q."""
    p, q = rand_pq(16)
    e1, e2 = G.basis_e1_e2(p, q)

    # Check p*e1 = e1*q
    lhs1 = Q.mul(p, e1)
    rhs1 = Q.mul(e1, q)
    assert torch.allclose(lhs1, rhs1, atol=1e-5), f"e1 not in E_pq: max err = {(lhs1 - rhs1).abs().max()}"

    # Check p*e2 = e2*q
    lhs2 = Q.mul(p, e2)
    rhs2 = Q.mul(e2, q)
    assert torch.allclose(lhs2, rhs2, atol=1e-5), f"e2 not in E_pq: max err = {(lhs2 - rhs2).abs().max()}"


def test_e1_e2_orthogonal():
    """<e1, e2> = 0 in R^4."""
    p, q = rand_pq(16)
    e1, e2 = G.basis_e1_e2(p, q)
    inner = (e1 * e2).sum(dim=-1)
    assert torch.allclose(inner, torch.zeros_like(inner), atol=1e-6)


def test_e1_e2_equal_norm():
    """|e1|^2 = |e2|^2 = 2(1 + c)."""
    p, q = rand_pq(16)
    e1, e2 = G.basis_e1_e2(p, q)
    f = G.canonical_frame(p, q)
    expected = 2.0 * (1.0 + f.c)

    assert torch.allclose(Q.norm_sq(e1), expected, atol=1e-5)
    assert torch.allclose(Q.norm_sq(e2), expected, atol=1e-5)


def test_orthonormal_basis_unit_norm():
    """The normalized basis has unit norm."""
    p, q = rand_pq(16)
    e1h, e2h = G.orthonormal_basis(p, q)
    assert torch.allclose(Q.norm(e1h), torch.ones(16), atol=1e-5)
    assert torch.allclose(Q.norm(e2h), torch.ones(16), atol=1e-5)


def test_e1_purely_spatial():
    """e1 = (0, d) has no time component."""
    p, q = rand_pq(16)
    e1, _ = G.basis_e1_e2(p, q)
    assert torch.allclose(Q.real(e1), torch.zeros(16), atol=1e-6)


def test_e2_time_component():
    """e2 = (1+c, -s). Time component equals 1+c."""
    p, q = rand_pq(16)
    _, e2 = G.basis_e1_e2(p, q)
    f = G.canonical_frame(p, q)
    assert torch.allclose(Q.real(e2), 1.0 + f.c, atol=1e-5)


# ---- Sanity: a point in span{e1, e2} is in E_{p,q} --------------------------

def test_span_lies_in_Epq():
    """Any linear combination alpha*e1 + beta*e2 satisfies p*x = x*q."""
    p, q = rand_pq(8)
    e1, e2 = G.basis_e1_e2(p, q)
    for _ in range(5):
        alpha = torch.randn(8, 1)
        beta = torch.randn(8, 1)
        x = alpha * e1 + beta * e2
        lhs = Q.mul(p, x)
        rhs = Q.mul(x, q)
        assert torch.allclose(lhs, rhs, atol=1e-5)


# ---- Line <-> (p, q) correspondence -----------------------------------------

def test_line_to_pq_produces_unit_imag():
    """The output p, q should be unit imaginary quaternions."""
    x = torch.randn(10, 3)
    u = torch.randn(10, 3)
    p, q = G.line_to_pq(x, u, t=1.0)
    assert torch.allclose(Q.real(p), torch.zeros(10), atol=1e-6)
    assert torch.allclose(Q.real(q), torch.zeros(10), atol=1e-6)
    assert torch.allclose(Q.norm(p), torch.ones(10), atol=1e-5)
    assert torch.allclose(Q.norm(q), torch.ones(10), atol=1e-5)


def test_line_roundtrip():
    """line -> (p, q) -> line recovers the same affine line.

    Since line_to_pq normalizes to standard form, pq_to_line returns (y, u_hat)
    where y is the foot-of-perpendicular point. The recovered line must be
    equal as a SET to the original: the direction agrees, and the recovered
    point lies on the original line.
    """
    torch.manual_seed(0)
    x = torch.randn(20, 3)
    u = torch.randn(20, 3)
    u = u / u.norm(dim=-1, keepdim=True)
    t = 1.0

    p, q = G.line_to_pq(x, u, t=t)
    y_rec, u_rec = G.pq_to_line(p, q, t=t)

    # Direction: recovered exactly (sign preserved because the derivation fixes orientation)
    assert torch.allclose(u_rec, u, atol=1e-5), f"u not recovered; max err = {(u_rec-u).abs().max()}"

    # Recovered point y_rec must lie on the original line: y_rec - x || u.
    # Equivalently, (y_rec - x) x u = 0.
    diff = y_rec - x
    cross = torch.cross(diff, u, dim=-1)
    cross_norm = cross.norm(dim=-1)
    assert cross_norm.max() < 1e-4, f"recovered point not on line; max cross norm = {cross_norm.max()}"

    # Additionally: y_rec should be in standard form (perpendicular to u).
    dot = (y_rec * u_rec).sum(dim=-1).abs()
    assert dot.max() < 1e-5, f"y_rec not perpendicular to u_rec; max dot = {dot.max()}"


def test_line_roundtrip_orientation_flip():
    """t < 0 should reverse the orientation, i.e. swap p and q."""
    torch.manual_seed(1)
    x = torch.randn(10, 3)
    u = torch.randn(10, 3)
    u = u / u.norm(dim=-1, keepdim=True)

    p_pos, q_pos = G.line_to_pq(x, u, t=1.0)
    p_neg, q_neg = G.line_to_pq(x, u, t=-1.0)

    assert torch.allclose(p_neg, q_pos, atol=1e-5)
    assert torch.allclose(q_neg, p_pos, atol=1e-5)


def test_line_roundtrip_different_times():
    """For t with the same sign, the recovered line is identical."""
    x = torch.randn(10, 3)
    u = torch.randn(10, 3)
    u = u / u.norm(dim=-1, keepdim=True)

    for t in [0.5, 1.0, 2.0]:
        p, q = G.line_to_pq(x, u, t=t)
        y_rec, u_rec = G.pq_to_line(p, q, t=t)

        # Direction recovered
        assert torch.allclose(u_rec, u, atol=1e-5)

        # Point lies on original line
        diff = y_rec - x
        cross_norm = torch.cross(diff, u, dim=-1).norm(dim=-1)
        assert cross_norm.max() < 1e-4


def test_embedding_into_canonical_plane():
    """For a line L = {x + lambda u}, the image phi_1(L) = span{(1, y), (0, u_hat)}
    must lie inside E_{p, q}, where (p, q) = line_to_pq(x, u). This is the
    DEFINING property of the embedding (Grassmann paper, Section 2).
    """
    torch.manual_seed(7)
    x = torch.randn(20, 3)
    u = torch.randn(20, 3)
    u = u / u.norm(dim=-1, keepdim=True)

    p, q = G.line_to_pq(x, u, t=1.0)
    y, u_hat = G.line_standard_form(x, u)

    # Basis point 1: (1, y)
    ones = torch.ones(20, 1)
    z1 = torch.cat([ones, y], dim=-1)
    # Basis point 2: (0, u_hat)
    zeros = torch.zeros(20, 1)
    z2 = torch.cat([zeros, u_hat], dim=-1)

    err1 = (Q.mul(p, z1) - Q.mul(z1, q)).abs().max()
    err2 = (Q.mul(p, z2) - Q.mul(z2, q)).abs().max()
    assert err1 < 1e-5, f"(1, y) not in E_pq: err = {err1}"
    assert err2 < 1e-5, f"(0, u_hat) not in E_pq: err = {err2}"


def test_c_matches_derived_formula():
    """The derived formula gives c = (1 - |y|^2) / (1 + |y|^2)."""
    torch.manual_seed(11)
    x = torch.randn(15, 3)
    u = torch.randn(15, 3)

    p, q = G.line_to_pq(x, u, t=1.0)
    f = G.canonical_frame(p, q)

    y, _ = G.line_standard_form(x, u)
    y_norm_sq = (y * y).sum(dim=-1)
    expected_c = (1.0 - y_norm_sq) / (1.0 + y_norm_sq)

    assert torch.allclose(f.c, expected_c, atol=1e-5), \
        f"c mismatch; max err = {(f.c - expected_c).abs().max()}"


def test_antidiagonal_excluded():
    """By Lemma 2.1, the image of the line embedding avoids the antidiagonal
    A = {(p, -p) : p in S^2}. With our derived formula, c = (1-|y|^2)/(1+|y|^2)
    which is always > -1 for finite |y|; c -> -1 only as |y| -> infinity
    (line infinitely far from origin). For random finite inputs, c should be
    well above -1.
    """
    torch.manual_seed(99)
    x = torch.randn(100, 3)
    u = torch.randn(100, 3)
    p, q = G.line_to_pq(x, u, t=1.0)
    f = G.canonical_frame(p, q)
    # c = (1 - |y|^2)/(1 + |y|^2): for |y| <= ~10, c > -0.98.
    assert (f.c > -0.99).all(), f"some c too close to -1: min = {f.c.min()}"


def test_canonical_frame_shapes():
    """Batch shapes propagate."""
    p, q = rand_pq(12)
    f = G.canonical_frame(p, q)
    assert f.c.shape == (12,)
    assert f.d.shape == (12, 3)
    assert f.s.shape == (12, 3)
    assert f.r.shape == (12,)


def test_s_perpendicular_to_d():
    """s = p x q is perpendicular to d = p + q.
    (Because p x q is perpendicular to both p and q individually.)
    This is used implicitly in the orthogonality proof.
    """
    p, q = rand_pq(16)
    f = G.canonical_frame(p, q)
    dot = (f.d * f.s).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
