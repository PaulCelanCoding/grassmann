"""Unit tests for the quaternion module."""
import pytest
import torch

from grassmann import quaternion as Q


torch.manual_seed(42)


def rand_quat(shape=(), device="cpu"):
    """Random non-zero quaternion."""
    x = torch.randn(*shape, 4, device=device)
    return x


def rand_unit_quat(shape=(), device="cpu"):
    return Q.normalize(rand_quat(shape, device))


# ---- Accessors and constructors ---------------------------------------------

def test_real_imag_roundtrip():
    x = rand_quat((5,))
    r, v = Q.real(x), Q.imag(x)
    assert r.shape == (5,)
    assert v.shape == (5, 3)
    x2 = Q.from_real_imag(r, v)
    assert torch.allclose(x, x2)


def test_pure_imag():
    v = torch.randn(3, 3)
    x = Q.pure_imag(v)
    assert torch.allclose(Q.real(x), torch.zeros(3))
    assert torch.allclose(Q.imag(x), v)


# ---- Hamilton product is associative and satisfies i^2 = j^2 = k^2 = ijk = -1

def test_i_squared_is_minus_one():
    one = torch.tensor([1.0, 0.0, 0.0, 0.0])
    i = torch.tensor([0.0, 1.0, 0.0, 0.0])
    j = torch.tensor([0.0, 0.0, 1.0, 0.0])
    k = torch.tensor([0.0, 0.0, 0.0, 1.0])
    minus_one = -one

    assert torch.allclose(Q.mul(i, i), minus_one)
    assert torch.allclose(Q.mul(j, j), minus_one)
    assert torch.allclose(Q.mul(k, k), minus_one)
    # ij = k, jk = i, ki = j
    assert torch.allclose(Q.mul(i, j), k)
    assert torch.allclose(Q.mul(j, k), i)
    assert torch.allclose(Q.mul(k, i), j)
    # ijk = -1
    assert torch.allclose(Q.mul(Q.mul(i, j), k), minus_one)


def test_multiplication_associative():
    a = rand_quat((10,))
    b = rand_quat((10,))
    c = rand_quat((10,))
    left = Q.mul(Q.mul(a, b), c)
    right = Q.mul(a, Q.mul(b, c))
    assert torch.allclose(left, right, atol=1e-6)


def test_multiplication_not_commutative():
    # Generic case: i*j != j*i
    i = torch.tensor([0.0, 1.0, 0.0, 0.0])
    j = torch.tensor([0.0, 0.0, 1.0, 0.0])
    assert not torch.allclose(Q.mul(i, j), Q.mul(j, i))


# ---- Norm, conjugate, inverse ------------------------------------------------

def test_conj_involution():
    x = rand_quat((5,))
    assert torch.allclose(Q.conj(Q.conj(x)), x)


def test_norm_multiplicative():
    """|a*b| = |a| * |b|."""
    a = rand_quat((5,))
    b = rand_quat((5,))
    lhs = Q.norm(Q.mul(a, b))
    rhs = Q.norm(a) * Q.norm(b)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_inverse_identity():
    """x * x^{-1} = 1."""
    x = rand_quat((5,))
    one = torch.tensor([1.0, 0.0, 0.0, 0.0])
    prod = Q.mul(x, Q.inverse(x))
    assert torch.allclose(prod, one.expand_as(prod), atol=1e-5)
    prod2 = Q.mul(Q.inverse(x), x)
    assert torch.allclose(prod2, one.expand_as(prod2), atol=1e-5)


def test_unit_imag_is_unit_and_pure():
    v = torch.randn(7, 3)
    x = Q.unit_imag(v)
    assert torch.allclose(Q.real(x), torch.zeros(7), atol=1e-6)
    assert torch.allclose(Q.norm(x), torch.ones(7), atol=1e-6)


def test_pure_unit_imag_squares_to_minus_one():
    """For a pure imaginary unit quaternion p, p^2 = -1. This is the identity
    used in the Jacobian paper's proof of Proposition 1."""
    v = torch.randn(5, 3)
    p = Q.unit_imag(v)
    sq = Q.mul(p, p)
    minus_one = torch.tensor([-1.0, 0.0, 0.0, 0.0]).expand_as(sq)
    assert torch.allclose(sq, minus_one, atol=1e-5)


# ---- Batch shape broadcasting -----------------------------------------------

def test_batch_shapes():
    x = rand_quat((3, 5))
    y = rand_quat((3, 5))
    z = Q.mul(x, y)
    assert z.shape == (3, 5, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
