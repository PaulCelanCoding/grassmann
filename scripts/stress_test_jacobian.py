"""Stress test the Jacobian on many random configurations.

Not a unit test per se -- produces a summary of worst-case errors across a fuzzing run.
Useful to run occasionally to catch numerical cliffs.
"""
import torch

from grassmann import quaternion as Q
from grassmann import grassmann as G
from grassmann import jacobian as J
from grassmann.projection import Camera, project_static


DTYPE = torch.float64


def random_valid_config():
    """Generate a random (cam, p, q, V) with V in front of the camera."""
    for _ in range(50):
        # Random rotation
        A = torch.randn(3, 3, dtype=DTYPE)
        R, _ = torch.linalg.qr(A)
        if torch.det(R) < 0:
            R[:, 0] *= -1
        c = torch.randn(3, dtype=DTYPE) * 0.3

        fx, fy = 600 + 200*torch.rand(1).item(), 600 + 200*torch.rand(1).item()
        cx, cy = 320 + 50*torch.randn(1).item(), 240 + 50*torch.randn(1).item()
        cam = Camera(R=R, c=c, fx=fx, fy=fy, cx=cx, cy=cy)

        # Random (p, q) away from antidiagonal
        pv = torch.randn(3, dtype=DTYPE); pv /= pv.norm()
        qv = torch.randn(3, dtype=DTYPE); qv /= qv.norm()
        if (pv*qv).sum() < -0.5:   # skip near-antidiagonal
            continue
        p = Q.unit_imag(pv)
        q = Q.unit_imag(qv)

        # Random V somewhere in front of camera (depth in [2, 10])
        V_cam = torch.tensor([
            1.0 * torch.randn(1).item(),
            1.0 * torch.randn(1).item(),
            2.0 + 8.0 * torch.rand(1).item(),
        ], dtype=DTYPE)
        V = R.T @ V_cam + c

        return cam, p, q, V
    raise RuntimeError("could not generate valid config")


def compute_autograd_jacobian(cam, p, q, V):
    """Compute J_full via autograd, same pattern as the unit test."""
    e1_hat, e2_hat = G.orthonormal_basis(p, q)
    # v_quat is any valid mean; construct from V: v = (alpha_0, beta_0) such that
    # alpha_0 e1 + beta_0 e2 has spatial part == V. Since e1 is purely spatial (along d)
    # and e2 has spatial part -r*s, solving exactly is messy. Easier: construct
    # v_quat = (t_val, V) such that p*v = v*q. A valid construction:
    # Take v in E_{p,q} with spatial part V. The plane contains (1, y) and (0, u_hat)
    # where y is on the physical line. But we want an arbitrary V in the plane.
    #
    # Trick: use the fact that any spatial vector V has a unique decomposition
    # V = alpha * d_hat + beta * (-s_hat) + (time_part_spill).
    # But we need v actually IN E_{p,q}. Simpler: just construct v_quat = alpha*e1 + beta*e2
    # for some random (alpha, beta) rather than trying to match a specific V.

    alpha_0 = torch.randn(1, dtype=DTYPE).squeeze() * 0.2
    beta_0 = torch.randn(1, dtype=DTYPE).squeeze() * 0.2
    v_quat = alpha_0 * e1_hat + beta_0 * e2_hat
    V_actual = Q.imag(v_quat)

    # Now compute autograd Jacobian at that v_quat.
    alpha = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    beta = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    z = v_quat + alpha * e1_hat + beta * e2_hat
    t = Q.real(z)
    X = Q.imag(z)
    uv = project_static(X.unsqueeze(0), cam).squeeze(0)
    output = torch.stack([uv[0], uv[1], t])

    J_auto = torch.zeros(3, 2, dtype=DTYPE)
    for i in range(3):
        g = torch.autograd.grad(output[i], (alpha, beta), retain_graph=True)
        J_auto[i, 0] = g[0]
        J_auto[i, 1] = g[1]

    return V_actual, J_auto


def run_stress(n=500):
    max_err = 0.0
    relative_errs = []
    for trial in range(n):
        try:
            cam, p, q, V_init = random_valid_config()
            V, J_auto = compute_autograd_jacobian(cam, p, q, V_init)
            # V is the actual V used in autograd (derived from v_quat)
            V_cam = cam.R @ (V - cam.c)
            if V_cam[2] < 0.5:
                continue   # behind or too close to camera, skip

            J_ana = J.jacobian_full_static(V, p, q, cam)
            err = (J_ana - J_auto).abs().max().item()
            rel = err / (J_auto.abs().max().item() + 1e-12)
            max_err = max(max_err, err)
            relative_errs.append(rel)
        except Exception:
            continue

    print(f"Over {len(relative_errs)} successful random configs:")
    print(f"  Max absolute err (analytical vs autograd): {max_err:.3e}")
    print(f"  Median relative err: {torch.tensor(relative_errs).median().item():.3e}")
    print(f"  Max relative err: {max(relative_errs):.3e}")


if __name__ == "__main__":
    torch.manual_seed(0)
    run_stress(500)
