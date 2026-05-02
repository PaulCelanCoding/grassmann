"""Visualize what the Jacobian does geometrically.

We take a Gaussian mean v in E_{p,q} and visualize:
  1. The canonical plane E_{p,q} in 3D space, by sampling points alpha*e1_hat + beta*e2_hat.
  2. Their perspective projections on the image plane.
  3. The two basis directions e1_hat (purely spatial, green) and e2_hat (space+time, red).
  4. The Jacobian columns overlaid in pixel space, showing the first-order
     approximation of motion in alpha and beta directions.

Output: a PNG with two panels.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from grassmann import quaternion as Q
from grassmann import grassmann as G
from grassmann import jacobian as J
from grassmann.projection import Camera, project_static, world_to_camera


def visualize_jacobian():
    torch.manual_seed(0)
    dtype = torch.float64

    # Setup: a horizontal line that passes through (2, 1, 5) in direction roughly +x.
    # We'll put the Gaussian mean on that line, well in front of the camera at origin.
    x_line = torch.tensor([2.0, 1.0, 5.0], dtype=dtype)
    u_dir = torch.tensor([1.0, 0.2, 0.0], dtype=dtype)
    u_dir = u_dir / u_dir.norm()
    p, q = G.line_to_pq(x_line.unsqueeze(0), u_dir.unsqueeze(0))
    p, q = p[0], q[0]

    # The Gaussian mean v lives in E_{p,q}. The plane E_{p,q} contains (1, y) and (0, u_hat)
    # (in quaternion coordinates), where y is the standard-form point on the line.
    # We want V (spatial part of v) to be on the physical line in R^3. A valid choice is
    # v = (1, y) -- this is the embedded representation of the line's closest-to-origin
    # point at time t=1.
    y_std, u_hat_std = G.line_standard_form(x_line, u_dir)
    v_quat = Q.from_real_imag(torch.tensor(1.0, dtype=dtype), y_std)    # (1, y) in E_{p,q}
    V = Q.imag(v_quat)   # spatial part: this is y, which lies on the line

    # We want the point to be about 5 units in front of a camera at origin.
    # Shift v if needed: add to x to move the line center closer to the camera.
    # Actually simpler: put the camera at origin looking along +z; V is ~(0, ?, 5).
    cam = Camera(
        R=torch.eye(3, dtype=dtype),
        c=torch.zeros(3, dtype=dtype),
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
    )

    # Basis vectors of E_{p,q}
    e1_hat, e2_hat = G.orthonormal_basis(p, q)

    # --- Panel 1: plane E_{p,q} sampled in 3D ---
    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")

    # Sample a grid (alpha, beta) in smaller range so projection stays well-behaved.
    alphas = np.linspace(-0.8, 0.8, 11)
    betas = np.linspace(-0.4, 0.4, 11)
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    pts = []
    times = []
    for a, b in zip(A.flatten(), B.flatten()):
        z = v_quat + float(a) * e1_hat + float(b) * e2_hat
        pts.append(Q.imag(z).numpy())
        times.append(Q.real(z).item())
    pts = np.array(pts).reshape(A.shape + (3,))
    times = np.array(times).reshape(A.shape)

    # Color by time component (beta axis).
    surf = ax1.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2],
                            facecolors=plt.cm.coolwarm(
                                (times - times.min()) / (np.ptp(times) + 1e-12)),
                            alpha=0.6, linewidth=0)
    # Basis arrows
    V_np = V.numpy()
    e1_spatial = J.jacobian_embed(p, q)[..., 0].numpy()   # spatial part of e1_hat
    e2_spatial = J.jacobian_embed(p, q)[..., 1].numpy()
    ax1.quiver(*V_np, *e1_spatial, color="green", length=1.0, linewidth=2,
               label="e1_hat (spatial)", arrow_length_ratio=0.15)
    ax1.quiver(*V_np, *e2_spatial, color="red", length=1.0, linewidth=2,
               label="e2_hat (space+time)", arrow_length_ratio=0.15)

    # Camera
    ax1.scatter([0], [0], [0], color="black", s=60, marker="^", label="camera")
    # Line through v in direction of p+q (the physical line the Gaussian represents)
    d = (p[1:] + q[1:]).numpy()
    d /= np.linalg.norm(d)
    t_line = np.linspace(-2, 2, 40)
    line = V_np[None, :] + t_line[:, None] * d[None, :]
    ax1.plot(line[:, 0], line[:, 1], line[:, 2], "k--", alpha=0.6, label="physical line (dir p+q)")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("Canonical plane $E_{p,q}$ embedded in 3D\n(color = time component)")
    ax1.legend(loc="upper left", fontsize=8)

    # --- Panel 2: image-space projection + Jacobian ---
    ax2 = fig.add_subplot(1, 2, 2)

    # Project the whole grid onto image plane (ignoring time for spatial plot).
    uv_grid = []
    for a, b in zip(A.flatten(), B.flatten()):
        z = v_quat + float(a) * e1_hat + float(b) * e2_hat
        uv = project_static(Q.imag(z).unsqueeze(0), cam).squeeze(0).numpy()
        uv_grid.append(uv)
    uv_grid = np.array(uv_grid).reshape(A.shape + (2,))

    # Plot curves of constant beta (varying alpha) and constant alpha (varying beta).
    for i in range(A.shape[0]):
        ax2.plot(uv_grid[i, :, 0], uv_grid[i, :, 1], color="gray", alpha=0.3, lw=0.5)
    for j in range(A.shape[1]):
        ax2.plot(uv_grid[:, j, 0], uv_grid[:, j, 1], color="gray", alpha=0.3, lw=0.5)

    # Projected mean and Jacobian arrows.
    uv_mean = project_static(V.unsqueeze(0), cam).squeeze(0).numpy()

    J_full = J.jacobian_full_static(V, p, q, cam).numpy()
    J_spatial = J_full[:2]   # (2, 2) -- first two rows
    # Column 0: dP/dalpha (pixel motion when varying alpha)
    # Column 1: dP/dbeta  (pixel motion when varying beta)
    ax2.arrow(uv_mean[0], uv_mean[1], J_spatial[0, 0], J_spatial[1, 0],
              color="green", head_width=3, length_includes_head=True, label="dP/d$\\alpha$")
    ax2.arrow(uv_mean[0], uv_mean[1], J_spatial[0, 1], J_spatial[1, 1],
              color="red", head_width=3, length_includes_head=True, label="dP/d$\\beta$")
    ax2.scatter([uv_mean[0]], [uv_mean[1]], color="black", s=40, zorder=5, label="projected mean")

    ax2.set_xlabel("u (pixels)")
    ax2.set_ylabel("v (pixels)")
    ax2.set_title("Image plane: $(\\alpha, \\beta)$ grid + Jacobian arrows")
    ax2.invert_yaxis()   # image convention: y grows downward
    ax2.legend(fontsize=9)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)

    # Also: print the time row to remind the user it exists.
    J_time_row = J_full[2]
    ax2.text(0.02, 0.98, f"time row of J: [{J_time_row[0]:.3f}, {J_time_row[1]:.3f}]",
             transform=ax2.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    plt.savefig("/home/claude/grassmann/jacobian_viz.png", dpi=110)
    plt.close()
    print("Saved visualization to jacobian_viz.png")


if __name__ == "__main__":
    visualize_jacobian()
