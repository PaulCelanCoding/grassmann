"""Phase 3 visualizations.

Four demos that honestly show what the Grassmann Gaussian splatting model does.
All work with the natural (alpha, beta) parameterization and its geometry.

Demo 1: Temporal opacity fade. A single splat at a fixed physical location.
        It rises and falls in opacity as t sweeps across v_0.

Demo 2: Occlusion. Two splats at different depths, same pixel column. The
        near one correctly occludes the far one.

Demo 3: On-screen motion, using the geometric fact that moving along e2_hat
        drags the splat through space at velocity c_world / Sigma_tt_pure.
        (This spatial drift is intrinsic to the plane's tilt.)

Demo 4: A small multi-Gaussian "scene" with three differently colored splats.

Images are saved in /home/claude/grassmann/ and can be opened as PNGs.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from grassmann import quaternion as Q
from grassmann import grassmann as G
from grassmann.projection import Camera
from grassmann.gaussian import GaussianParams, compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize


DTYPE = torch.float64
H, W = 120, 200


def fresh_cam():
    return Camera(
        R=torch.eye(3, dtype=DTYPE),
        c=torch.zeros(3, dtype=DTYPE),
        fx=400, fy=400, cx=W / 2, cy=H / 2,
    )


def ray_gaussian(
    *,
    direction,                     # ray direction in world/camera coords (R^3)
    depth_mean=5.0,                # where along the ray the Gaussian mean sits
    sigma_aa=0.005, sigma_bb=0.005, sigma_ab=0.0,
    opacity=0.95, color=(1.0, 0.4, 0.4),
    sigma_k=5.0,
):
    """Construct a single Gaussian on a RAY FROM THE ORIGIN in the given direction.

    This is the natural parameterization for scene reconstruction: each Gaussian
    represents a contribution along a camera ray. Because the line passes through
    the origin (y = 0), we get c = 1, s = 0, and e2_hat is purely temporal.
    There is NO inherent spatial drift -- the Gaussian stays put unless we
    deliberately introduce one via sigma_ab != 0.
    """
    # Line through origin: x = (0, 0, 0), u = direction.
    x_line_t = torch.zeros(1, 3, dtype=DTYPE)
    u_dir_t = torch.tensor(direction, dtype=DTYPE).unsqueeze(0)
    u_dir_t = u_dir_t / u_dir_t.norm(dim=-1, keepdim=True)
    p, q = G.line_to_pq(x_line_t, u_dir_t)

    # y = 0, so the canonical embedded point is (1, 0) in H coordinates.
    # Equivalently: alpha_0 and beta_0 chosen so that v = (1, 0).
    # For a line through origin, e1_hat = (0, u) and e2_hat = (1, 0), so
    # v = (1, 0) = 1 * e2_hat + 0 * e1_hat. We need alpha = 0 and beta = 1.
    # But to place the mean at depth `depth_mean`, we want V_k = depth_mean * u_hat.
    # For line through origin, spatial part of v is alpha * u_hat (since e1_hat spatial = u_hat).
    # So set alpha_0 = depth_mean, beta_0 = 1.
    e1_hat, e2_hat = G.orthonormal_basis(p, q)
    # Verify our assumption.
    # alpha_0 * e1_hat + beta_0 * e2_hat should have spatial part = alpha_0 * u_hat
    # and time part = beta_0 * 1.
    alpha_val = torch.tensor([float(depth_mean)], dtype=DTYPE)
    beta_val = torch.tensor([1.0], dtype=DTYPE)    # time = 1 (arbitrary offset)

    # Cholesky of Sigma_k.
    sa = np.sqrt(sigma_aa)
    L10 = sigma_ab / sa
    inner = sigma_bb - sigma_ab * sigma_ab / sigma_aa
    if inner <= 0:
        raise ValueError(f"Sigma_k not PD: inner = {inner}")
    L11 = np.sqrt(inner)
    L = torch.tensor([[[sa, 0.0], [L10, L11]]], dtype=DTYPE)

    return GaussianParams(
        p_im=Q.imag(p), q_im=Q.imag(q),
        alpha_0=alpha_val, beta_0=beta_val, L=L,
        opacity=torch.tensor([opacity], dtype=DTYPE),
        color=torch.tensor([list(color)], dtype=DTYPE),
        sigma_k=sigma_k,
    )


def single_gaussian(
    *,
    x_line, u_dir,
    sigma_aa=0.01, sigma_ab=0.0, sigma_bb=0.01,
    opacity=0.95, color=(1.0, 0.4, 0.4),
    sigma_k=5.0,
):
    """Construct a single Gaussian at the canonical embedded point of a line."""
    x_line_t = torch.tensor(x_line, dtype=DTYPE).unsqueeze(0)
    u_dir_t = torch.tensor(u_dir, dtype=DTYPE).unsqueeze(0)
    u_dir_t = u_dir_t / u_dir_t.norm(dim=-1, keepdim=True)
    p, q = G.line_to_pq(x_line_t, u_dir_t)

    # Place the mean at (1, y) where y is the standard-form point.
    y, _ = G.line_standard_form(x_line_t, u_dir_t)
    canonical = torch.cat([torch.ones(1, 1, dtype=DTYPE), y], dim=-1)
    e1_hat, e2_hat = G.orthonormal_basis(p, q)
    alpha_val = (canonical * e1_hat).sum(dim=-1)
    beta_val = (canonical * e2_hat).sum(dim=-1)

    # Cholesky factor of Sigma_k = L L^T.
    sa = np.sqrt(sigma_aa)
    L10 = sigma_ab / sa
    inner = sigma_bb - sigma_ab * sigma_ab / sigma_aa
    if inner <= 0:
        raise ValueError(f"Sigma_k not PD: sigma_ab too large, inner = {inner}")
    L11 = np.sqrt(inner)
    L = torch.tensor([[[sa, 0.0], [L10, L11]]], dtype=DTYPE)

    return GaussianParams(
        p_im=Q.imag(p), q_im=Q.imag(q),
        alpha_0=alpha_val, beta_0=beta_val, L=L,
        opacity=torch.tensor([opacity], dtype=DTYPE),
        color=torch.tensor([list(color)], dtype=DTYPE),
        sigma_k=sigma_k,
    )


def concat_params(*param_list):
    """Combine multiple single-Gaussian GaussianParams into one batched object."""
    return GaussianParams(
        p_im=torch.cat([p.p_im for p in param_list]),
        q_im=torch.cat([p.q_im for p in param_list]),
        alpha_0=torch.cat([p.alpha_0 for p in param_list]),
        beta_0=torch.cat([p.beta_0 for p in param_list]),
        L=torch.cat([p.L for p in param_list]),
        opacity=torch.cat([p.opacity for p in param_list]),
        color=torch.cat([p.color for p in param_list]),
        sigma_k=param_list[0].sigma_k,
    )


def render_frame(params, t, cam, background):
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t)
    sg = project_to_screen(params, tc, cam)
    img = rasterize(sg, H=H, W=W, background=background)
    return img.numpy().clip(0, 1)


# -----------------------------------------------------------------------------
# Demo 1: Temporal opacity fade
# -----------------------------------------------------------------------------

def demo_temporal_fade():
    """A splat on a ray from the camera -> no spatial drift. Pure temporal fade."""
    params = ray_gaussian(
        direction=[0.0, 0.0, 1.0],   # straight ahead -- splat at image center
        depth_mean=5.0,
        sigma_aa=0.01, sigma_ab=0.0, sigma_bb=0.05,
        opacity=0.98, color=(1.0, 0.4, 0.4), sigma_k=5.0,
    )
    derived = compute_derived(params)
    v0 = derived.v_0.item()
    stt = derived.Sigma_tt.sqrt().item()
    print(f"[Demo 1] V_k={derived.V_k[0].tolist()}, v_0={v0:.3f}, sqrt(Sigma_tt)={stt:.3f}")
    print(f"         c_world = {derived.c_world[0].tolist()}   (should be ~0)")

    ts = np.linspace(v0 - 2 * stt, v0 + 2 * stt, 7)
    cam = fresh_cam()
    bg = torch.tensor([0.05, 0.05, 0.1], dtype=DTYPE)

    fig, axes = plt.subplots(1, len(ts), figsize=(2.2 * len(ts), 2.5))
    for ax, t in zip(axes, ts):
        img = render_frame(params, float(t), cam, bg)
        tc = condition_on_time(params, derived, float(t))
        ax.imshow(img)
        ax.set_title(f"t = {t:.2f}\n$w_t$ = {tc.w_t.item():.2f}")
        ax.axis("off")
    fig.suptitle(
        "Demo 1: Pure temporal fade (ray-from-camera, no spatial drift)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("docs/images/demo1_temporal_fade.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved demo1_temporal_fade.png")


# -----------------------------------------------------------------------------
# Demo 2: Occlusion
# -----------------------------------------------------------------------------

def demo_occlusion():
    """Two splats on the same ray, red at depth 3, green at depth 8."""
    p_red = ray_gaussian(
        direction=[0.0, 0.0, 1.0], depth_mean=3.0,
        sigma_aa=0.005, sigma_bb=0.05,
        opacity=0.95, color=(1.0, 0.3, 0.3), sigma_k=8.0,
    )
    p_green = ray_gaussian(
        direction=[0.0, 0.0, 1.0], depth_mean=8.0,
        sigma_aa=0.005, sigma_bb=0.05,
        opacity=0.95, color=(0.3, 1.0, 0.3), sigma_k=15.0,
    )
    params_both = concat_params(p_red, p_green)

    cam = fresh_cam()
    bg = torch.tensor([0.05, 0.05, 0.1], dtype=DTYPE)

    d_red = compute_derived(p_red)
    d_green = compute_derived(p_green)
    t0 = 0.5 * (d_red.v_0.item() + d_green.v_0.item())

    img_green_alone = render_frame(p_green, t0, cam, bg)
    img_red_alone = render_frame(p_red, t0, cam, bg)
    img_both = render_frame(params_both, t0, cam, bg)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img_green_alone); axes[0].set_title("Green alone (depth 8)"); axes[0].axis("off")
    axes[1].imshow(img_red_alone); axes[1].set_title("Red alone (depth 3)"); axes[1].axis("off")
    axes[2].imshow(img_both); axes[2].set_title("Both: red occludes green"); axes[2].axis("off")
    fig.suptitle("Demo 2: Alpha-composited occlusion (front-to-back)", fontsize=11)
    plt.tight_layout()
    plt.savefig("docs/images/demo2_occlusion.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved demo2_occlusion.png")


# -----------------------------------------------------------------------------
# Demo 3: On-screen motion
# -----------------------------------------------------------------------------

def demo_motion():
    """A splat that moves along its ray over time (depth changes).

    On a ray from the origin, e1_hat is purely along the ray direction (spatial)
    and e2_hat is purely temporal. Setting sigma_ab != 0 couples alpha (along-ray)
    with beta (time), producing a splat whose depth changes linearly with time.
    On screen, a splat moving in depth appears to shrink/grow and shift toward
    the principal point (unless the ray passes through it).
    """
    params = ray_gaussian(
        direction=[0.2, 0.0, 1.0],   # ray to the right and forward
        depth_mean=5.0,
        sigma_aa=0.5, sigma_ab=0.3, sigma_bb=0.3,   # temporal extent big, coupled with alpha
        opacity=0.98, color=(1.0, 0.4, 0.4), sigma_k=5.0,
    )
    derived = compute_derived(params)
    v0 = derived.v_0.item()
    stt = derived.Sigma_tt.sqrt().item()
    vel = (derived.c_world[0] / derived._sigma_tt_pure.item()).tolist()
    print(f"[Demo 3] v_0={v0:.3f}, sqrt(Sigma_tt)={stt:.3f}, implied dV/dt={vel}")

    ts = np.linspace(v0 - 1.5 * stt, v0 + 1.5 * stt, 7)
    cam = fresh_cam()
    bg = torch.tensor([0.05, 0.05, 0.1], dtype=DTYPE)

    fig, axes = plt.subplots(1, len(ts), figsize=(2.2 * len(ts), 2.5))
    for ax, t in zip(axes, ts):
        img = render_frame(params, float(t), cam, bg)
        tc = condition_on_time(params, derived, float(t))
        ax.imshow(img)
        ax.set_title(f"t = {t:.2f}\n$w_t$={tc.w_t.item():.2f}")
        ax.axis("off")
    fig.suptitle(
        "Demo 3: Motion along the ray via $\\sigma_{\\alpha\\beta}$ coupling\n"
        f"($dV/dt \\approx$ [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}])",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("docs/images/demo3_motion.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved demo3_motion.png")


# -----------------------------------------------------------------------------
# Demo 4: Multi-Gaussian scene
# -----------------------------------------------------------------------------

def demo_scene():
    """Three Gaussians on rays at different screen positions, depths, and temporal centers."""
    # Build three ray Gaussians; override beta_0 so each has a different v_0.
    p1 = ray_gaussian(
        direction=[-0.15, 0.1, 1.0], depth_mean=4.0,
        sigma_aa=0.01, sigma_bb=0.05,
        opacity=0.95, color=(1.0, 0.3, 0.3), sigma_k=5.0,
    )
    p2 = ray_gaussian(
        direction=[0.15, -0.05, 1.0], depth_mean=6.0,
        sigma_aa=0.01, sigma_bb=0.05,
        opacity=0.95, color=(0.3, 1.0, 0.3), sigma_k=5.0,
    )
    p3 = ray_gaussian(
        direction=[0.0, 0.05, 1.0], depth_mean=8.0,
        sigma_aa=0.01, sigma_bb=0.05,
        opacity=0.95, color=(0.3, 0.5, 1.0), sigma_k=5.0,
    )
    # Shift beta_0 to stagger the temporal centers.
    p1.beta_0[:] = 0.0       # v_0 ~ 0
    p2.beta_0[:] = 1.0
    p3.beta_0[:] = 2.0
    params = concat_params(p1, p2, p3)
    derived = compute_derived(params)
    v0s = derived.v_0.tolist()
    print(f"[Demo 4] v_0 per Gaussian = {[f'{v:.2f}' for v in v0s]}")

    t_range = (min(v0s) - 1.5, max(v0s) + 1.5)
    ts = np.linspace(*t_range, 7)
    cam = fresh_cam()
    bg = torch.tensor([0.05, 0.05, 0.1], dtype=DTYPE)

    fig, axes = plt.subplots(1, len(ts), figsize=(2.2 * len(ts), 2.5))
    for ax, t in zip(axes, ts):
        img = render_frame(params, float(t), cam, bg)
        ax.imshow(img)
        ax.set_title(f"t = {t:.2f}")
        ax.axis("off")
    fig.suptitle(
        f"Demo 4: Three Gaussians with staggered temporal centers "
        f"(v_0 = {v0s[0]:.1f}, {v0s[1]:.1f}, {v0s[2]:.1f}).\n"
        f"Watch them fade in / out.",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("docs/images/demo4_scene.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved demo4_scene.png")


if __name__ == "__main__":
    demo_temporal_fade()
    demo_occlusion()
    demo_motion()
    demo_scene()
    print("\nAll demos done. PNGs saved in /home/claude/grassmann/")
