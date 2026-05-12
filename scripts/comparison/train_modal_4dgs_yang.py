"""Modal entry for Yang et al. 4D Gaussian Splatting (ICLR 2024,
fudan-zvg/4d-gaussian-splatting) on monocular HyperNeRF-format scenes.

Architectural rationale: Yang's method is the only mature open-source 4DGS that
uses NATIVE 4D Gaussians with marginalization (Schur-on-time) rather than a
deformation-field on top of a 3D Gaussian. That makes it the right baseline for
testing whether the residual gap between our rank-2 disk + epsilon I projector
and D3DGS comes from representation cost (full-rank 4D vs rank-2 disk) or from
training/densification differences.

Differences vs upstream Yang's repo (we PATCH at container start):
  - Add `readHyperNeRFInfo` to scene/dataset_readers.py and a HyperNeRF branch
    in scene/__init__.py. Upstream supports only Colmap + Blender-DNeRF readers.
  - Drop in a minimal render.py (upstream has none — only train.py renders to
    TensorBoard, never saves PNGs to disk).
  - Pre-rectify NeRFies cameras with cv2.undistort once at container start
    (slice-banana has nonzero radial+tangential distortion; Yang's pinhole
    rasterizer cannot model it).

CUDA / image notes:
  - Yang's environment.yml pins cudatoolkit=11.6 + pytorch=1.12.1 + py3.7.
    We use pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel as the closest stable
    pre-built image and set TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" so the
    PTX fallback JIT-compiles for L4's sm_89 at runtime (CUDA 11.6 toolchain
    does not natively support sm_89; sm_89 was added in CUDA 11.8).
  - The custom rasterizer build is the highest tail-risk step. If it fails,
    the only sanctioned fallback is hustvl/4DGaussians (CVPR 2024, has native
    HyperNeRF + render.py) — but that swap loses the architectural argument
    (Wu et al. is HexPlane-deformation, same family as D3DGS, not 4D Gaussian).

Usage:
  modal run scripts/comparison/train_modal_4dgs_yang.py --cmd train --scene slice-banana \\
      --iters 14000 --resolution 4
  modal run scripts/comparison/train_modal_4dgs_yang.py --cmd render --scene slice-banana \\
      --resolution 8 --out-dir /checkpoints/yang4dgs-slice-banana-14000it
"""
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent.parent
PATCHES_DIR = REPO / "scripts" / "comparison" / "yang_4dgs_patches"

YANG_REPO = "https://github.com/fudan-zvg/4d-gaussian-splatting.git"
YANG_REV = "63725f21d4adc29669e565ae10e6b3ad6e0d1250"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel",
        add_python="3.10",
    )
    .apt_install(
        "git", "build-essential", "ninja-build", "wget",
        "libgl1", "libglib2.0-0",
    )
    .env({
        # PTX fallback for L4 (sm_89). CUDA 11.6 toolchain does not natively
        # produce sm_89 cubins; PTX from sm_86 will JIT-compile at runtime.
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6+PTX",
        "FORCE_CUDA": "1",
    })
    .pip_install(
        "numpy<2",
        "matplotlib",
        "pillow",
        "tqdm==4.66.1",
        "plyfile==0.8.1",
        "imageio==2.27.0",
        "imageio-ffmpeg",
        "opencv-python",
        "scipy",
        "lpips",
        "torchvision",
        "torchmetrics==0.11.4",
        "imagesize==1.4.1",
        "kornia==0.6.12",
        "omegaconf==2.3.0",
        "tensorboard",
    )
    .run_commands(
        f"cd /root && git clone --recursive {YANG_REPO} && "
        f"cd 4d-gaussian-splatting && git checkout {YANG_REV}",
        gpu="L4",
    )
    # PyTorch's cpp_extension JIT linker hard-codes -L/opt/conda/lib64 but the
    # conda image places libcudart.so in /opt/conda/lib (no lib64). Symlink so
    # the diff-gaussian-rasterization JIT load can find -lcudart.
    .run_commands("ln -sfn /opt/conda/lib /opt/conda/lib64")
    # CUDA extensions: simple-knn and pointops2 install via pip; their setup.py
    # uses CUDAExtension which produces a `.so` that pip places under the
    # installed package. NOTE: diff-gaussian-rasterization is NOT pip-installed
    # because Yang's repo loads it via torch.utils.cpp_extension.load() at first
    # import (see gaussian_renderer/diff_gaussian_rasterization.py:17-28). The
    # setup.py in that subdir is broken (declares packages=['diff_gaussian_
    # rasterization'] without a source dir) -- but it's a leftover from
    # upstream 3DGS; Yang doesn't use it.
    .run_commands(
        "cd /root/4d-gaussian-splatting/simple-knn && pip install .",
        gpu="L4",
    )
    # pointops2 will NOT build inside the cuda-11.6 base image: torch's
    # CUDAContext.h transitively pulls in cusolverDn.h which is not present
    # (the dev image ships nvcc and runtime libs but not cuSOLVER dev headers).
    # Yang only uses pointops2 for the rigid-loss kNN; our config sets
    # lambda_rigid=0 so we install a stub package inline whose imports succeed
    # and whose functions raise if ever called.
    .run_commands(
        # Two places need the stub: the site-packages location (used when cwd
        # doesn't contain pointops2/) AND the in-repo location at
        # /root/4d-gaussian-splatting/pointops2/functions/pointops.py (used
        # when cwd=YANG_DIR, which Python prefers via cwd-on-sys.path[0]).
        # The real functions/pointops.py imports `pointops2_cuda` which we
        # never built; overwriting with the stub makes both import paths safe.
        "mkdir -p /opt/conda/lib/python3.10/site-packages/pointops2/functions && "
        "printf '%s\\n' "
        "'def _stub(*a, **kw):' "
        "'    raise NotImplementedError(\"pointops2 stubbed -- only safe with lambda_rigid=0\")' "
        "'furthestsampling = _stub' "
        "'knnquery = _stub' "
        "| tee /opt/conda/lib/python3.10/site-packages/pointops2/functions/pointops.py "
        "      /root/4d-gaussian-splatting/pointops2/functions/pointops.py "
        ">/dev/null && "
        "echo '' > /opt/conda/lib/python3.10/site-packages/pointops2/__init__.py && "
        "echo '' > /opt/conda/lib/python3.10/site-packages/pointops2/functions/__init__.py",
    )
    # Pre-warm the JIT compile of diff_gaussian_rasterization so the cache
    # lands in the image and the first training run doesn't pay 5+ minutes of
    # nvcc latency. The `-g` flag in Yang's load() call is keep-debug-symbols,
    # not nvcc -G (full debug); release-quality kernels.
    .run_commands(
        "cd /root/4d-gaussian-splatting && python -c "
        "'from gaussian_renderer.diff_gaussian_rasterization import _C; "
        "print(\"diff_gaussian_rasterization JIT-loaded OK\")'",
        gpu="L4",
    )
    .add_local_dir(str(PATCHES_DIR), "/root/yang_patches")
)

app = modal.App("grassmann-yang-4dgs", image=image)

mono_vol = modal.Volume.from_name("gs-mono", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gs-checkpoints", create_if_missing=True)

VOLUMES = {"/data": mono_vol, "/checkpoints": ckpt_vol}
GPU = "L4"
YANG_DIR = "/root/4d-gaussian-splatting"


def _apply_patches() -> None:
    """Copy patched files from /root/yang_patches into the cloned repo and
    surgically disable TensorBoard in train.py.

    We keep the patches as standalone files in scripts/comparison/yang_4dgs_patches/ so
    they're versioned with our repo (and reviewable as diffs against upstream).
    Applied at container start rather than baked into the image so an iteration
    of the patches doesn't trigger a rasterizer rebuild.

    Why disable TB: tensorboard 2.20 + numpy 1.26 raises TypeError in
    `add_histogram` ("No loop matching the specified signature and casting was
    found for ufunc greater"); upstream's `tb_writer.add_histogram` for the
    opacity histogram fires every test iteration. Setting `tb_writer = None`
    upstream of all `if tb_writer: ...` blocks bypasses every TB call without
    affecting the actual test/eval loop (which we need for PSNR logging and
    for save_iterations bookkeeping).
    """
    import shutil
    import subprocess as _sp
    patches = {
        "dataset_readers.py": f"{YANG_DIR}/scene/dataset_readers.py",
        "scene_init.py":      f"{YANG_DIR}/scene/__init__.py",
        "render.py":          f"{YANG_DIR}/render.py",
    }
    for src_name, dst in patches.items():
        src = f"/root/yang_patches/{src_name}"
        print(f"  patch {src} -> {dst}", flush=True)
        shutil.copy(src, dst)
    # Force tb_writer = None in prepare_output_and_logger.
    _sp.run([
        "sed", "-i",
        "s|tb_writer = SummaryWriter(args.model_path)|tb_writer = None  # patched: tensorboard 2.20 + numpy 1.26 add_histogram TypeError|",
        f"{YANG_DIR}/train.py",
    ], check=True)
    print("  patched train.py: tb_writer = None", flush=True)


def _rectify_scene(scene: str, image_scale: int) -> str:
    """Pre-rectify a NeRFies-format scene with cv2.undistort. Returns the
    path to the rectified scene tree (suitable as Yang's source_path).

    Reads from /data/<scene>/ (raw, distorted) and writes to
    /tmp/<scene>-rect/ (camera/, rgb/<S>x/, dataset.json, metadata.json,
    points.npy, scene.json) with distortion zeroed in the camera JSONs.
    """
    import sys
    sys.path.insert(0, "/root/yang_patches")
    from rectify import rectify_scene
    return rectify_scene(
        src=f"/data/{scene}", dst=f"/tmp/{scene}-rect",
        image_scale=image_scale,
    )


@app.function(gpu=GPU, volumes=VOLUMES, timeout=4 * 3600)
def train(
    scene: str,
    num_iters: int,
    resolution: int,
    run_tag: str,
) -> None:
    import os, time as _time
    _apply_patches()
    scene_dir = _rectify_scene(scene, image_scale=resolution)

    suffix = f"-{run_tag}" if run_tag else ""
    out_dir = f"/checkpoints/yang4dgs-{scene}-{num_iters}it{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    # Write a slice-banana config to /tmp using the requested iters and
    # source path (saves us a separate config file per scene/iter combo).
    cfg_path = "/tmp/slice_banana_run.yaml"
    import shutil
    shutil.copy("/root/yang_patches/slice_banana.yaml", cfg_path)
    # Patch source_path/model_path/iterations into the YAML in-place.
    with open(cfg_path) as f:
        cfg = f.read()
    cfg = (cfg
        .replace("__SOURCE_PATH__", scene_dir)
        .replace("__MODEL_PATH__", out_dir)
        .replace("__ITERATIONS__", str(num_iters))
        .replace("__RESOLUTION__", "1"))   # we already scaled in rectify
    with open(cfg_path, "w") as f:
        f.write(cfg)
    print(f"--- config ---\n{cfg}\n--------------", flush=True)

    train_argv = [
        "python", "train.py",
        "--config", cfg_path,
        "--exhaust_test",
    ]
    print(f">>> {' '.join(train_argv)}", flush=True)
    _t0 = _time.perf_counter()
    subprocess.run(train_argv, check=True, cwd=YANG_DIR)
    _wall = _time.perf_counter() - _t0
    print(f"TRAIN_WALL_S={_wall:.1f}", flush=True)

    # Render test set with our patched render.py (saves PNGs to
    # <out_dir>/test/ours_<iter>/{renders,gt}/<NNNNN>.png)
    render_argv = [
        "python", "render.py",
        "--model_path", out_dir,
        "--iteration", str(num_iters),
        "--skip_train",
    ]
    print(f">>> {' '.join(render_argv)}", flush=True)
    subprocess.run(render_argv, check=True, cwd=YANG_DIR)
    ckpt_vol.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=1 * 3600)
def render_only(
    out_dir: str,
    resolution: int,
    iteration: int,
    scene: str = "slice-banana",
    override_resolution: int = -1,
) -> None:
    """Re-render an existing checkpoint. Re-rectifies the scene first since
    /tmp doesn't persist across containers.

    `resolution` is the scale used to find the rectified RGBs (rgb/<S>x/).
    `override_resolution` (>0) re-scales the cameras at render time -- pass
    2 to render at scale 8 from a scale-4-trained model.
    """
    _apply_patches()
    _rectify_scene(scene, image_scale=resolution)
    render_argv = [
        "python", "render.py",
        "--model_path", out_dir,
        "--iteration", str(iteration),
        "--skip_train",
    ]
    if override_resolution > 0:
        render_argv += ["--override_resolution", str(override_resolution)]
    print(f">>> {' '.join(render_argv)}", flush=True)
    subprocess.run(render_argv, check=True, cwd=YANG_DIR)
    ckpt_vol.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=600)
def smoke() -> None:
    """Verifies the image builds, patches apply, rectification works, and the
    rasterizer can load. Cheapest signal that the entire pipeline is wired up."""
    _apply_patches()
    out = _rectify_scene("slice-banana", image_scale=4)
    print(f"rectified: {out}", flush=True)
    # Confirm all CUDA extensions are loadable. diff_gaussian_rasterization is
    # imported via Yang's wrapper module (which JIT-compiles on first call;
    # pre-warmed in the image so the cache hit is fast).
    import os
    os.chdir(YANG_DIR)
    import sys
    sys.path.insert(0, YANG_DIR)
    from gaussian_renderer.diff_gaussian_rasterization import _C as _dgr_C  # noqa: F401
    import simple_knn._C  # noqa: F401
    # pointops2 is the stub (see image build); just confirm import path.
    from pointops2.functions.pointops import knnquery  # noqa: F401
    print("OK: all extensions importable (pointops2 = stub)", flush=True)


@app.local_entrypoint()
def main(
    cmd: str = "train",
    scene: str = "slice-banana",
    iters: int = 14000,
    resolution: int = 4,
    iteration: int = 14000,
    run_tag: str = "",
    out_dir: str = "",
    override_resolution: int = -1,
):
    if cmd == "smoke":
        smoke.remote()
    elif cmd == "train":
        train.remote(scene=scene, num_iters=iters, resolution=resolution, run_tag=run_tag)
    elif cmd == "render":
        if not out_dir:
            raise SystemExit("--cmd render requires --out-dir")
        render_only.remote(out_dir=out_dir, resolution=resolution,
                           iteration=iteration, scene=scene,
                           override_resolution=override_resolution)
    else:
        raise SystemExit(f"unknown --cmd {cmd!r}")
