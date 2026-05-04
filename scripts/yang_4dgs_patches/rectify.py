"""Pre-rectify a NeRFies-format scene: undistort RGB at the requested image
scale and zero out radial+tangential distortion in copied camera JSONs.

Yang's diff-gaussian-rasterization is pinhole-only. Slice-banana cameras have
nonzero radial AND tangential distortion (real handheld phone capture). We
must rectify before training, otherwise pixels at image edges land several
percent off.

Strategy (matches the (B) alternative in grassmann/datasets/nerfies.py:40-45):
  - Load each frame's distorted PNG and the camera intrinsics.
  - cv2.undistort with K = [[fx,0,cx],[0,fy,cy],[0,0,1]] and D = [k1,k2,p1,p2,k3].
  - Save the rectified PNG; copy the JSON with distortion fields zeroed.
  - Other files (dataset.json, metadata.json, points.npy, scene.json) are
    symlinked since they don't depend on distortion.

The original NeRFies tree is read-only on the gs-mono Modal volume so we write
to /tmp/<scene>-rect/ which is per-container. Cheap (~30s for 330 frames at 4x).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np


def _rectify_one_camera(K_orig, D, W, H):
    """Compute the rectified camera intrinsics for cv2.undistort with the
    default new K (alpha=0). Since cv2.undistort with newCameraMatrix=None
    keeps K unchanged, the returned new_K equals K_orig and we can keep the
    distortion-free principal point and focal length as-is."""
    import cv2
    new_K, _ = cv2.getOptimalNewCameraMatrix(K_orig, D, (W, H), alpha=0.0)
    # We pin alpha=0 (crop the rectified image to the inscribed valid region)
    # so border pixels are well-defined. For our small distortion magnitudes
    # the resulting new_K is close to K_orig; we use new_K for correctness.
    return new_K


def rectify_scene(src: str, dst: str, image_scale: int = 4) -> str:
    """Rectify all frames in `src` and write to `dst`. Returns dst.

    src layout (NeRFies):
        camera/<id>.json, rgb/<S>x/<id>.png, dataset.json, metadata.json,
        points.npy, scene.json

    dst layout (same, but distortion=0 in camera JSONs and rgb/<S>x/<id>.png
    is undistorted).
    """
    import cv2
    src = Path(src)
    dst = Path(dst)
    if dst.exists() and (dst / "_rectified.OK").exists():
        print(f"[rectify] cached at {dst}", flush=True)
        return str(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Symlink scene-level files unchanged.
    for fname in ("dataset.json", "metadata.json", "points.npy", "scene.json"):
        sp = src / fname
        if sp.exists():
            (dst / fname).symlink_to(sp.resolve())

    (dst / "camera").mkdir(exist_ok=True)
    (dst / "rgb").mkdir(exist_ok=True)
    (dst / "rgb" / f"{image_scale}x").mkdir(parents=True, exist_ok=True)

    # We rectify at the requested image_scale (faster than at full res then
    # downsampling) by scaling the camera matrix accordingly before undistort.
    cam_dir = src / "camera"
    rgb_dir = src / "rgb" / f"{image_scale}x"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"{rgb_dir} missing -- pre-downsampled RGBs required")

    cam_files = sorted(cam_dir.glob("*.json"))
    print(f"[rectify] {len(cam_files)} frames @ scale {image_scale}", flush=True)

    for i, cam_path in enumerate(cam_files):
        with open(cam_path) as f:
            data = json.load(f)
        item_id = cam_path.stem
        rad = data.get('radial_distortion', [0, 0, 0])
        tan = data.get('tangential_distortion', [0, 0]) or data.get('tangential', [0, 0])
        f_raw = data['focal_length']
        if isinstance(f_raw, (list, tuple)):
            fx, fy = float(f_raw[0]), float(f_raw[1])
        else:
            fx = fy = float(f_raw)
        cx, cy = float(data['principal_point'][0]), float(data['principal_point'][1])
        Wf, Hf = int(data['image_size'][0]), int(data['image_size'][1])

        # Scale intrinsics to the rectified resolution.
        s = float(image_scale)
        fx_s, fy_s = fx / s, fy / s
        cx_s, cy_s = cx / s, cy / s
        Ws, Hs = Wf // image_scale, Hf // image_scale

        K = np.array([[fx_s, 0.0, cx_s],
                      [0.0, fy_s, cy_s],
                      [0.0,  0.0, 1.0]], dtype=np.float64)
        D = np.array([rad[0], rad[1], tan[0], tan[1], rad[2] if len(rad) > 2 else 0.0],
                     dtype=np.float64)

        # Read distorted RGB at the requested scale.
        in_png = rgb_dir / f"{item_id}.png"
        if not in_png.exists():
            raise FileNotFoundError(f"{in_png} missing")
        img = cv2.imread(str(in_png), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cv2.imread failed: {in_png}")
        if img.shape[1] != Ws or img.shape[0] != Hs:
            raise ValueError(
                f"{in_png}: image shape {(img.shape[1], img.shape[0])} != "
                f"camera-scaled size {(Ws, Hs)}"
            )

        # Undistort. Keep newCameraMatrix=K so principal point & focal stay
        # the same -- this means our scene-level intrinsics are unchanged.
        und = cv2.undistort(img, K, D, newCameraMatrix=K)
        out_png = dst / "rgb" / f"{image_scale}x" / f"{item_id}.png"
        cv2.imwrite(str(out_png), und)

        # Write a copy of the camera JSON with distortion zeroed and intrinsics
        # at FULL resolution unchanged (the loader will scale them by 1/s
        # itself when image_scale=4).
        new_data = dict(data)
        new_data['radial_distortion'] = [0.0, 0.0, 0.0]
        new_data['tangential_distortion'] = [0.0, 0.0]
        new_data.pop('tangential', None)
        with open(dst / "camera" / cam_path.name, 'w') as f:
            json.dump(new_data, f)

        if (i + 1) % 50 == 0:
            print(f"[rectify] {i+1}/{len(cam_files)} done", flush=True)

    (dst / "_rectified.OK").touch()
    print(f"[rectify] -> {dst}", flush=True)
    return str(dst)


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rectify_scene(src, dst, image_scale=scale)
