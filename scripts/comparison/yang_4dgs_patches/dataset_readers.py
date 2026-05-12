#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm
import torch
from utils.general_utils import fps
from multiprocessing.pool import ThreadPool
import imagesize

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float = 0.0
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    if 'nx' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    if 'time' in vertices:
        timestamp = vertices['time'][:, None]
    else:
        timestamp = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals, time=timestamp)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, num_pts_ratio=1.0):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    if num_pts_ratio > 1.001:
        num_pts = int((num_pts_ratio - 1) * pcd.points.shape[0])
        mean_xyz = pcd.points.mean(axis=0)
        min_rand_xyz = mean_xyz - np.array([0.5, 0.5, 0.5])
        max_rand_xyz = mean_xyz + np.array([0.5, 2.0, 0.5])
        xyz = np.concatenate([pcd.points, 
                              np.random.random((num_pts, 3)) * (max_rand_xyz - min_rand_xyz) + min_rand_xyz], 
                              axis=0)
        colors = np.concatenate([pcd.colors, 
                              SH2RGB(np.random.random((num_pts, 3)) / 255.0)], 
                              axis=0)
        normals = np.concatenate([pcd.normals, 
                              np.zeros((num_pts, 3))], 
                              axis=0)
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", time_duration=None, frame_ratio=1, dataloader=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
    if "camera_angle_x" in contents:
        fovx = contents["camera_angle_x"]
        
    frames = contents["frames"]
    tbar = tqdm(range(len(frames)))
    def frame_read_fn(idx_frame):
        idx = idx_frame[0]
        frame = idx_frame[1]
        timestamp = frame.get('time', 0.0)
        if frame_ratio > 1:
            timestamp /= frame_ratio
        if time_duration is not None and 'time' in frame:
            if timestamp < time_duration[0] or timestamp > time_duration[1]:
                return

        cam_name = os.path.join(path, frame["file_path"] + extension)

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name) # .replace('hdImgs_unditorted', 'hdImgs_unditorted_rgba').replace('.jpg', '.png')
        image_name = Path(cam_name).stem
        
        if not dataloader:
            with Image.open(image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            if norm_data[:, :, 3:4].min() < 1:
                arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
            else:
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            width, height = image.size[0], image.size[1]
        else:
            image = np.empty(0)
            width, height = imagesize.get(image_path)
        
        if 'depth_path' in frame:
            depth_name = frame["depth_path"]
            if not extension in frame["depth_path"]:
                depth_name = frame["depth_path"] + extension
            depth_path = os.path.join(path, depth_name)
            depth = Image.open(depth_path).copy()
        else:
            depth = None
        tbar.update(1)
        if 'fl_x' in frame and 'fl_y' in frame and 'cx' in frame and 'cy' in frame:
            FovX = FovY = -1.0
            fl_x = frame['fl_x']
            fl_y = frame['fl_y']
            cx = frame['cx']
            cy = frame['cy']
            return CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                        image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp,
                        fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
            
        elif 'fl_x' in contents and 'fl_y' in contents and 'cx' in contents and 'cy' in contents:
            FovX = FovY = -1.0
            fl_x = contents['fl_x']
            fl_y = contents['fl_y']
            cx = contents['cx']
            cy = contents['cy']
            return CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                        image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp,
                        fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy)
        else:
            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy
            FovX = fovx
            return CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                            image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp)
    
    with ThreadPool() as pool:
        cam_infos = pool.map(frame_read_fn, zip(list(range(len(frames))), frames))
        pool.close()
        pool.join()
        
    cam_infos = [cam_info for cam_info in cam_infos if cam_info is not None]
    
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", num_pts=100_000, time_duration=None, num_extra_pts=0, frame_ratio=1, dataloader=False):
    
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, time_duration=time_duration, frame_ratio=frame_ratio, dataloader=dataloader)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json" if not path.endswith('lego') else "transforms_val.json", white_background, extension, time_duration=time_duration, frame_ratio=frame_ratio, dataloader=dataloader)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if pcd.points.shape[0] > num_pts:
        mask = np.random.randint(0, pcd.points.shape[0], num_pts)
        # mask = fps(torch.from_numpy(pcd.points).cuda()[None], num_pts).cpu().numpy()
        if pcd.time is not None:
            times = pcd.time[mask]
        else:
            times = None
        xyz = pcd.points[mask]
        rgb = pcd.colors[mask]
        normals = pcd.normals[mask]
        if times is not None:
            time_mask = (times[:,0] < time_duration[1]) & (times[:,0] > time_duration[0])
            xyz = xyz[time_mask]
            rgb = rgb[time_mask]
            normals = normals[time_mask]
            times = times[time_mask]
        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals, time=times)
        
    if num_extra_pts > 0:
        times = pcd.time
        xyz = pcd.points
        rgb = pcd.colors
        normals = pcd.normals
        bound_min, bound_max = xyz.min(0), xyz.max(0)
        radius = 60.0 # (bound_max - bound_min).mean() + 10
        phi = 2.0 * np.pi * np.random.rand(num_extra_pts)
        theta = np.arccos(2.0 * np.random.rand(num_extra_pts) - 1.0)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        xyz_extra = np.stack([x, y, z], axis=1)
        normals_extra = np.zeros_like(xyz_extra)
        rgb_extra = np.ones((num_extra_pts, 3)) / 2
        
        xyz = np.concatenate([xyz, xyz_extra], axis=0)
        rgb = np.concatenate([rgb, rgb_extra], axis=0)
        normals = np.concatenate([normals, normals_extra], axis=0)
        
        if times is not None:
            times_extra = torch.zeros(((num_extra_pts, 3))) + (time_duration[0] + time_duration[1]) / 2
            times = np.concatenate([times, times_extra], axis=0)
            
        pcd = BasicPointCloud(points=xyz, 
                              colors=rgb,
                              normals=normals,
                              time=times)
        
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def _hyper_load_camera_pinhole(camera_path):
    """Load one NeRFies-format camera JSON. Assumes distortion has been
    pre-rectified (we error otherwise -- silent pinhole projection through
    distorted JSON gives geometrically wrong renders)."""
    with open(camera_path) as f:
        data = json.load(f)
    rad = data.get('radial_distortion', [0, 0, 0])
    tan = data.get('tangential_distortion', [0, 0]) or data.get('tangential', [0, 0])
    if any(abs(x) > 1e-8 for x in rad) or any(abs(x) > 1e-8 for x in tan):
        raise ValueError(
            f"{camera_path}: nonzero distortion (radial={rad}, tangential={tan}). "
            "Pre-rectify with cv2.undistort before training."
        )
    R_w2c = np.array(data['orientation'], dtype=np.float64)        # world->cam
    pos   = np.array(data['position'],    dtype=np.float64)        # camera center in world
    f_raw = data['focal_length']
    if isinstance(f_raw, (list, tuple)):
        fx, fy = float(f_raw[0]), float(f_raw[1])
    else:
        fx = fy = float(f_raw)
    pp = data['principal_point']
    cx, cy = float(pp[0]), float(pp[1])
    W, H = int(data['image_size'][0]), int(data['image_size'][1])
    # Yang's Camera stores R = camera-to-world rotation (then transposed inside
    # getWorld2View2). T = world-to-camera translation.
    R_yang = R_w2c.T
    T_yang = -R_w2c @ pos
    return R_yang, T_yang, fx, fy, cx, cy, W, H


def readHyperNeRFInfo(path, white_background, eval, num_pts=100_000,
                      time_duration=None, extension=".png", num_extra_pts=0,
                      frame_ratio=1, dataloader=False):
    """HyperNeRF / NeRFies monocular reader for Yang's 4DGS.

    Layout expected (after pre-rectification):
        <path>/dataset.json          { ids, train_ids, val_ids }
        <path>/metadata.json         { id: { time_id, warp_id, appearance_id, camera_id } }
        <path>/camera/<id>.json      per-frame intrinsics+extrinsics, distortion=0
        <path>/rgb/<S>x/<id>.png     downscaled images
        <path>/points.npy            (N, 3) initial point cloud

    Train/test split: hardcoded to deformable_interp convention
    (ids[::4] train, ids[2::4] val) to match D3DGS iso14k baseline. NeRFies'
    own dataset.json/train_ids/val_ids is ignored (slice-banana ships with
    train_ids=all-ids, val_ids=[] which is not what any monocular benchmark
    actually uses).

    Time normalization: time_id divided by (T-1) to land in [0, 1]. Yang's
    time_duration default expects [0, 1] (matches dnerf configs).
    """
    print(f"[HyperNeRF reader] loading {path}", flush=True)
    image_scale = 4  # The rectifier always writes rgb/4x/. We tell Yang's
    # loadCam to skip its own downscale by passing args.resolution=1 in the
    # YAML config; the cx/cy/fl_x/fl_y here are already at the rectified scale.

    with open(os.path.join(path, "dataset.json")) as f:
        ds = json.load(f)
    with open(os.path.join(path, "metadata.json")) as f:
        meta = json.load(f)
    ids = list(ds['ids'])
    T_total = len(ids)

    # Hardcoded interp convention (matches D3DGS iso14k baseline + our SH3 14k).
    train_ids = set(ids[::4])
    val_ids   = set(ids[2::4])

    cam_dir = os.path.join(path, "camera")
    rgb_dir = os.path.join(path, "rgb", f"{image_scale}x")
    points_npy = os.path.join(path, "points.npy")

    train_cam_infos = []
    test_cam_infos = []
    for idx, item_id in enumerate(tqdm(ids, desc="HyperNeRF cameras")):
        R, T, fx, fy, cx, cy, raw_W, raw_H = _hyper_load_camera_pinhole(
            os.path.join(cam_dir, f"{item_id}.json")
        )
        # Intrinsics scale by 1/image_scale; image dims scale too.
        fl_x = fx / image_scale
        fl_y = fy / image_scale
        cx_s = cx / image_scale
        cy_s = cy / image_scale
        W = raw_W // image_scale
        H = raw_H // image_scale
        FovX = 2.0 * np.arctan(W / (2.0 * fl_x))
        FovY = 2.0 * np.arctan(H / (2.0 * fl_y))

        timestamp = meta[item_id]['time_id'] / max(T_total - 1, 1)

        image_path = os.path.join(rgb_dir, f"{item_id}{extension}")
        if not dataloader:
            with Image.open(image_path) as im_load:
                im_data = np.array(im_load.convert("RGB"))
            arr = im_data / 255.0
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            width, height = image.size[0], image.size[1]
            assert (width, height) == (W, H), f"{item_id}: image {(width,height)} != camera {(W,H)}"
        else:
            image = np.empty(0)
            width, height = W, H

        ci = CameraInfo(
            uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
            image=image, depth=None,
            image_path=image_path, image_name=item_id,
            width=width, height=height, timestamp=timestamp,
            fl_x=fl_x, fl_y=fl_y, cx=cx_s, cy=cy_s,
        )
        if item_id in val_ids:
            test_cam_infos.append(ci)
        if item_id in train_ids:
            train_cam_infos.append(ci)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Init point cloud from points.npy. NeRFies ships only positions; we color
    # them with random SH-decoded RGB the same way readNerfSyntheticInfo does
    # when no PLY exists.
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"[HyperNeRF reader] writing {ply_path} from points.npy")
        xyz = np.load(points_npy).astype(np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"points.npy has unexpected shape {xyz.shape}")
        if xyz.shape[0] > num_pts:
            mask = np.random.choice(xyz.shape[0], num_pts, replace=False)
            xyz = xyz[mask]
        shs = np.random.random((xyz.shape[0], 3)) / 255.0
        rgb = SH2RGB(shs) * 255.0
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    if num_extra_pts > 0:
        xyz = pcd.points
        rgb = pcd.colors
        normals = pcd.normals
        radius = 60.0
        phi = 2.0 * np.pi * np.random.rand(num_extra_pts)
        theta = np.arccos(2.0 * np.random.rand(num_extra_pts) - 1.0)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        xyz_extra = np.stack([x, y, z], axis=1)
        xyz = np.concatenate([xyz, xyz_extra], axis=0)
        rgb = np.concatenate([rgb, np.ones((num_extra_pts, 3)) / 2], axis=0)
        normals = np.concatenate([normals, np.zeros_like(xyz_extra)], axis=0)
        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals, time=None)

    print(
        f"[HyperNeRF reader] {len(train_cam_infos)} train / "
        f"{len(test_cam_infos)} test, {pcd.points.shape[0]} init pts, "
        f"radius={nerf_normalization['radius']:.4f}",
        flush=True,
    )

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "HyperNeRF": readHyperNeRFInfo,
}