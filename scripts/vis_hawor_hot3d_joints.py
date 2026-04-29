#!/usr/bin/env python3
"""
Visualize HaWoR predicted hand joints projected onto extracted RGB frames.

This script is CPU-only by default and reuses the same steps as eval_hawor_hot3d:
1) MANO forward from world_space_res.pth
2) SE3 align SLAM world -> HOT3D world using SLAM keyframe camera centers vs GT camera centers
3) Project aligned 3D joints to image with K from vrs_meta.npz
4) Draw overlays and optionally export mp4

python scripts/vis_hawor_hot3d_joints.py \
  --sequence_dir output/eval_hawor_val/per_sequence/P0001_15c4300c \
  --gt_root /data/hot3d_gt/val \
  --draw_gt \
  --draw_mesh \
  --mesh_edge_stride 8

python scripts/vis_hawor_hot3d_joints.py \
  --sequence_dir output/eval_hawor_val/per_sequence/P0001_15c4300c \
  --gt_root /data/hot3d_gt/val \
  --draw_gt \
  --draw_both_conventions \
  --solid_render \
  --solid_alpha 0.7
"""

import argparse
import os
import sys
from glob import glob
from typing import Optional

import cv2
import joblib
import numpy as np
import torch

# Match eval_hawor_hot3d.py import behavior when launched as "python scripts/..."
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left

try:
    from projectaria_tools.core.calibration import LINEAR
    from projectaria_tools.core.stream_id import StreamId
    from data_loaders.loader_object_library import ObjectLibrary, load_object_library
    from dataset_api import Hot3dDataProvider
except Exception:
    LINEAR = None
    StreamId = None
    ObjectLibrary = None
    load_object_library = None
    Hot3dDataProvider = None


HOT3D_TO_MANO = [16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]
LEFT_COLOR = (80, 220, 80)   # BGR
RIGHT_COLOR = (80, 140, 255)
# Solid mesh only: vivid blue (BGR) so right-hand mesh reads clearly vs joint overlay colors.
RIGHT_SOLID_MESH_BGR = (255, 120, 60)
GT_COLOR = (255, 255, 255)
MANO_BOTH_COLOR = (220, 80, 220)   # magenta-ish
HOT3D_BOTH_COLOR = (0, 230, 230)   # cyan-ish
HAND_EDGES = [
    # thumb
    (5, 6), (6, 7), (7, 0),
    # index
    (5, 8), (8, 9), (9, 10), (10, 1),
    # middle
    (5, 11), (11, 12), (12, 13), (13, 2),
    # ring
    (5, 14), (14, 15), (15, 16), (16, 3),
    # pinky
    (5, 17), (17, 18), (18, 19), (19, 4),
]
MANO_EDGES = [
    # thumb (wrist=0, tip=16)
    (0, 1), (1, 2), (2, 3), (3, 16),
    # index (tip=17)
    (0, 4), (4, 5), (5, 6), (6, 17),
    # middle (tip=18)
    (0, 7), (7, 8), (8, 9), (9, 18),
    # ring (tip=19)
    (0, 10), (10, 11), (11, 12), (12, 19),
    # pinky (tip=20)
    (0, 13), (13, 14), (14, 15), (15, 20),
]
MISSING_MANO_IDX = sorted(set(range(21)) - set(HOT3D_TO_MANO))


def align_se3_traj(t_src: np.ndarray, t_dst: np.ndarray):
    c_src = t_src.mean(axis=0)
    c_dst = t_dst.mean(axis=0)
    X = t_src - c_src
    Y = t_dst - c_dst
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    return R, t


def project_points_world_to_image(points_world: np.ndarray, T_world_cam: np.ndarray, K: np.ndarray):
    T_cam_world = np.linalg.inv(T_world_cam).astype(np.float32)
    pts_h = np.concatenate([points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam_h = (T_cam_world @ pts_h.T).T
    pts_cam = pts_cam_h[:, :3]

    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((points_world.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        x = pts_cam[valid, 0] / z[valid]
        y = pts_cam[valid, 1] / z[valid]
        uv[valid, 0] = K[0, 0] * x + K[0, 2]
        uv[valid, 1] = K[1, 1] * y + K[1, 2]
    return uv, valid


def load_gt_frame_map(seq_gt_dir: str):
    out = {}
    for p in glob(os.path.join(seq_gt_dir, "*.npz")):
        name = os.path.basename(p)
        if name.endswith("_cache.npz") or name == "object_surface_cache.npz":
            continue
        stem = os.path.splitext(name)[0]
        try:
            ts = int(stem)
        except ValueError:
            continue
        out[ts] = p
    return out


def sorted_image_files(img_dir: str):
    paths = glob(os.path.join(img_dir, "*.jpg"))
    if not paths:
        return []

    def _key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(stem)
        except ValueError:
            return stem

    return sorted(paths, key=_key)


def draw_hand_skeleton(img: np.ndarray, uv: np.ndarray, vis: np.ndarray, color, edges):
    for a, b in edges:
        if bool(vis[a]) and bool(vis[b]):
            xa, ya = int(round(float(uv[a, 0]))), int(round(float(uv[a, 1])))
            xb, yb = int(round(float(uv[b, 0]))), int(round(float(uv[b, 1])))
            cv2.line(img, (xa, ya), (xb, yb), color, 2, lineType=cv2.LINE_AA)


def build_mesh_edges(faces: np.ndarray):
    edges = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return sorted(edges)


def draw_mesh_wireframe(img: np.ndarray, uv: np.ndarray, vis: np.ndarray, edges, color, stride: int):
    if stride < 1:
        stride = 1
    for ei in range(0, len(edges), stride):
        a, b = edges[ei]
        if bool(vis[a]) and bool(vis[b]):
            xa, ya = int(round(float(uv[a, 0]))), int(round(float(uv[a, 1])))
            xb, yb = int(round(float(uv[b, 0]))), int(round(float(uv[b, 1])))
            cv2.line(img, (xa, ya), (xb, yb), color, 1, lineType=cv2.LINE_AA)


def draw_legend(img: np.ndarray, items):
    """
    Draw a compact legend box.
    items: list of (label, bgr_color)
    """
    if not items:
        return
    x0, y0 = 12, 34
    line_h = 18
    box_w = 230
    box_h = 10 + line_h * len(items) + 8
    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 16), (x0 - 8 + box_w, y0 - 16 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
    for i, (label, color) in enumerate(items):
        yy = y0 + i * line_h
        cv2.line(img, (x0, yy), (x0 + 24, yy), color, 3, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 32, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)


def _bgr_to_rgba01(color_bgr, alpha=0.85):
    b, g, r = color_bgr
    return np.array([r, g, b, int(alpha * 255)], dtype=np.uint8) / 255.0


def render_solid_hands_overlay(
    img_bgr: np.ndarray,
    verts_world: np.ndarray,  # [2, V, 3]
    valid_mask: np.ndarray,   # [2]
    T_world_cam: np.ndarray,
    K: np.ndarray,
    faces_right: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Solid render hands with pyrender in camera frame, then alpha-blend on input image.
    Falls back silently if pyrender/trimesh is unavailable.
    """
    try:
        import pyrender
        import trimesh
    except Exception:
        return img_bgr

    h, w = img_bgr.shape[:2]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.06, 0.06, 0.06, 1.0])
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=50.0)
    scene.add(camera, pose=np.eye(4, dtype=np.float32))

    # CV camera (+Z forward, +Y down) -> OpenGL camera (-Z forward, +Y up)
    cv_to_gl = np.eye(4, dtype=np.float32)
    cv_to_gl[1, 1] = -1.0
    cv_to_gl[2, 2] = -1.0

    T_cam_world = np.linalg.inv(T_world_cam).astype(np.float32)
    colors = [LEFT_COLOR, RIGHT_SOLID_MESH_BGR]
    for side in range(2):
        if not bool(valid_mask[side]):
            continue
        verts_h = np.concatenate([verts_world[side].astype(np.float32), np.ones((verts_world.shape[1], 1), dtype=np.float32)], axis=1)
        verts_cam_h = (T_cam_world @ verts_h.T).T
        verts_cam = verts_cam_h[:, :3]
        if np.count_nonzero(verts_cam[:, 2] > 1e-4) < 10:
            continue
        verts_cam_gl_h = (cv_to_gl @ np.concatenate([verts_cam, np.ones((verts_cam.shape[0], 1), dtype=np.float32)], axis=1).T).T
        verts_cam_gl = verts_cam_gl_h[:, :3]
        faces_use = faces_right if side == 1 else faces_right[:, [0, 2, 1]]
        mesh_tm = trimesh.Trimesh(vertices=verts_cam_gl, faces=faces_use, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.7,
            baseColorFactor=_bgr_to_rgba01(colors[side], alpha=0.92),
        )
        mesh_pr = pyrender.Mesh.from_trimesh(mesh_tm, material=material, smooth=True)
        scene.add(mesh_pr)

    # Basic lights in camera frame
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    light_pose = np.eye(4, dtype=np.float32)
    light_pose[:3, 3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    overlay_bgr = cv2.cvtColor(color_rgba[..., :3], cv2.COLOR_RGB2BGR)
    out = img_bgr.copy()
    mask = depth > 0
    if np.any(mask):
        a = float(np.clip(alpha, 0.0, 1.0))
        out[mask] = ((1.0 - a) * out[mask] + a * overlay_bgr[mask]).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser(description="Visualize HaWoR projected joints on extracted RGB frames")
    p.add_argument("--sequence_dir", required=True, help="Per-sequence eval folder, e.g. .../per_sequence/P0001_15c4300c")
    p.add_argument("--gt_root", required=True, help="HOT3D GT root containing <sequence_name>/*.npz")
    p.add_argument("--out_dir", default=None, help="Output folder for overlay frames/video")
    p.add_argument("--sequence_folder", default=None, help="HOT3D dataset sequence folder (for exact calibration), e.g. /data/hot3d/dataset/P0001_xxx")
    p.add_argument("--object_library_folder", default=None, help="HOT3D object_library folder (for provider init)")
    p.add_argument("--rgb_stream_id", default="214-1", help="Aria RGB stream id for calibration lookup")
    p.add_argument("--max_frames", type=int, default=0, help="0 means all frames")
    p.add_argument("--fps", type=int, default=30, help="Output video fps")
    p.add_argument("--draw_gt", action="store_true", help="Overlay GT hand_landmarks as white circles")
    p.add_argument("--pred_convention", choices=["mano", "hot3d"], default="mano",
                   help="mano: visualize raw MANO-21 joints (no mapping). hot3d: map pred joints to HOT3D-20 for GT-aligned comparison.")
    p.add_argument("--draw_both_conventions", action="store_true",
                   help="Overlay both conventions: MANO as lines+dots, HOT3D as lines+crosses, and show mapping error.")
    p.add_argument("--draw_mesh", action="store_true", help="Overlay predicted hand mesh wireframe")
    p.add_argument("--mesh_edge_stride", type=int, default=8, help="Draw every N-th mesh edge to reduce clutter/cost")
    p.add_argument("--solid_render", action="store_true",
                   help="Render solid hand meshes (pyrender) and alpha-blend onto image.")
    p.add_argument("--solid_alpha", type=float, default=0.55, help="Alpha for solid mesh overlay.")
    p.add_argument("--no_video", action="store_true", help="Do not export mp4, only png frames")
    args = p.parse_args()

    seq_dir = os.path.abspath(args.sequence_dir)
    seq_name = os.path.basename(seq_dir.rstrip("/"))
    out_dir = args.out_dir or os.path.join(seq_dir, "joint_overlay")
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(seq_dir, "vrs_meta.npz")
    world_res_path = os.path.join(seq_dir, "world_space_res.pth")
    img_dir = os.path.join(seq_dir, "extracted_images")
    slam_candidates = sorted(glob(os.path.join(seq_dir, "SLAM", "hawor_slam_w_scale_*.npz")))
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    if not os.path.isfile(world_res_path):
        raise FileNotFoundError(f"Missing {world_res_path}")
    if not slam_candidates:
        raise FileNotFoundError(f"Missing SLAM/hawor_slam_w_scale_*.npz under {seq_dir}")

    slam_npz = slam_candidates[-1]
    meta = np.load(meta_path)
    timestamps_ns = np.asarray(meta["timestamps_ns"], dtype=np.int64)
    focal = float(meta["focal"])
    img_center = np.asarray(meta["img_center"], dtype=np.float32)
    K = np.array([[focal, 0.0, img_center[0]], [0.0, focal, img_center[1]], [0.0, 0.0, 1.0]], dtype=np.float32)
    # Prefer exact pinhole intrinsics from HOT3D provider when available.
    if (
        args.sequence_folder
        and args.object_library_folder
        and LINEAR is not None
        and StreamId is not None
        and Hot3dDataProvider is not None
    ):
        try:
            if ObjectLibrary is not None and os.path.isfile(os.path.join(args.object_library_folder, "instance.json")):
                object_library = ObjectLibrary.from_folder(args.object_library_folder)
            elif load_object_library is not None:
                object_library = load_object_library(args.object_library_folder)
            else:
                object_library = None
            hot3d_dp = Hot3dDataProvider(
                sequence_folder=args.sequence_folder,
                object_library=object_library,
                fail_on_missing_data=False,
            )
            stream_id = StreamId(args.rgb_stream_id)
            _, calib = hot3d_dp.device_data_provider.get_camera_calibration(stream_id, camera_model=LINEAR)
            fx, fy = calib.get_focal_lengths()
            cx, cy = calib.get_principal_point()
            K = np.array([[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]], dtype=np.float32)
            print(f"[calib] Using provider intrinsics fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
        except Exception as e:
            print(f"[calib] WARN: provider calibration unavailable, fallback to vrs_meta K: {e}")
    else:
        print("[calib] Using vrs_meta intrinsics (fx=focal, fy=focal, cx=img_center[0], cy=img_center[1])")

    imgfiles = sorted_image_files(img_dir)
    if not imgfiles:
        raise FileNotFoundError(f"No jpg frames found under {img_dir}")

    # 1) Predicted joints in SLAM world
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res_path)
    pred_trans = pred_trans.float()
    pred_rot = pred_rot.float()
    pred_hand_pose = pred_hand_pose.float()
    pred_betas = pred_betas.float()
    if pred_hand_pose.ndim == 3 and pred_hand_pose.shape[-1] == 45:
        pred_hand_pose = pred_hand_pose.view(*pred_hand_pose.shape[:2], 15, 3)

    with torch.no_grad():
        left_out = run_mano_left(pred_trans[0:1], pred_rot[0:1], pred_hand_pose[0:1], betas=pred_betas[0:1], use_cuda=False)
        right_out = run_mano(pred_trans[1:2], pred_rot[1:2], pred_hand_pose[1:2], betas=pred_betas[1:2], use_cuda=False)

    left_j_mano = left_out["joints"][0].cpu().numpy()    # [T,21,3]
    right_j_mano = right_out["joints"][0].cpu().numpy()  # [T,21,3]
    left_j_hot3d = left_j_mano[:, HOT3D_TO_MANO, :]      # [T,20,3]
    right_j_hot3d = right_j_mano[:, HOT3D_TO_MANO, :]
    left_v = left_out["vertices"][0].cpu().numpy()  # [T,778,3]
    right_v = right_out["vertices"][0].cpu().numpy()
    joints_slam_mano = np.stack([left_j_mano, right_j_mano], axis=1)      # [T,2,21,3]
    joints_slam_hot3d = np.stack([left_j_hot3d, right_j_hot3d], axis=1)    # [T,2,20,3]
    if args.pred_convention == "hot3d":
        joints_slam = joints_slam_hot3d
        pred_edges = HAND_EDGES
        pred_wrist_idx = 5
    else:
        joints_slam = joints_slam_mano
        pred_edges = MANO_EDGES
        pred_wrist_idx = 0
    verts_slam = np.stack([left_v, right_v], axis=1)   # [T,2,778,3]

    if torch.is_tensor(pred_valid):
        pv = pred_valid.detach().cpu().float().numpy()
    else:
        pv = np.asarray(pred_valid, dtype=np.float32)
    pred_valid_np = (pv > 0.5).T if pv.shape[0] == 2 else (pv > 0.5)

    # 2) Compute SE3 alignment from keyframe camera centers
    slam_data = dict(np.load(slam_npz, allow_pickle=True))
    traj = slam_data["traj"]                 # [T,7]
    scale = float(slam_data["scale"])
    tstamp = slam_data["tstamp"].astype(int)
    t_slam_kf = traj[tstamp, :3] * scale

    gt_map = load_gt_frame_map(os.path.join(os.path.abspath(args.gt_root), seq_name))
    if not gt_map:
        raise RuntimeError(f"No GT npz frames found under {os.path.join(args.gt_root, seq_name)}")

    gt_centers = []
    valid_kf = []
    for i, fi in enumerate(tstamp):
        if fi < 0 or fi >= len(timestamps_ns):
            continue
        ts = int(timestamps_ns[fi])
        gt_path = gt_map.get(ts)
        if gt_path is None:
            continue
        with np.load(gt_path, allow_pickle=True) as npz:
            if "T_world_cam" not in npz.files:
                continue
            T_world_cam = npz["T_world_cam"].astype(np.float32)
        gt_centers.append(T_world_cam[:3, 3])
        valid_kf.append(i)

    if len(valid_kf) < 4:
        raise RuntimeError(f"Too few valid keyframes for alignment: {len(valid_kf)}")

    t_slam_valid = t_slam_kf[valid_kf]
    t_gt_valid = np.stack(gt_centers, axis=0)
    R_align, t_align = align_se3_traj(t_slam_valid, t_gt_valid)

    n_frames = min(len(imgfiles), joints_slam.shape[0], len(timestamps_ns))
    joints_aligned_mano = (R_align.astype(np.float32) @ joints_slam_mano.reshape(-1, 3).T).T.reshape(joints_slam_mano.shape) + t_align.astype(np.float32)
    joints_aligned_hot3d = (R_align.astype(np.float32) @ joints_slam_hot3d.reshape(-1, 3).T).T.reshape(joints_slam_hot3d.shape) + t_align.astype(np.float32)
    joints_aligned = joints_aligned_hot3d if args.pred_convention == "hot3d" else joints_aligned_mano
    if args.draw_gt and args.pred_convention == "mano":
        print("[warn] draw_gt with pred_convention=mano mixes conventions (GT=HOT3D20 vs pred=MANO21).")

    verts_aligned = (R_align.astype(np.float32) @ verts_slam.reshape(-1, 3).T).T.reshape(verts_slam.shape) + t_align.astype(np.float32)
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)

    mesh_edges = None
    if args.draw_mesh:
        mesh_edges = build_mesh_edges(get_mano_faces())
    faces_right = get_mano_faces().astype(np.int32)

    video_writer = None
    mapping_err3d = []
    if args.pred_convention == "mano":
        pred_main_label = "Pred-MANO"
    else:
        pred_main_label = "Pred-HOT3D"
    pred_alt_label = "Pred-HOT3D" if args.pred_convention == "mano" else "Pred-MANO"

    for fi in range(n_frames):
        img = cv2.imread(imgfiles[fi], cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        ts = int(timestamps_ns[fi])
        gt_path = gt_map.get(ts)
        if gt_path is None:
            continue
        with np.load(gt_path, allow_pickle=True) as npz:
            if "T_world_cam" not in npz.files:
                continue
            T_world_cam = npz["T_world_cam"].astype(np.float32)
            gt_hand = npz["hand_landmarks"].astype(np.float32) if ("hand_landmarks" in npz.files and args.draw_gt) else None
            gt_valid = npz["hand_valid"].astype(bool) if "hand_valid" in npz.files else np.array([True, True], dtype=bool)

        for side, color in ((0, LEFT_COLOR), (1, RIGHT_COLOR)):
            if not bool(pred_valid_np[fi, side]):
                continue
            # Default single-convention drawing
            if not args.draw_both_conventions:
                uv, vis = project_points_world_to_image(joints_aligned[fi, side], T_world_cam, K)
                draw_hand_skeleton(img, uv, vis, color, pred_edges)
                for j in range(uv.shape[0]):
                    if not vis[j]:
                        continue
                    x, y = int(round(float(uv[j, 0]))), int(round(float(uv[j, 1])))
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(img, (x, y), 3, color, -1, lineType=cv2.LINE_AA)

                # wrist highlight
                if vis[pred_wrist_idx]:
                    xw, yw = int(round(float(uv[pred_wrist_idx, 0]))), int(round(float(uv[pred_wrist_idx, 1])))
                    if 0 <= xw < W and 0 <= yw < H:
                        cv2.circle(img, (xw, yw), 5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            if mesh_edges is not None:
                uv_v, vis_v = project_points_world_to_image(verts_aligned[fi, side], T_world_cam, K)
                draw_mesh_wireframe(img, uv_v, vis_v, mesh_edges, color, args.mesh_edge_stride)

            if args.draw_gt and gt_hand is not None and bool(gt_valid[side]):
                uv_gt, vis_gt = project_points_world_to_image(gt_hand[side], T_world_cam, K)
                draw_hand_skeleton(img, uv_gt, vis_gt, GT_COLOR, HAND_EDGES)
                for j in range(uv_gt.shape[0]):
                    if not vis_gt[j]:
                        continue
                    xg, yg = int(round(float(uv_gt[j, 0]))), int(round(float(uv_gt[j, 1])))
                    if 0 <= xg < W and 0 <= yg < H:
                        cv2.circle(img, (xg, yg), 2, GT_COLOR, -1, lineType=cv2.LINE_AA)

            if args.draw_both_conventions:
                mano = joints_aligned_mano[fi, side]
                hot = joints_aligned_hot3d[fi, side]
                mano20 = mano[HOT3D_TO_MANO]
                err = np.linalg.norm(mano20 - hot, axis=-1)
                mapping_err3d.append(float(err.mean()))

                uv_m, vis_m = project_points_world_to_image(mano, T_world_cam, K)
                uv_h, vis_h = project_points_world_to_image(hot, T_world_cam, K)
                # Pred-MANO: lines + dots
                draw_hand_skeleton(img, uv_m, vis_m, MANO_BOTH_COLOR, MANO_EDGES)
                for j in range(uv_m.shape[0]):
                    if bool(vis_m[j]):
                        xm, ym = int(round(float(uv_m[j, 0]))), int(round(float(uv_m[j, 1])))
                        if 0 <= xm < W and 0 <= ym < H:
                            cv2.circle(img, (xm, ym), 4, MANO_BOTH_COLOR, -1, lineType=cv2.LINE_AA)
                # Pred-HOT3D: crosses only (no skeleton lines)
                for j in range(uv_h.shape[0]):
                    if bool(vis_h[j]):
                        xh, yh = int(round(float(uv_h[j, 0]))), int(round(float(uv_h[j, 1])))
                        if 0 <= xh < W and 0 <= yh < H:
                            cv2.drawMarker(img, (xh, yh), HOT3D_BOTH_COLOR, markerType=cv2.MARKER_CROSS, markerSize=7, thickness=1, line_type=cv2.LINE_AA)

        if args.solid_render:
            img = render_solid_hands_overlay(
                img_bgr=img,
                verts_world=verts_aligned[fi],
                valid_mask=pred_valid_np[fi],
                T_world_cam=T_world_cam,
                K=K,
                    faces_right=faces_right,
                alpha=args.solid_alpha,
            )

        legend_items = [(pred_main_label, RIGHT_COLOR)]
        if args.draw_both_conventions:
            legend_items = [("Pred-MANO (line+dot)", MANO_BOTH_COLOR), ("Pred-HOT3D (cross only)", HOT3D_BOTH_COLOR)]
        if args.draw_gt:
            legend_items.append(("GT", GT_COLOR))
        draw_legend(img, legend_items)
        if args.draw_both_conventions and mapping_err3d:
            cv2.putText(
                img,
                f"map err frame mean: {mapping_err3d[-1]:.6e} m",
                (12, img.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(img, f"{seq_name}  frame={fi}  ts={ts}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        out_png = os.path.join(out_dir, f"frame_{fi:05d}.png")
        cv2.imwrite(out_png, img)

        if not args.no_video:
            if video_writer is None:
                out_mp4 = os.path.join(out_dir, f"{seq_name}_joints_overlay.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(out_mp4, fourcc, float(args.fps), (W, H))
            video_writer.write(img)

    if video_writer is not None:
        video_writer.release()

    if args.draw_both_conventions and mapping_err3d:
        arr = np.asarray(mapping_err3d, dtype=np.float64)
        print(
            f"[mapping_check] mean_3d_err={arr.mean():.9f} m  "
            f"p95_3d_err={np.percentile(arr,95):.9f} m  max_3d_err={arr.max():.9f} m"
        )

    print(f"[done] Saved overlays to: {out_dir}")


if __name__ == "__main__":
    main()

