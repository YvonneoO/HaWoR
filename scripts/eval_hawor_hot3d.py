#!/usr/bin/env python3
"""
eval_hawor_hot3d.py — HaWoR inference + evaluation on HOT3D VRS recordings.

Pipeline
────────
1  extract_frames_from_vrs   read undistorted pinhole frames from VRS
2  detect_track              HaWoR hand detector + ByteTrack
3  hawor_slam                DROID-SLAM + Metric3D scale → SLAM/hawor_slam_w_scale_*.npz
4  hawor_motion_estimation   camera-space HAWOR → cam_space/ + model_masks.npy
5  hawor_infiller            cam→world (SLAM) + infiller → world_space_res.pth
6  align_slam_to_hot3d       SE3 align SLAM camera traj → GT camera traj (no scale)
7  evaluate                  abs-MPJPE / W-MPJPE / WA-MPJPE / PA-MPJPE / MPVPE

World-frame notes
─────────────────
- HaWoR SLAM world frame has an arbitrary origin (first DROID-SLAM keyframe).
- HOT3D GT is in OptiTrack world frame.
- Step 6 finds (R_align, t_align) via SVD on matched camera positions so that
      t_gt_cam ≈ R_align @ t_slam_cam + t_align
  then applies the same SE3 to all predicted hand joints before metric computation.
- Scale is NOT estimated (Metric3D already gives metric depth).

Joint index mapping
───────────────────
- HaWoR outputs 21 MANO joints; HOT3D GT has 20 landmarks in HOT3D order.
- HOT3D_TO_MANO selects from MANO output to produce HOT3D-ordered joints.
- HOT3D hand index 0 = left, 1 = right — matches HaWoR convention.

Usage
─────
Single sequence (full pipeline):
  python scripts/eval_hawor_hot3d.py \\
      --checkpoint  weights/hawor/checkpoints/epoch=....ckpt \\
      --infiller_weight weights/hawor/infiller.pth \\
      --sequence_folder $HOT3D_REPO_DIR/dataset/P0003_02 \\
      --object_library_folder $HOT3D_REPO_DIR/dataset/object_library \\
      --gt_root $HOT3D_GT_DIR/val \\
      --sequence_name P0003_02 \\
      --out_dir $STAGE1_DIR

Re-evaluate from existing world_space_res.pth (skip inference):
  add --skip_inference

All sequences under a split:
  add --all_sequences  (scans --gt_root for sub-dirs and matches --sequence_folder parent)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import json
import types
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import torch
import joblib
from tqdm import tqdm
from glob import glob
from natsort import natsorted

# ── HOT3D data provider ────────────────────────────────────────────────────────
_HOT3D_REPO = os.environ.get('HOT3D_REPO_DIR', '/lp-dev/qianqian/hot3d-private/hot3d')
sys.path.insert(0, _HOT3D_REPO)
from data_loaders.loader_object_library import ObjectLibrary, load_object_library  # noqa: E402
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # noqa: E402
from projectaria_tools.core.calibration import LINEAR                   # noqa: E402
from projectaria_tools.core.stream_id import StreamId                   # noqa: E402

try:
    from dataset_api import Hot3dDataProvider  # noqa: E402
except Exception as _dataset_import_err:
    # Some environments miss pyvrs/pyvrs2 (Quest-only dependency) even for Aria
    # sequences. Provide an Aria-only fallback provider for this eval script.
    from data_loaders.AriaDataProvider import AriaDataProvider  # noqa: E402
    from data_loaders.PathProvider import Hot3dDataPathProvider  # noqa: E402
    from data_loaders.headsets import Headset  # noqa: E402
    from data_loaders.HeadsetPose3dProvider import (  # noqa: E402
        load_headset_pose_provider_from_csv,
    )

    class Hot3dDataProvider:  # type: ignore[no-redef]
        def __init__(self, sequence_folder: str, object_library, mano_hand_model=None, fail_on_missing_data: bool = True):
            self.path_provider = Hot3dDataPathProvider.fromRecordingFolder(
                recording_instance_folderpath=sequence_folder
            )
            if self.path_provider is None or not self.path_provider.is_valid():
                if fail_on_missing_data:
                    raise RuntimeError(f"Invalid HOT3D sequence folder: {sequence_folder}")
            if self.get_device_type() != Headset.Aria:
                raise RuntimeError(
                    f"Aria-only fallback cannot load non-Aria sequence: {sequence_folder}. "
                    f"Original import error: {_dataset_import_err}"
                )
            self._device_data_provider = AriaDataProvider(
                self.path_provider.vrs_filepath,
                self.path_provider.mps_folderpath,
            )
            self._device_pose_provider = load_headset_pose_provider_from_csv(
                self.path_provider.headset_trajectory_filepath
            )

        def get_device_type(self):
            import json as _json
            with open(self.path_provider.scene_metadata_filepath, "r") as f:
                md = _json.load(f)
            return Headset[md["headset"]]

        @property
        def device_data_provider(self):
            return self._device_data_provider

        @property
        def device_pose_data_provider(self):
            return self._device_pose_provider

# ── HaWoR pipeline modules ────────────────────────────────────────────────────
from lib.pipeline.tools import detect_track, parse_chunks_hand_frame   # noqa: E402
from lib.eval_utils.custom_utils import load_slam_cam                  # noqa: E402
from hawor.utils.process import run_mano, run_mano_left                # noqa: E402

# Imported lazily in run_inference() to avoid loading GPU models at import time.
# from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
# from scripts.scripts_test_video.hawor_slam import hawor_slam

# ── Constants ──────────────────────────────────────────────────────────────────
RGB_STREAM_ID = StreamId("214-1")          # Aria RGB camera
TRAJ_SEGMENT_LEN = 100                     # frames per W/WA-MPJPE segment

# HOT3D landmark order → MANO joint indices.
# Selects 20 joints from MANO's 21 to match HOT3D's 20-landmark layout.
HOT3D_TO_MANO = [16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]

# Index of the wrist root in HOT3D-ordered joints.
# HOT3D_TO_MANO[5] == 0 == MANO wrist joint.
WRIST_HOT3D_IDX = 5


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — VRS frame extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames_from_vrs(hot3d_dp: Hot3dDataProvider, stream_id: StreamId,
                             out_folder: str):
    """
    Read undistorted pinhole frames from VRS and save as 000000.jpg, 000001.jpg, …

    Returns
    -------
    timestamps_ns : (T,) int64  — per-frame capture timestamps
    focal         : float        — fx from pinhole calibration (pixels)
    img_center    : (2,) float   — [cx, cy]
    T_device_cam  : (4,4) float  — homogeneous T_device_camera from factory calib
    """
    device_dp = hot3d_dp.device_data_provider
    timestamps = np.array(device_dp.get_sequence_timestamps(
        stream_id, TimeDomain.TIME_CODE), dtype=np.int64)

    # Pinhole calibration — T_device_cam from native FISHEYE624, intrinsics pinhole
    T_device_cam_se3, calib = device_dp.get_camera_calibration(stream_id, camera_model=LINEAR)
    focal = float(calib.get_focal_lengths()[0])
    W, H = calib.get_image_size()
    img_center = np.array([W / 2.0, H / 2.0], dtype=np.float32)
    T_device_cam = np.array(T_device_cam_se3.to_matrix(), dtype=np.float64)  # (4,4)

    os.makedirs(out_folder, exist_ok=True)
    for i, ts in enumerate(tqdm(timestamps, desc='extracting VRS frames')):
        out_path = os.path.join(out_folder, f'{i:06d}.jpg')
        if os.path.exists(out_path):
            continue
        img = device_dp.get_undistorted_image(ts, stream_id)
        if img is None:
            # Write a black placeholder so frame indices stay aligned
            img = np.zeros((int(H), int(W), 3), dtype=np.uint8)
        # Aria images are RGB; OpenCV expects BGR
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return timestamps, focal, img_center, T_device_cam


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — SE3 alignment: SLAM world → HOT3D world
# ══════════════════════════════════════════════════════════════════════════════

def align_se3_traj(t_src: np.ndarray, t_dst: np.ndarray):
    """
    Find SE3 (R, t) — no scale — that maps src positions to dst positions.
      t_dst ≈ R @ t_src + t   (least squares, SVD)

    Parameters
    ----------
    t_src : (N, 3)  SLAM camera positions (metric, after Metric3D scale)
    t_dst : (N, 3)  GT camera positions in HOT3D world

    Returns
    -------
    R_align : (3, 3)
    t_align : (3,)
    """
    assert t_src.shape == t_dst.shape and t_src.ndim == 2
    mu_s = t_src.mean(0)
    mu_d = t_dst.mean(0)
    X = t_src - mu_s
    Y = t_dst - mu_d
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float64)
    t = mu_d - R @ mu_s
    return R, t


def get_gt_cam_positions(hot3d_dp: Hot3dDataProvider, T_device_cam: np.ndarray,
                          timestamps_ns: np.ndarray, frame_indices: np.ndarray):
    """
    Return GT camera-centre positions in HOT3D world for the given frame indices.

    T_world_camera = T_world_device @ T_device_camera
    camera centre  = T_world_camera[:3, 3]
    """
    device_pose_dp = hot3d_dp.device_pose_data_provider
    positions = []
    for fi in frame_indices:
        ts = int(timestamps_ns[fi])
        pose_with_dt = device_pose_dp.get_pose_at_timestamp(
            ts, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE)
        if pose_with_dt is None or pose_with_dt.pose3d.T_world_device is None:
            positions.append(None)
            continue
        T_wd = np.array(pose_with_dt.pose3d.T_world_device.to_matrix(), dtype=np.float64)
        T_wc = T_wd @ T_device_cam          # T_world_camera  (4×4)
        positions.append(T_wc[:3, 3])
    return positions                        # list of (3,) or None


# ══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ══════════════════════════════════════════════════════════════════════════════

def _procrustes_transform(S1: np.ndarray, S2: np.ndarray):
    """
    Similarity transform (scale, R, t) mapping S1 → S2.
    Returns scale, R (3×3), t (3,).
    """
    S1t = S1.T; S2t = S2.T
    mu1 = S1t.mean(1, keepdims=True)
    mu2 = S2t.mean(1, keepdims=True)
    X1 = S1t - mu1; X2 = S2t - mu2
    var1 = float(np.sum(X1 ** 2))
    K = X1 @ X2.T
    U, _, Vt = np.linalg.svd(K)
    Z = np.eye(3); Z[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = float(np.trace(R @ K) / max(var1, 1e-8))
    t = (mu2 - scale * (R @ mu1)).squeeze()
    return scale, R, t


def _apply_sim3(pts: np.ndarray, scale, R, t) -> np.ndarray:
    """Apply Sim3 to (..., 3) array."""
    shape = pts.shape
    flat = pts.reshape(-1, 3)
    out = (scale * (R @ flat.T) + t[:, None]).T
    return out.reshape(shape)


def _pa_mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-frame PA-MPJPE in mm. pred/gt: (N, 3)."""
    s, R, t = _procrustes_transform(pred, gt)
    aligned = _apply_sim3(pred, s, R, t)
    return float(np.linalg.norm(aligned - gt, axis=-1).mean() * 1000.0)


def _compute_ate(traj_pred: np.ndarray, traj_gt: np.ndarray):
    """
    Compute wrist ATE and ATE-S for a single-hand trajectory.

    Parameters
    ----------
    traj_pred : (N, 3)  predicted wrist positions (already SE3-aligned to HOT3D world)
    traj_gt   : (N, 3)  GT wrist positions

    Returns
    -------
    ate_m   : float  — Sim3-aligned mean error (meters); scale normalized away
    ate_s_m : float  — SE3-aligned mean error (meters); scale error preserved
    """
    assert traj_pred.shape == traj_gt.shape and traj_pred.ndim == 2

    # ATE: similarity alignment (scale free)
    s, R, t = _procrustes_transform(traj_pred, traj_gt)
    aligned_sim3 = _apply_sim3(traj_pred, s, R, t)
    ate_m = float(np.linalg.norm(aligned_sim3 - traj_gt, axis=-1).mean())

    # ATE-S: rigid alignment only (scale preserved → penalises Metric3D scale error)
    R_se3, t_se3 = align_se3_traj(traj_pred, traj_gt)
    aligned_se3 = (R_se3 @ traj_pred.T).T + t_se3
    ate_s_m = float(np.linalg.norm(aligned_se3 - traj_gt, axis=-1).mean())

    return ate_m, ate_s_m


def compute_metrics(joints_pred_aligned: np.ndarray,
                    verts_pred_aligned,
                    pred_valid_np: np.ndarray,
                    gt_frame_paths,
                    timestamps_ns=None,
                    segment_len: int = TRAJ_SEGMENT_LEN):
    """
    Compute all hand metrics.

    Parameters
    ----------
    joints_pred_aligned : (T, 2, 20, 3)  — joints in HOT3D world (SE3-aligned)
    verts_pred_aligned  : (T, 2, 778, 3) or None
    pred_valid_np       : (T, 2) bool
    gt_frame_paths      : list of GT npz paths (one per frame, aligned by index)
    timestamps_ns       : (T,) int64 or None — frame timestamps; if None assumes 30 fps
    segment_len         : frames per segment (W/WA-MPJPE, ATE, RTE, Accel)

    Returns
    -------
    dict with keys:
      pa_mpjpe_mm, mpvpe_mm             — per-frame
      w_mpjpe_mm, wa_mpjpe_mm           — per-segment Sim3-aligned MPJPE
      ate_m, ate_s_m                    — per-segment wrist ATE (meters)
      rte_pct                           — per-segment wrist RTE (%)
      accel_ms2                         — per-segment acceleration error (m/s²)
      (all averaged over both hands and all valid segments/frames)
    """
    T = joints_pred_aligned.shape[0]

    # Frame timestamps in seconds (for Accel)
    if timestamps_ns is not None:
        ts_sec = np.asarray(timestamps_ns, dtype=np.float64) / 1e9
    else:
        ts_sec = np.arange(T, dtype=np.float64) / 30.0   # assume 30 fps

    # ── per-frame metrics ──────────────────────────────────────────────────────
    pa_errs, mpvpe_errs = [], []

    # Segment buffers: nan where invalid
    joints_for_seg    = np.full((T, 2, 20, 3), np.nan)
    gt_joints_for_seg = np.full((T, 2, 20, 3), np.nan)

    for t, gt_path in enumerate(gt_frame_paths):
        with np.load(gt_path, allow_pickle=True) as npz:
            if 'hand_landmarks' not in npz.files:
                continue
            gt_j = npz['hand_landmarks'].astype(np.float32)   # (2, 20, 3)
            gt_valid = (npz['hand_valid'].astype(bool)
                        if 'hand_valid' in npz.files
                        else np.ones(2, dtype=bool))
            gt_v = (npz['hand_vertices'].astype(np.float32)
                    if 'hand_vertices' in npz.files else None)

        for side in range(2):
            if not gt_valid[side] or not pred_valid_np[t, side]:
                continue
            pj = joints_pred_aligned[t, side]   # (20, 3)
            gj = gt_j[side]                     # (20, 3)

            pa_errs.append(_pa_mpjpe_mm(pj, gj))
            joints_for_seg[t, side]    = pj
            gt_joints_for_seg[t, side] = gj

            if verts_pred_aligned is not None and gt_v is not None:
                pv = verts_pred_aligned[t, side]
                gv = gt_v[side]
                mpvpe_errs.append(float(np.linalg.norm(pv - gv, axis=-1).mean() * 1000.0))

    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else float('nan')

    # ── per-segment metrics ────────────────────────────────────────────────────
    w_errs, wa_errs = [], []
    ate_vals, ate_s_vals = [], []
    rte_vals = []
    accel_vals = []

    for seg_start in range(0, T, segment_len):
        seg_end  = min(seg_start + segment_len, T)
        seg_ts   = ts_sec[seg_start:seg_end]

        for side in range(2):
            seg_pred = joints_for_seg[seg_start:seg_end, side]     # (S, 20, 3)
            seg_gt   = gt_joints_for_seg[seg_start:seg_end, side]

            valid_mask = ~(np.isnan(seg_pred).any(axis=(-1, -2)) |
                           np.isnan(seg_gt).any(axis=(-1, -2)))
            if valid_mask.sum() < 4:
                continue

            vp  = seg_pred[valid_mask]    # (V, 20, 3)
            vg  = seg_gt[valid_mask]
            vts = seg_ts[valid_mask]      # (V,)

            # W-MPJPE: Sim3 from first 2 valid frames
            anchor = min(2, vp.shape[0])
            s, R, t_sim = _procrustes_transform(
                vp[:anchor].reshape(-1, 3), vg[:anchor].reshape(-1, 3))
            aligned_w = _apply_sim3(vp, s, R, t_sim)
            w_errs.append(float(np.linalg.norm(aligned_w - vg, axis=-1).mean() * 1000.0))

            # WA-MPJPE: Sim3 from whole segment
            s, R, t_sim = _procrustes_transform(vp.reshape(-1, 3), vg.reshape(-1, 3))
            aligned_wa = _apply_sim3(vp, s, R, t_sim)
            wa_errs.append(float(np.linalg.norm(aligned_wa - vg, axis=-1).mean() * 1000.0))

            # Wrist positions for ATE / RTE
            wp = vp[:, WRIST_HOT3D_IDX]   # (V, 3)
            wg = vg[:, WRIST_HOT3D_IDX]

            # ATE (Sim3) and ATE-S (SE3) — per segment
            ate_m, ate_s_m = _compute_ate(wp, wg)
            ate_vals.append(ate_m)
            ate_s_vals.append(ate_s_m)

            # RTE: rigid-align wrist traj, mean error / GT displacement * 100%
            R_rte, t_rte = align_se3_traj(wp, wg)
            wp_aligned   = (R_rte @ wp.T).T + t_rte
            mean_err_rte = float(np.linalg.norm(wp_aligned - wg, axis=-1).mean())
            gt_disp      = float(np.linalg.norm(np.diff(wg, axis=0), axis=-1).sum())
            if gt_disp > 1e-6:
                rte_vals.append(mean_err_rte / gt_disp * 100.0)

            # Accel: second-derivative of mean joint position (m/s²)
            if vp.shape[0] >= 3:
                pp = vp.mean(axis=1)   # (V, 3) — mean over 20 joints
                gp = vg.mean(axis=1)
                dt = float(np.mean(np.diff(vts))) if len(vts) > 1 else 1.0 / 30.0
                pred_accel = (pp[2:] - 2 * pp[1:-1] + pp[:-2]) / (dt ** 2)   # (V-2, 3)
                gt_accel   = (gp[2:] - 2 * gp[1:-1] + gp[:-2]) / (dt ** 2)
                accel_err  = np.linalg.norm(pred_accel - gt_accel, axis=-1)   # (V-2,) m/s²
                accel_vals.extend(accel_err.tolist())

    return {
        'pa_mpjpe_mm':    _safe_mean(pa_errs),
        'mpvpe_mm':       _safe_mean(mpvpe_errs),
        'w_mpjpe_mm':     _safe_mean(w_errs),
        'wa_mpjpe_mm':    _safe_mean(wa_errs),
        'ate_m':          _safe_mean(ate_vals),
        'ate_s_m':        _safe_mean(ate_s_vals),
        'rte_pct':        _safe_mean(rte_vals),
        'accel_ms2':      _safe_mean(accel_vals),
        'n_valid_frames': len(pa_errs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GT frame path loading (reuse logic from eval_hot3d_headless.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_gt_frame_map(seq_gt_dir: str):
    import json as _json
    manifest = os.path.join(seq_gt_dir, 'manifest.jsonl')
    if os.path.isfile(manifest):
        ts_to_path = {}
        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                ts = rec.get('timestamp_ns')
                if ts is None:
                    continue
                rel = rec.get('npz_relpath') or f"{rec['timestamp_ns']}.npz"
                p = os.path.join(seq_gt_dir, rel)
                if os.path.isfile(p):
                    ts_to_path[int(ts)] = p
        if ts_to_path:
            return ts_to_path

    ts_to_path = {}
    for name in os.listdir(seq_gt_dir):
        if not name.endswith('.npz'):
            continue
        if name.endswith('_cache.npz') or name == 'object_surface_cache.npz':
            continue
        stem = os.path.splitext(name)[0]
        try:
            ts_to_path[int(stem)] = os.path.join(seq_gt_dir, name)
        except ValueError:
            pass
    return ts_to_path


# ══════════════════════════════════════════════════════════════════════════════
# Per-sequence runner
# ══════════════════════════════════════════════════════════════════════════════

def run_sequence(args, sequence_name: str, sequence_folder: str):
    """Full pipeline for one HOT3D sequence. Returns metrics dict."""

    seq_folder = os.path.join(args.out_dir, sequence_name)
    os.makedirs(seq_folder, exist_ok=True)
    img_folder = os.path.join(seq_folder, 'extracted_images')

    # fake video_path so hawor_* functions derive the correct seq_folder
    fake_video_path = os.path.join(args.out_dir, f'{sequence_name}.mp4')

    use_cuda = torch.cuda.is_available()

    # ── load HOT3D data provider ───────────────────────────────────────────────
    # HOT3D API differs by version: newer code exposes load_object_library()
    # while some forks may provide ObjectLibrary.from_folder().
    # Some server datasets miss object_library/instance.json entirely. For this
    # eval path we only need Aria image/pose providers, so allow an empty
    # object library fallback instead of hard-failing.
    instance_json = os.path.join(args.object_library_folder, "instance.json")
    if os.path.isfile(instance_json):
        if hasattr(ObjectLibrary, "from_folder"):
            object_library = ObjectLibrary.from_folder(args.object_library_folder)
        else:
            object_library = load_object_library(args.object_library_folder)
    else:
        print(f"[WARN] instance.json not found under {args.object_library_folder}; using empty object library fallback")
        object_library = ObjectLibrary({}, args.object_library_folder)
    hot3d_dp = Hot3dDataProvider(
        sequence_folder=sequence_folder,
        object_library=object_library,
        fail_on_missing_data=False,
    )

    # ── Step 1: extract frames ─────────────────────────────────────────────────
    meta_path  = os.path.join(seq_folder, 'vrs_meta.npz')

    if not args.skip_inference and (
            not os.path.isfile(meta_path) or
            len(glob(os.path.join(img_folder, '*.jpg'))) == 0):
        print(f'[{sequence_name}] extracting VRS frames …')
        timestamps_ns, focal, img_center, T_device_cam = extract_frames_from_vrs(
            hot3d_dp, RGB_STREAM_ID, img_folder)
        np.savez(meta_path,
                 timestamps_ns=timestamps_ns,
                 focal=np.float64(focal),
                 img_center=img_center,
                 T_device_cam=T_device_cam)
    else:
        meta = np.load(meta_path)
        timestamps_ns = meta['timestamps_ns']
        focal         = float(meta['focal'])
        img_center    = meta['img_center']
        T_device_cam  = meta['T_device_cam']

    imgfiles = natsorted(glob(os.path.join(img_folder, '*.jpg')))
    n_frames = len(imgfiles)
    start_idx, end_idx = 0, n_frames

    if not args.skip_inference:
        # ── Step 2: detection + tracking ──────────────────────────────────────
        track_dir = os.path.join(seq_folder, f'tracks_{start_idx}_{end_idx}')
        os.makedirs(track_dir, exist_ok=True)
        if not os.path.exists(os.path.join(track_dir, 'model_tracks.npy')):
            print(f'[{sequence_name}] detect + track …')
            boxes_, tracks_ = detect_track(imgfiles, thresh=0.2)
            np.save(os.path.join(track_dir, 'model_boxes.npy'), boxes_)
            np.save(os.path.join(track_dir, 'model_tracks.npy'), tracks_)

        # ── Steps 3-5: SLAM + motion estimation + infiller ────────────────────
        # Import here to defer heavy GPU initialisation
        from scripts.scripts_test_video.hawor_video import (
            hawor_motion_estimation, hawor_infiller)
        from scripts.scripts_test_video.hawor_slam import hawor_slam

        hw_args = types.SimpleNamespace(
            video_path=fake_video_path,
            img_focal=focal,
            checkpoint=args.checkpoint,
            infiller_weight=args.infiller_weight,
        )

        print(f'[{sequence_name}] hawor_motion_estimation …')
        frame_chunks_all, _ = hawor_motion_estimation(
            hw_args, start_idx, end_idx, seq_folder)

        print(f'[{sequence_name}] hawor_slam …')
        hawor_slam(hw_args, start_idx, end_idx)

        print(f'[{sequence_name}] hawor_infiller …')
        hawor_infiller(hw_args, start_idx, end_idx, frame_chunks_all)

    # ── Step 6: SE3 align SLAM world → HOT3D world ────────────────────────────
    slam_npz = os.path.join(seq_folder, 'SLAM',
                            f'hawor_slam_w_scale_{start_idx}_{end_idx}.npz')
    if not os.path.isfile(slam_npz):
        raise FileNotFoundError(f'SLAM file not found: {slam_npz}')

    slam_data = dict(np.load(slam_npz, allow_pickle=True))
    traj  = slam_data['traj']                        # (T, 7) all frames
    scale = float(slam_data['scale'])
    tstamp = slam_data['tstamp'].astype(int)          # keyframe frame indices

    # SLAM camera positions at keyframes (metric scale applied)
    t_slam_kf = traj[tstamp, :3] * scale             # (K, 3)

    # GT camera positions at those keyframes
    gt_positions_list = get_gt_cam_positions(
        hot3d_dp, T_device_cam, timestamps_ns, tstamp)

    # Filter out frames where GT pose is unavailable
    valid_kf = [i for i, p in enumerate(gt_positions_list) if p is not None]
    if len(valid_kf) < 4:
        raise RuntimeError(
            f'[{sequence_name}] too few keyframes with GT poses ({len(valid_kf)})')

    t_slam_valid = t_slam_kf[valid_kf]
    t_gt_valid   = np.stack([gt_positions_list[i] for i in valid_kf])

    R_align, t_align = align_se3_traj(t_slam_valid, t_gt_valid)
    print(f'[{sequence_name}] SE3 alignment residual: '
          f'{np.linalg.norm(t_gt_valid - (R_align @ t_slam_valid.T).T - t_align, axis=1).mean()*100:.1f} cm')

    # ── Step 7: MANO forward → joints in SLAM world → apply SE3 ──────────────
    world_res_path = os.path.join(seq_folder, 'world_space_res.pth')
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res_path)

    pred_trans     = pred_trans.float()
    pred_rot       = pred_rot.float()
    pred_hand_pose = pred_hand_pose.float()
    pred_betas     = pred_betas.float()

    if pred_hand_pose.ndim == 3 and pred_hand_pose.shape[-1] == 45:
        pred_hand_pose = pred_hand_pose.view(*pred_hand_pose.shape[:2], 15, 3)

    with torch.no_grad():
        left_out  = run_mano_left(pred_trans[0:1], pred_rot[0:1],
                                  pred_hand_pose[0:1], betas=pred_betas[0:1],
                                  use_cuda=use_cuda)
        right_out = run_mano(pred_trans[1:2], pred_rot[1:2],
                             pred_hand_pose[1:2], betas=pred_betas[1:2],
                             use_cuda=use_cuda)

    # joints: (1, T, 21, 3) → select HOT3D-ordered 20 joints
    left_j  = left_out['joints'][0, :, HOT3D_TO_MANO, :].cpu().numpy()   # (T, 20, 3)
    right_j = right_out['joints'][0, :, HOT3D_TO_MANO, :].cpu().numpy()
    left_v  = left_out['vertices'][0].cpu().numpy()                        # (T, 778, 3)
    right_v = right_out['vertices'][0].cpu().numpy()

    # Stack: dim 0 = left (0), right (1) — matches HOT3D convention
    joints_slam = np.stack([left_j, right_j], axis=1)   # (T, 2, 20, 3)
    verts_slam  = np.stack([left_v, right_v], axis=1)   # (T, 2, 778, 3)

    # Apply SE3 alignment
    R_f = R_align.astype(np.float32)
    t_f = t_align.astype(np.float32)
    joints_aligned = (R_f @ joints_slam.reshape(-1, 3).T).T.reshape(
        n_frames, 2, 20, 3) + t_f
    verts_aligned  = (R_f @ verts_slam.reshape(-1, 3).T).T.reshape(
        n_frames, 2, 778, 3) + t_f

    # joblib may load pred_valid as Tensor or ndarray depending on save path
    if torch.is_tensor(pred_valid):
        pv = pred_valid.detach().cpu().float().numpy()
    else:
        pv = np.asarray(pred_valid, dtype=np.float32)
    if pv.ndim != 2 or 2 not in pv.shape:
        raise ValueError(f'pred_valid expected shape (*, 2) or (2, *), got {pv.shape}')
    pred_valid_np = (pv > 0.5).T if pv.shape[0] == 2 else (pv > 0.5)  # (T, 2)

    # ── Step 8: load GT frames and compute metrics ────────────────────────────
    seq_gt_dir = os.path.join(args.gt_root, sequence_name)
    if not os.path.isdir(seq_gt_dir):
        raise FileNotFoundError(f'GT sequence directory not found: {seq_gt_dir}')

    gt_frame_map = load_gt_frame_map(seq_gt_dir)
    if not gt_frame_map:
        raise RuntimeError(f'[{sequence_name}] no GT npz frames found under {seq_gt_dir}')

    pred_ts = np.asarray(timestamps_ns[:n_frames], dtype=np.int64)
    matched_pred_idx = []
    matched_gt_paths = []
    for i, ts in enumerate(pred_ts):
        p = gt_frame_map.get(int(ts))
        if p is None:
            continue
        matched_pred_idx.append(i)
        matched_gt_paths.append(p)

    if not matched_pred_idx:
        raise RuntimeError(f'[{sequence_name}] no timestamp_ns overlap between prediction frames and GT npz')

    if len(matched_pred_idx) != n_frames:
        print(f'[WARN] {sequence_name}: matched {len(matched_pred_idx)}/{n_frames} frames by timestamp_ns; '
              'metrics run on matched subset only.')

    matched_pred_idx = np.asarray(matched_pred_idx, dtype=np.int64)
    metrics = compute_metrics(
        joints_aligned[matched_pred_idx],
        verts_aligned[matched_pred_idx],
        pred_valid_np[matched_pred_idx],
        matched_gt_paths,
        timestamps_ns=timestamps_ns[matched_pred_idx],
    )
    if args.cleanup_extracted_images:
        shutil.rmtree(img_folder, ignore_errors=True)
        metrics['cleaned_extracted_images'] = True
    else:
        metrics['cleaned_extracted_images'] = False
    metrics['sequence'] = sequence_name
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation + output
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(rows):
    float_keys = ['pa_mpjpe_mm', 'mpvpe_mm',
                  'w_mpjpe_mm', 'wa_mpjpe_mm',
                  'ate_m', 'ate_s_m',
                  'rte_pct', 'accel_ms2']
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in float_keys}
    out['n_sequences'] = len(rows)
    out['n_valid_frames_total'] = sum(r['n_valid_frames'] for r in rows)
    return out


def _parse_gpu_ids(gpu_ids: str):
    ids = [x.strip() for x in str(gpu_ids).split(',') if x.strip() != ""]
    return ids or ["0"]


def _run_sequence_subprocess(args, sequence_name: str, sequence_folder: str, gpu_id: str, per_seq_out_dir: str):
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, script_path,
        '--checkpoint', args.checkpoint,
        '--infiller_weight', args.infiller_weight,
        '--gt_root', args.gt_root,
        '--out_dir', per_seq_out_dir,
        '--object_library_folder', args.object_library_folder,
        '--sequence_name', sequence_name,
        '--sequence_folder', sequence_folder,
    ]
    if args.skip_inference:
        cmd.append('--skip_inference')
    if args.cpu:
        cmd.append('--cpu')

    env = os.environ.copy()
    if not args.cpu:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    ret = subprocess.run(cmd, env=env)
    metrics_path = os.path.join(per_seq_out_dir, f'{sequence_name}_metrics.json')
    if ret.returncode != 0:
        return sequence_name, None, f'worker failed with return code {ret.returncode}'
    if not os.path.isfile(metrics_path):
        return sequence_name, None, f'metrics file not found: {metrics_path}'
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return sequence_name, metrics, None


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing + main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='HaWoR eval on HOT3D VRS')
    p.add_argument('--checkpoint',           required=True)
    p.add_argument('--infiller_weight',      required=True)
    p.add_argument('--gt_root',              required=True,
                   help='HOT3D GT root, e.g. $HOT3D_GT_DIR/val')
    p.add_argument('--out_dir',              required=True,
                   help='Output root; per-sequence sub-dirs created here')
    p.add_argument('--object_library_folder', required=True)

    seq_grp = p.add_mutually_exclusive_group(required=True)
    seq_grp.add_argument('--sequence_name', default=None,
                         help='Single sequence name, e.g. P0003_02')
    seq_grp.add_argument('--all_sequences', action='store_true',
                         help='Run all sequences found under --gt_root')

    p.add_argument('--sequence_folder', default=None,
                   help='Path to single HOT3D VRS sequence folder '
                        '(required when --sequence_name is used)')
    p.add_argument('--sequence_root', default=None,
                   help='Root containing all sequence folders '
                        '(required when --all_sequences is used)')
    p.add_argument('--skip_inference', action='store_true',
                   help='Skip steps 1-5 (VRS extract, detect, SLAM, HAWOR); '
                        'requires world_space_res.pth + vrs_meta.npz to exist')
    p.add_argument('--cleanup_extracted_images', action='store_true',
                   help='Delete per-sequence extracted_images/ after metrics are computed')
    p.add_argument('--cpu', action='store_true', help='Force CPU (slow)')
    p.add_argument('--gpu_ids', default='0',
                   help='Comma-separated visible GPU IDs for all_sequences mode, e.g. "0,1,2,3"')
    p.add_argument('--num_parallel', type=int, default=1,
                   help='Max number of sequences to run in parallel for all_sequences mode')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.num_parallel < 1:
        raise ValueError('--num_parallel must be >= 1')

    if args.sequence_name:
        if not args.sequence_folder:
            raise ValueError('--sequence_folder is required with --sequence_name')
        metrics = run_sequence(args, args.sequence_name, args.sequence_folder)
        print('\n[RESULT]', json.dumps(metrics, indent=2))
        with open(os.path.join(args.out_dir, f'{args.sequence_name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        return

    # all_sequences
    if not args.sequence_root:
        raise ValueError('--sequence_root is required with --all_sequences')

    sequence_names = sorted(
        d for d in os.listdir(args.gt_root)
        if os.path.isdir(os.path.join(args.gt_root, d)))

    all_rows = []
    valid_tasks = []
    for seq_name in sequence_names:
        seq_folder = os.path.join(args.sequence_root, seq_name)
        if not os.path.isdir(seq_folder):
            print(f'[SKIP] {seq_name}: folder not found at {seq_folder}')
            continue
        valid_tasks.append((seq_name, seq_folder))

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    per_seq_out_dir = os.path.join(args.out_dir, 'per_sequence')
    os.makedirs(per_seq_out_dir, exist_ok=True)

    if args.num_parallel == 1:
        for seq_name, seq_folder in valid_tasks:
            try:
                m = run_sequence(args, seq_name, seq_folder)
                all_rows.append(m)
                print(f'[OK] {seq_name}: '
                      f'PA={m["pa_mpjpe_mm"]:.1f}  '
                      f'W={m["w_mpjpe_mm"]:.1f}  '
                      f'WA={m["wa_mpjpe_mm"]:.1f}  mm  '
                      f'ATE={m["ate_m"]*100:.1f}  '
                      f'ATE-S={m["ate_s_m"]*100:.1f}  cm  '
                      f'RTE={m["rte_pct"]:.2f}%  '
                      f'Accel={m["accel_ms2"]:.3f} m/s²')
            except Exception as e:
                print(f'[ERROR] {seq_name}: {e}')
    else:
        n_workers = min(args.num_parallel, len(valid_tasks))
        print(f'[all_sequences] Parallel mode: workers={n_workers}, gpu_ids={gpu_ids}')
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for idx, (seq_name, seq_folder) in enumerate(valid_tasks):
                gpu_id = gpu_ids[idx % len(gpu_ids)]
                futures.append(
                    ex.submit(_run_sequence_subprocess, args, seq_name, seq_folder, gpu_id, per_seq_out_dir)
                )

            for fut in as_completed(futures):
                seq_name, m, err = fut.result()
                if err is not None:
                    print(f'[ERROR] {seq_name}: {err}')
                    continue
                all_rows.append(m)
                print(f'[OK] {seq_name}: '
                      f'PA={m["pa_mpjpe_mm"]:.1f}  '
                      f'W={m["w_mpjpe_mm"]:.1f}  '
                      f'WA={m["wa_mpjpe_mm"]:.1f}  mm  '
                      f'ATE={m["ate_m"]*100:.1f}  '
                      f'ATE-S={m["ate_s_m"]*100:.1f}  cm  '
                      f'RTE={m["rte_pct"]:.2f}%  '
                      f'Accel={m["accel_ms2"]:.3f} m/s²')

    if not all_rows:
        raise RuntimeError('No sequence metrics collected; all runs failed or were skipped.')
    overall = aggregate(all_rows)
    print('\n[OVERALL]', json.dumps(overall, indent=2))
    with open(os.path.join(args.out_dir, 'all_sequences_metrics.json'), 'w') as f:
        json.dump({'overall': overall, 'per_sequence': all_rows}, f, indent=2)


if __name__ == '__main__':
    main()
