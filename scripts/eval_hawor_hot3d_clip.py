#!/usr/bin/env python3
"""
HaWoR on HOT3D-Clips (untarred folder): prepare pinhole frames → detect/track → SLAM →
motion → infiller, then align SLAM→GT world using per-frame camera centers from clip GT
npz, compute the same metrics as ``eval_hawor_hot3d.py``, and optionally export a camera-view mp4.

Run from HaWoR repo root (so ``./weights/...`` and imports resolve), e.g.:

  python scripts/eval_hawor_hot3d_clip.py \\
    --clip-dir example/clip/clip-001881 \\
    --gt-seq-dir /data/hot3d-clip/clip_gt/train/clip-001881 \\
    --hawor-work-root example/hawor_work \\
    --seq-name clip-001881 \\
    --hot3d-module-path /workspace/hot3d-private/hot3d \\
    --checkpoint ./weights/hawor/checkpoints/hawor.ckpt \\
    --infiller-weight ./weights/hawor/checkpoints/infiller.pt \\
    --out-dir example/hawor_out/clip-001881 \\
    --export-video

Requires: same deps as ``demo.py`` / ``eval_hawor_hot3d.py`` (YOLO detector, DROID-SLAM, etc.).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import types
from glob import glob
from typing import Optional

import cv2
import joblib
import numpy as np
import torch
from natsort import natsorted


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_eval_hot3d_module():
    """Load eval_hawor_hot3d.py without requiring ``scripts`` to be a package."""
    path = os.path.join(_repo_root(), "scripts", "eval_hawor_hot3d.py")
    spec = importlib.util.spec_from_file_location("eval_hawor_hot3d", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_vrs_meta_from_prepare(seq_folder: str) -> None:
    """Write ``vrs_meta.npz`` (same keys as VRS eval) from ``hawor_clip_frame_map.json`` + est_focal."""
    map_path = os.path.join(seq_folder, "hawor_clip_frame_map.json")
    focal_path = os.path.join(seq_folder, "est_focal.txt")
    if not os.path.isfile(map_path):
        raise FileNotFoundError(map_path)
    if not os.path.isfile(focal_path):
        raise FileNotFoundError(focal_path)
    with open(map_path) as f:
        m = json.load(f)
    frames = m.get("frames") or []
    if not frames:
        raise RuntimeError(f"Empty frames in {map_path}")
    ts = np.array([int(x["clip_frame_id"]) for x in frames], dtype=np.int64)
    focal = float(open(focal_path).read().strip())
    img0 = cv2.imread(os.path.join(seq_folder, "extracted_images", "000000.jpg"))
    if img0 is None:
        raise FileNotFoundError(os.path.join(seq_folder, "extracted_images", "000000.jpg"))
    h, w = img0.shape[:2]
    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    T_device_cam = np.eye(4, dtype=np.float64)
    meta_path = os.path.join(seq_folder, "vrs_meta.npz")
    np.savez(
        meta_path,
        timestamps_ns=ts,
        focal=np.float64(focal),
        img_center=img_center,
        T_device_cam=T_device_cam,
    )


def _get_gt_cam_centers_from_clip_npz(
    gt_seq_dir: str,
    timestamps_ns: np.ndarray,
    frame_indices: np.ndarray,
    load_gt_frame_map,
):
    gt_map = load_gt_frame_map(gt_seq_dir)
    positions = []
    for fi in np.asarray(frame_indices).astype(int).reshape(-1):
        if fi < 0 or fi >= len(timestamps_ns):
            positions.append(None)
            continue
        ts = int(timestamps_ns[fi])
        p = gt_map.get(ts)
        if p is None or not os.path.isfile(p):
            positions.append(None)
            continue
        with np.load(p, allow_pickle=True) as z:
            if "T_world_cam" not in z.files:
                positions.append(None)
                continue
            twc = z["T_world_cam"].astype(np.float64)
            if twc.ndim == 3:
                twc = twc[0]
            positions.append(twc[:3, 3].copy())
    return positions


def _run_prepare(
    clip_dir: str,
    hawor_work_root: str,
    seq_name: str,
    stream_id: str,
    hot3d_module_path: Optional[str],
    overwrite: bool,
):
    prep = os.path.join(_repo_root(), "scripts", "prepare_hawor_hot3d_clip.py")
    cmd = [
        sys.executable,
        prep,
        "--clip-dir",
        os.path.abspath(clip_dir),
        "--hawor-seq-root",
        os.path.abspath(hawor_work_root),
        "--seq-name",
        seq_name,
        "--stream-id",
        stream_id,
    ]
    if overwrite:
        cmd.append("--overwrite")
    if hot3d_module_path:
        cmd += ["--hot3d-module-path", os.path.abspath(hot3d_module_path)]
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip-dir", required=True, help="Untarred HOT3D-clip folder (fisheye jpg + cameras.json).")
    p.add_argument("--gt-seq-dir", required=True, help="clip_gt sequence dir with manifest / frame npz (same clip id).")
    p.add_argument("--hawor-work-root", required=True, help="Parent dir for HaWoR seq folder + dummy .mp4 (see prepare script).")
    p.add_argument("--seq-name", required=True, help="Folder name under hawor-work-root, e.g. clip-001881.")
    p.add_argument("--stream-id", default="214-1")
    p.add_argument("--hot3d-module-path", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--infiller-weight", required=True)
    p.add_argument("--out-dir", required=True, help="Writes metrics JSON and optional vis mp4 here.")
    p.add_argument(
        "--overwrite-prepare",
        action="store_true",
        help="Re-export jpg from clip (passed through to prepare_hawor_hot3d_clip.py).",
    )
    p.add_argument("--skip-prepare", action="store_true", help="Assume extracted_images + hawor_clip_frame_map already exist.")
    p.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip detect/SLAM/HaWoR; requires world_space_res.pth + SLAM npz + vrs_meta.npz under seq folder.",
    )
    p.add_argument("--export-video", action="store_true", help="Export camera-view mp4 via ARCTICViewer (headless).")
    p.add_argument(
        "--vis-interactive",
        action="store_true",
        help="If set with --export-video, open interactive viewer instead of writing mp4.",
    )
    p.add_argument("--thresh", type=float, default=0.2, help="Hand detector threshold for detect_track.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.insert(0, _repo_root())

    ev = _load_eval_hot3d_module()
    load_gt_frame_map = ev.load_gt_frame_map
    compute_metrics = ev.compute_metrics
    align_se3_traj = ev.align_se3_traj
    HOT3D_TO_MANO = ev.HOT3D_TO_MANO

    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
    from lib.eval_utils.custom_utils import load_slam_cam
    from lib.pipeline.tools import detect_track
    from lib.vis.run_vis2 import run_vis2_on_video_cam

    seq_folder = os.path.join(os.path.abspath(args.hawor_work_root), args.seq_name)
    fake_video_path = os.path.join(os.path.abspath(args.hawor_work_root), f"{args.seq_name}.mp4")
    img_folder = os.path.join(seq_folder, "extracted_images")

    if not args.skip_prepare:
        os.makedirs(os.path.abspath(args.hawor_work_root), exist_ok=True)
        _run_prepare(
            args.clip_dir,
            args.hawor_work_root,
            args.seq_name,
            args.stream_id,
            args.hot3d_module_path,
            overwrite=args.overwrite_prepare,
        )
    _write_vrs_meta_from_prepare(seq_folder)

    with open(os.path.join(seq_folder, "est_focal.txt")) as f:
        focal = float(f.read().strip())

    meta = np.load(os.path.join(seq_folder, "vrs_meta.npz"))
    timestamps_ns = meta["timestamps_ns"].astype(np.int64)

    imgfiles = natsorted(glob(os.path.join(img_folder, "*.jpg")))
    n_frames = len(imgfiles)
    if n_frames == 0:
        raise RuntimeError(f"No jpg under {img_folder}")
    if len(timestamps_ns) != n_frames:
        raise RuntimeError(f"timestamps_ns ({len(timestamps_ns)}) != n_frames ({n_frames})")

    world_res_path = os.path.join(seq_folder, "world_space_res.pth")
    if os.path.isfile(world_res_path):
        pred_trans_probe, *_ = joblib.load(world_res_path)
        t_m = int(pred_trans_probe.shape[1])
        if t_m != n_frames:
            raise RuntimeError(
                f"Cached world_space_res.pth has T={t_m} but extracted_images has {n_frames} frames; "
                "delete HaWoR outputs under the seq folder or re-run with --overwrite-prepare and without --skip-inference."
            )

    start_idx, end_idx = 0, n_frames
    if not os.path.isfile(fake_video_path):
        open(fake_video_path, "a").close()

    if not args.skip_inference:
        track_dir = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}")
        os.makedirs(track_dir, exist_ok=True)
        tracks_np = os.path.join(track_dir, "model_tracks.npy")
        if not os.path.isfile(tracks_np):
            boxes_, tracks_ = detect_track(imgfiles, thresh=args.thresh)
            np.save(os.path.join(track_dir, "model_boxes.npy"), boxes_)
            np.save(os.path.join(track_dir, "model_tracks.npy"), tracks_)

        from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
        from scripts.scripts_test_video.hawor_slam import hawor_slam

        hw_args = types.SimpleNamespace(
            video_path=fake_video_path,
            img_focal=focal,
            checkpoint=args.checkpoint,
            infiller_weight=args.infiller_weight,
        )
        frame_chunks_all, _ = hawor_motion_estimation(hw_args, start_idx, end_idx, seq_folder)
        slam_npz = os.path.join(seq_folder, "SLAM", f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.isfile(slam_npz):
            hawor_slam(hw_args, start_idx, end_idx)
        hawor_infiller(hw_args, start_idx, end_idx, frame_chunks_all)

    slam_npz = os.path.join(seq_folder, "SLAM", f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.isfile(slam_npz):
        raise FileNotFoundError(slam_npz)

    slam_data = dict(np.load(slam_npz, allow_pickle=True))
    traj = slam_data["traj"]
    scale = float(slam_data["scale"])
    tstamp = slam_data["tstamp"].astype(int)
    t_slam_kf = traj[tstamp, :3] * scale

    gt_seq_dir = os.path.abspath(args.gt_seq_dir)
    if not os.path.isdir(gt_seq_dir):
        raise FileNotFoundError(gt_seq_dir)

    gt_positions_list = _get_gt_cam_centers_from_clip_npz(
        gt_seq_dir, timestamps_ns, tstamp, load_gt_frame_map
    )
    valid_kf = [i for i, p in enumerate(gt_positions_list) if p is not None]
    if len(valid_kf) < 4:
        raise RuntimeError(f"Too few SLAM keyframes with GT camera pose: {len(valid_kf)}")

    t_slam_valid = t_slam_kf[valid_kf]
    t_gt_valid = np.stack([gt_positions_list[i] for i in valid_kf])
    R_align, t_align = align_se3_traj(t_slam_valid, t_gt_valid)
    print(
        f"[{args.seq_name}] SE3 alignment residual (cam centers): "
        f"{np.linalg.norm(t_gt_valid - (R_align @ t_slam_valid.T).T - t_align, axis=1).mean() * 100:.1f} cm"
    )

    if not os.path.isfile(world_res_path):
        raise FileNotFoundError(world_res_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res_path)
    pred_trans = pred_trans.float()
    pred_rot = pred_rot.float()
    pred_hand_pose = pred_hand_pose.float()
    pred_betas = pred_betas.float()
    if pred_hand_pose.ndim == 3 and pred_hand_pose.shape[-1] == 45:
        pred_hand_pose = pred_hand_pose.view(*pred_hand_pose.shape[:2], 15, 3)

    use_cuda = torch.cuda.is_available()
    with torch.no_grad():
        left_out = run_mano_left(
            pred_trans[0:1],
            pred_rot[0:1],
            pred_hand_pose[0:1],
            betas=pred_betas[0:1],
            use_cuda=use_cuda,
        )
        right_out = run_mano(
            pred_trans[1:2],
            pred_rot[1:2],
            pred_hand_pose[1:2],
            betas=pred_betas[1:2],
            use_cuda=use_cuda,
        )

    left_j = left_out["joints"][0, :, HOT3D_TO_MANO, :].cpu().numpy()
    right_j = right_out["joints"][0, :, HOT3D_TO_MANO, :].cpu().numpy()
    left_v = left_out["vertices"][0].cpu().numpy()
    right_v = right_out["vertices"][0].cpu().numpy()
    joints_slam = np.stack([left_j, right_j], axis=1)
    verts_slam = np.stack([left_v, right_v], axis=1)

    R_f = R_align.astype(np.float32)
    t_f = t_align.astype(np.float32)
    joints_aligned = (R_f @ joints_slam.reshape(-1, 3).T).T.reshape(n_frames, 2, 20, 3) + t_f
    verts_aligned = (R_f @ verts_slam.reshape(-1, 3).T).T.reshape(n_frames, 2, 778, 3) + t_f

    if torch.is_tensor(pred_valid):
        pv = pred_valid.detach().cpu().float().numpy()
    else:
        pv = np.asarray(pred_valid, dtype=np.float32)
    if pv.ndim != 2 or 2 not in pv.shape:
        raise ValueError(f"pred_valid expected shape (*, 2) or (2, *), got {pv.shape}")
    pred_valid_np = (pv > 0.5).T if pv.shape[0] == 2 else (pv > 0.5)

    gt_frame_map = load_gt_frame_map(gt_seq_dir)
    if not gt_frame_map:
        raise RuntimeError(f"No GT npz under {gt_seq_dir}")

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
        raise RuntimeError("No timestamp_ns overlap between HaWoR frames and GT npz")

    if len(matched_pred_idx) != n_frames:
        print(
            f"[WARN] matched {len(matched_pred_idx)}/{n_frames} frames by timestamp_ns; "
            "metrics on matched subset only."
        )

    matched_pred_idx = np.asarray(matched_pred_idx, dtype=np.int64)
    metrics = compute_metrics(
        joints_aligned[matched_pred_idx],
        verts_aligned[matched_pred_idx],
        pred_valid_np[matched_pred_idx],
        matched_gt_paths,
        timestamps_ns=pred_ts[matched_pred_idx],
    )
    metrics["sequence"] = args.seq_name
    metrics["n_hawor_frames"] = int(n_frames)
    metrics["n_matched_gt_frames"] = int(len(matched_pred_idx))

    out_json = os.path.join(args.out_dir, f"{args.seq_name}_hawor_clip_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_json}")

    if args.export_video:
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_npz)
        faces = get_mano_faces()
        faces_new = np.array(
            [
                [92, 38, 234],
                [234, 38, 239],
                [38, 122, 239],
                [239, 122, 279],
                [122, 118, 279],
                [279, 118, 215],
                [118, 117, 215],
                [215, 117, 214],
                [117, 119, 214],
                [214, 119, 121],
                [119, 120, 121],
                [121, 120, 78],
                [120, 108, 78],
                [78, 108, 79],
            ]
        )
        faces_right = np.concatenate([faces, faces_new], axis=0)
        faces_left = faces_right[:, [0, 2, 1]]
        T_vis = min(n_frames, pred_trans.shape[1], len(R_w2c_sla_all))
        vis_start, vis_end = 0, T_vis
        pred_glob_r = run_mano(
            pred_trans[1:2, vis_start:vis_end],
            pred_rot[1:2, vis_start:vis_end],
            pred_hand_pose[1:2, vis_start:vis_end],
            betas=pred_betas[1:2, vis_start:vis_end],
            use_cuda=use_cuda,
        )
        pred_glob_l = run_mano_left(
            pred_trans[0:1, vis_start:vis_end],
            pred_rot[0:1, vis_start:vis_end],
            pred_hand_pose[0:1, vis_start:vis_end],
            betas=pred_betas[0:1, vis_start:vis_end],
            use_cuda=use_cuda,
        )
        right_dict = {"vertices": pred_glob_r["vertices"][0].unsqueeze(0), "faces": faces_right}
        left_dict = {"vertices": pred_glob_l["vertices"][0].unsqueeze(0), "faces": faces_left}
        R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
        R_c2w = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all[vis_start:vis_end])
        t_c2w = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all[vis_start:vis_end])
        R_w2c = R_c2w.transpose(-1, -2)
        t_w2c = -torch.einsum("bij,bj->bi", R_w2c, t_c2w)
        left_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, left_dict["vertices"].cpu())
        right_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, right_dict["vertices"].cpu())

        output_pth = os.path.join(args.out_dir, f"vis_cam_{vis_start}_{vis_end}")
        os.makedirs(output_pth, exist_ok=True)
        image_names = imgfiles[vis_start:vis_end]
        vid = run_vis2_on_video_cam(
            left_dict,
            right_dict,
            output_pth,
            focal,
            image_names,
            R_w2c=R_w2c,
            t_w2c=t_w2c,
            interactive=bool(args.vis_interactive),
        )
        if vid:
            dst_mp4 = os.path.join(args.out_dir, f"{args.seq_name}_hawor_cam.mp4")
            if os.path.isfile(dst_mp4):
                os.remove(dst_mp4)
            os.replace(vid, dst_mp4)
            print(f"Wrote {dst_mp4}")
        else:
            print("Interactive vis finished (no mp4 path returned).")


if __name__ == "__main__":
    main()
