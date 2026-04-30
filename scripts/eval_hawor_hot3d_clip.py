#!/usr/bin/env python3
"""
HOT3D clip: prepare → HaWoR → metrics (``eval_hawor_hot3d``-aligned). Optional ``--export-video``
(mesh_pyrender + ffmpeg or ``--export-video-renderer cv2``). Repo root; prepare needs
``hand_tracking_toolkit`` (see HOT3D vis_clips). Optional ``HAWOR_DROID_BUFFER`` for VRAM.
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


def _write_pinhole_frames_mp4(image_paths: list[str], out_mp4: str, fps: float = 30.0) -> None:
    """Write one mp4 from undistorted pinhole jpgs using OpenCV only (no OpenGL / aitviewer)."""
    if not image_paths:
        raise ValueError("no image paths for mp4 export")
    first = cv2.imread(image_paths[0])
    if first is None:
        raise RuntimeError(f"cv2.imread failed: {image_paths[0]}")
    h, w = first.shape[:2]
    out_mp4 = os.path.abspath(out_mp4)
    parent = os.path.dirname(out_mp4)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV VideoWriter could not open {out_mp4} (codec mp4v). "
            "Check OpenCV was built with ffmpeg/GStreamer, or re-encode frames with ffmpeg."
        )
    writer.write(first)
    for p in image_paths[1:]:
        im = cv2.imread(p)
        if im is None:
            raise RuntimeError(f"cv2.imread failed: {p}")
        if im.shape[0] != h or im.shape[1] != w:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(im)
    writer.release()


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
    max_frames: Optional[int] = None,
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
    if max_frames is not None:
        cmd += ["--max-frames", str(int(max_frames))]
    env = os.environ.copy()
    htt = os.environ.get("HAND_TRACKING_TOOLKIT_PATH", "").strip()
    if htt:
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = htt if not prev else f"{htt}{os.pathsep}{prev}"
    subprocess.run(cmd, check=True, env=env)


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
    p.add_argument(
        "--export-video",
        action="store_true",
        help="Write *_hawor_cam.mp4 (see --export-video-renderer).",
    )
    p.add_argument(
        "--export-video-renderer",
        choices=("mesh_pyrender", "cv2"),
        default="mesh_pyrender",
        help="mesh_pyrender: undistort RGB + pyrender MANO, then ffmpeg. cv2: jpg→mp4 only.",
    )
    p.add_argument(
        "--export-video-fps",
        type=float,
        default=30.0,
        help="Output video fps (mesh path uses ffmpeg; cv2 path uses OpenCV writer).",
    )
    p.add_argument(
        "--export-video-mesh-alpha",
        type=float,
        default=0.75,
        help="Alpha blend for mesh overlay when --export-video-renderer is mesh_pyrender.",
    )
    p.add_argument(
        "--export-video-keep-frames",
        action="store_true",
        help="Keep temporary PNG sequence when using mesh_pyrender (for debugging).",
    )
    p.add_argument("--thresh", type=float, default=0.2, help="Hand detector threshold for detect_track.")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap frames for prepare + downstream HaWoR (smoke test; default = full clip).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.insert(0, _repo_root())

    # eval_hawor_hot3d.py imports HOT3D from HOT3D_REPO_DIR (see its module top); align with --hot3d-module-path.
    _hot3d_pkg = (
        args.hot3d_module_path
        or os.environ.get("HOT3D_MODULE_PATH")
        or os.environ.get("HOT3D_REPO_DIR")
    )
    if _hot3d_pkg:
        os.environ["HOT3D_REPO_DIR"] = os.path.abspath(_hot3d_pkg)

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
            max_frames=args.max_frames,
        )
    _write_vrs_meta_from_prepare(seq_folder)

    # Defer importing eval_hawor_hot3d (pulls YOLO/ultralytics) until after clip prepare so a missing
    # hand_tracking_toolkit fails fast on the prepare step with a clear stack trace.
    ev = _load_eval_hot3d_module()
    load_gt_frame_map = ev.load_gt_frame_map
    compute_metrics = ev.compute_metrics
    align_se3_traj = ev.align_se3_traj
    HOT3D_TO_MANO = ev.HOT3D_TO_MANO
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left

    # Import heavy deps here (after prepare, before detect/SLAM) so failures are not surprises
    # after a long GPU run. Skip paths avoid unused imports (e.g. no ultralytics if --skip-inference).
    if not args.skip_inference:
        from lib.pipeline.tools import detect_track

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
        T_vis = min(int(n_frames), int(pred_trans.shape[1]))
        vis_start, vis_end = 0, T_vis
        image_names = imgfiles[vis_start:vis_end]

        if args.export_video_renderer == "cv2":
            dst_mp4 = os.path.join(args.out_dir, f"{args.seq_name}_hawor_cam.mp4")
            _write_pinhole_frames_mp4(image_names, dst_mp4, fps=float(args.export_video_fps))
            print(f"Wrote {dst_mp4} (pinhole frames, OpenCV mux; no mesh overlay)")
        else:
            from lib.eval_utils.custom_utils import load_slam_cam
            from lib.vis.hawor_mesh_overlay import export_mesh_overlay_video_ffmpeg

            _, _, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_npz)
            n_cam = int(R_c2w_sla_all.shape[0])
            T_vis = min(T_vis, n_cam)
            vis_start, vis_end = 0, T_vis
            image_names = imgfiles[vis_start:vis_end]

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

            R_np = R_c2w_sla_all[vis_start:vis_end].detach().cpu().numpy().astype(np.float32)
            t_np = t_c2w_sla_all[vis_start:vis_end].detach().cpu().numpy().astype(np.float32)
            verts_slice = verts_aligned[vis_start:vis_end]
            pred_slice = pred_valid_np[vis_start:vis_end].astype(bool)

            dst_mp4 = os.path.join(args.out_dir, f"{args.seq_name}_hawor_cam.mp4")
            work_dir = os.path.join(args.out_dir, f"_mesh_frames_{args.seq_name}")
            export_mesh_overlay_video_ffmpeg(
                image_names,
                verts_slice,
                pred_slice,
                R_np,
                t_np,
                R_align,
                t_align,
                float(focal),
                faces_right,
                dst_mp4,
                fps=float(args.export_video_fps),
                mesh_alpha=float(args.export_video_mesh_alpha),
                work_dir=work_dir,
                keep_frames=bool(args.export_video_keep_frames),
            )
            print(f"Wrote {dst_mp4} (pinhole RGB + pyrender MANO, ffmpeg libx264)")


if __name__ == "__main__":
    main()
