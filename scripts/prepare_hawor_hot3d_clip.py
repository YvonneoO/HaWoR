#!/usr/bin/env python3
"""
Export HOT3D-Clip frames for HaWoR (demo.py / detect_track_video layout).

HaWoR expects a layout identical to ffmpeg-extracted video frames:

  <seq_root>/extracted_images/000000.jpg, 000001.jpg, ...
  <seq_root>/est_focal.txt          # single float fx (pixels)

demo.py resolves paths from --video_path as:

  seq_root = dirname(video_path) / basename(video_path, ext)
  images   = seq_root / extracted_images / *.jpg

So if you set:
  --hawor-seq-root /data/hawor_runs
  --seq-name clip-001849

this script writes:
  /data/hawor_runs/clip-001849/extracted_images/*.jpg
  /data/hawor_runs/clip-001849/est_focal.txt

and you should run HaWoR with:
  --video_path /data/hawor_runs/clip-001849.mp4
  --img_focal $(cat /data/hawor_runs/clip-001849/est_focal.txt)

Undistortion matches hot3d/clips/vis_clips.py: clip_util.convert_to_pinhole_camera
+ hand_tracking_toolkit.dataset.warp_image.

Optional: --max-frames limits export length for quick tests.

After export:
  cd HaWoR && python demo.py --video_path ... --img_focal ... --vis_mode cam
Use vis_mode=cam for image-plane projection (run_vis2_on_video_cam).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional, Tuple


def _ensure_clip_util(hot3d_module_path: Optional[str]) -> None:
    if hot3d_module_path:
        clips_pkg = os.path.join(hot3d_module_path, "clips")
        for p in (hot3d_module_path, clips_pkg):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)


def _list_clip_frame_ids(clip_dir: str) -> List[str]:
    suffix = ".cameras.json"
    frame_ids: List[str] = []
    for name in os.listdir(clip_dir):
        if not name.endswith(suffix):
            continue
        fid = name[: -len(suffix)]
        if not fid.isdigit():
            continue
        frame_ids.append(fid)
    frame_ids.sort(key=lambda x: int(x))
    return frame_ids


def _load_pinhole_rgb_and_fx(
    clip_dir: str,
    stream_id: str,
    frame_id: str,
) -> Tuple[Any, float]:
    import imageio.v2 as imageio
    import cv2

    try:
        from hand_tracking_toolkit.dataset import warp_image
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "hand_tracking_toolkit is required for fisheye→pinhole warp (same as HOT3D vis_clips). "
            "Install: pip install 'git+https://github.com/facebookresearch/hand_tracking_toolkit.git' "
            "or set HAND_TRACKING_TOOLKIT_PATH to a clone of that repo."
        ) from e

    import clip_util  # noqa: E402  — after _ensure_clip_util

    img_path = os.path.join(clip_dir, f"{frame_id}.image_{stream_id}.jpg")
    cam_path = os.path.join(clip_dir, f"{frame_id}.cameras.json")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not os.path.isfile(cam_path):
        raise FileNotFoundError(f"Missing cameras json: {cam_path}")

    rgb = imageio.imread(img_path)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    elif rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    with open(cam_path, "r") as f:
        cameras_raw = json.load(f)
    if stream_id not in cameras_raw:
        raise KeyError(f"stream {stream_id} not in {cam_path}")

    cam_model = clip_util.camera.from_json(cameras_raw[stream_id])
    cam_pin = clip_util.convert_to_pinhole_camera(cam_model)
    rgb_u = warp_image(src_camera=cam_model, dst_camera=cam_pin, src_image=rgb)
    fx = float(cam_pin.f[0])
    return rgb_u, fx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-dir", required=True, help="Extracted clip folder (per-frame jpg + cameras.json).")
    parser.add_argument("--stream-id", default="214-1", help="Aria RGB stream key in cameras.json.")
    parser.add_argument(
        "--hot3d-module-path",
        default=None,
        help="Path to hot3d repo root (contains clips/clip_util.py).",
    )
    parser.add_argument(
        "--hawor-seq-root",
        required=True,
        help="Parent directory for HaWoR sequence folder (same parent as the dummy .mp4 path).",
    )
    parser.add_argument(
        "--seq-name",
        required=True,
        help="Sequence folder name under hawor-seq-root (e.g. clip-001849).",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite jpg/focal/map even if extracted_images already exists.",
    )
    args = parser.parse_args()

    _ensure_clip_util(args.hot3d_module_path)

    clip_dir = os.path.abspath(args.clip_dir)
    seq_root = os.path.join(os.path.abspath(args.hawor_seq_root), args.seq_name)
    img_out = os.path.join(seq_root, "extracted_images")
    os.makedirs(img_out, exist_ok=True)

    frame_ids = _list_clip_frame_ids(clip_dir)
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise RuntimeError(f"No *.cameras.json under {clip_dir}")

    import cv2

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(it, **_kwargs):
            return it

    n = len(frame_ids)
    print(f"Undistorting {n} frames (fisheye→pinhole); full Aria res can take several minutes.", flush=True)

    map_records = []
    focal_ref: Optional[float] = None

    for i, fid in enumerate(tqdm(frame_ids, desc="prepare/warp", unit="frame")):
        out_jpg = os.path.join(img_out, f"{i:06d}.jpg")
        if os.path.isfile(out_jpg) and not args.overwrite:
            # Still read focal from first frame if missing est_focal
            if focal_ref is None:
                _, fx = _load_pinhole_rgb_and_fx(clip_dir, args.stream_id, fid)
                focal_ref = fx
            map_records.append({"seq_index": i, "clip_frame_id": fid})
            continue

        rgb_u, fx = _load_pinhole_rgb_and_fx(clip_dir, args.stream_id, fid)
        if focal_ref is None:
            focal_ref = fx
        bgr = cv2.cvtColor(rgb_u, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_jpg, bgr)
        map_records.append({"seq_index": i, "clip_frame_id": fid})

    assert focal_ref is not None
    est_path = os.path.join(seq_root, "est_focal.txt")
    with open(est_path, "w") as f:
        f.write(str(focal_ref))

    map_path = os.path.join(seq_root, "hawor_clip_frame_map.json")
    with open(map_path, "w") as f:
        json.dump(
            {
                "clip_dir": clip_dir,
                "stream_id": args.stream_id,
                "n_frames": len(map_records),
                "frames": map_records,
            },
            f,
            indent=2,
        )

    dummy_mp4 = os.path.join(os.path.abspath(args.hawor_seq_root), f"{args.seq_name}.mp4")
    print("Done.")
    print(f"  seq_root:     {seq_root}")
    print(f"  images:       {img_out}  ({len(map_records)} files)")
    print(f"  est_focal:    {est_path}  -> {focal_ref}")
    print(f"  frame map:    {map_path}")
    print()
    print("Run HaWoR demo (from HaWoR repo root), image-plane render:")
    print(
        f"  python demo.py \\\n"
        f"    --video_path {dummy_mp4} \\\n"
        f"    --img_focal {focal_ref} \\\n"
        f"    --checkpoint ./weights/hawor/checkpoints/hawor.ckpt \\\n"
        f"    --infiller_weight ./weights/hawor/checkpoints/infiller.pt \\\n"
        f"    --vis_mode cam"
    )
    print()
    print("Note: dummy_mp4 can be any file; detect_track skips ffmpeg if extracted_images/ exists.")


if __name__ == "__main__":
    main()
