#!/usr/bin/env python3
"""
Export frames for HaWoR (same layout as ffmpeg-extracted video frames).

Two modes (pick one):

1) HOT3D clip (default): ``--clip-dir`` holds **distorted raw** sensor images (e.g. Aria fisheye
   ``*.image_<stream>.jpg``) plus per-frame ``*.cameras.json`` (device intrinsics / distortion).
   This mode **warps** those frames to a pinhole view using hot3d/clips/vis_clips.py
   (``clip_util`` + ``hand_tracking_toolkit``).

2) Pinhole folder: ``--pinhole-image-dir`` + ``--cam-k`` (JSON 3×3 K). Copies already
   **undistorted / rectified** images from a flat directory; writes ``cam_K.json`` and integer frame IDs
   in ``hawor_clip_frame_map.json`` (for ``vrs_meta`` without HOT3D timestamps).

Output (both modes):

  <hawor-seq-root>/<seq-name>/extracted_images/000000.jpg, ...
  <seq-name>/est_focal.txt
  <seq-name>/hawor_clip_frame_map.json

Optional: ``--max-frames``. Dummy ``<hawor-seq-root>/<seq-name>.mp4`` is touched in pinhole mode.

After export:
  cd HaWoR && python demo.py --video_path ... --img_focal ... --vis_mode cam
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from natsort import natsorted


def _ensure_clip_util(hot3d_module_path: Optional[str]) -> None:
    if hot3d_module_path:
        clips_pkg = os.path.join(hot3d_module_path, "clips")
        for p in (hot3d_module_path, clips_pkg):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)


def _hint_if_clip_parent_not_leaf(clip_dir: str) -> str:
    """If ``clip_dir`` has no *.cameras.json but subfolders do, suggest a concrete --clip-dir."""
    if not os.path.isdir(clip_dir):
        return ""
    hints: List[str] = []
    try:
        for name in sorted(os.listdir(clip_dir)):
            sub = os.path.join(clip_dir, name)
            if not os.path.isdir(sub):
                continue
            try:
                if any(fn.endswith(".cameras.json") for fn in os.listdir(sub)):
                    hints.append(name)
            except OSError:
                continue
    except OSError:
        return ""
    if not hints:
        return ""
    ex = os.path.join(clip_dir, hints[0])
    tail = ""
    if len(hints) > 5:
        tail = f", … ({len(hints)} clip-like subdirs total)"
    else:
        tail = ", ".join(hints)
    return (
        f"\n  Hint: {clip_dir!r} has no *.cameras.json in its root — it may be a split root (e.g. train_aria). "
        f"Use one sequence folder, e.g. --clip-dir {ex!r}\n"
        f"  Clip-like subdirs here: {tail}"
    )


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


def _list_pinhole_images(image_dir: str, extensions: Sequence[str]) -> List[str]:
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(image_dir)
    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions)
    paths: List[str] = []
    for name in os.listdir(image_dir):
        low = name.lower()
        if not any(low.endswith(e) for e in exts):
            continue
        p = os.path.join(image_dir, name)
        if os.path.isfile(p):
            paths.append(p)
    return natsorted(paths)


def _load_pinhole_rgb_and_fx(
    clip_dir: str,
    stream_id: str,
    frame_id: str,
) -> Tuple[Any, float]:
    import imageio.v2 as imageio

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


def _export_pinhole_flat_dir(
    image_dir: str,
    seq_root: str,
    cam_k_json: str,
    extensions: str,
    max_frames: Optional[int],
    overwrite: bool,
) -> tuple[float, int]:
    K = np.asarray(json.loads(cam_k_json), dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"cam-k must be 3×3, got {K.shape}")

    img_out = os.path.join(seq_root, "extracted_images")
    os.makedirs(img_out, exist_ok=True)

    exts = [x.strip() for x in extensions.split(",") if x.strip()]
    imgs = _list_pinhole_images(image_dir, exts)
    if max_frames is not None:
        imgs = imgs[: int(max_frames)]
    if not imgs:
        raise RuntimeError(f"No images under {image_dir} (extensions={exts!r})")

    focal_ref = float(K[0, 0])
    map_records: List[dict] = []

    for i, src in enumerate(imgs):
        dst = os.path.join(img_out, f"{i:06d}.jpg")
        if os.path.isfile(dst) and not overwrite:
            map_records.append({"seq_index": i, "clip_frame_id": int(i)})
            continue
        im = cv2.imread(src, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"cv2.imread failed: {src}")
        cv2.imwrite(dst, im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        map_records.append({"seq_index": i, "clip_frame_id": int(i)})

    est_path = os.path.join(seq_root, "est_focal.txt")
    with open(est_path, "w") as f:
        f.write(str(focal_ref))

    k_path = os.path.join(seq_root, "cam_K.json")
    with open(k_path, "w") as f:
        json.dump(K.tolist(), f, indent=2)

    map_path = os.path.join(seq_root, "hawor_clip_frame_map.json")
    with open(map_path, "w") as f:
        json.dump(
            {
                "clip_dir": os.path.abspath(image_dir),
                "stream_id": "pinhole",
                "n_frames": len(map_records),
                "frames": map_records,
            },
            f,
            indent=2,
        )

    return focal_ref, len(map_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--clip-dir",
        default=None,
        help="HOT3D extracted clip: distorted raw jpgs + per-frame cameras.json. Not used with --pinhole-image-dir.",
    )
    parser.add_argument(
        "--pinhole-image-dir",
        default=None,
        help="Flat folder of rectified RGB frames (requires --cam-k). Mutually exclusive with --clip-dir.",
    )
    parser.add_argument(
        "--cam-k",
        default=None,
        help="JSON 3×3 intrinsics with --pinhole-image-dir, e.g. '[[fx,0,cx],[0,fy,cy],[0,0,1]]'.",
    )
    parser.add_argument(
        "--pinhole-extensions",
        default=".jpg,.jpeg,.png",
        help="Comma-separated suffixes for --pinhole-image-dir (default: .jpg,.jpeg,.png).",
    )
    parser.add_argument("--stream-id", default="214-1", help="Aria RGB stream key in cameras.json (HOT3D mode).")
    parser.add_argument(
        "--hot3d-module-path",
        default=None,
        help="Path to hot3d repo root (contains clips/clip_util.py); HOT3D mode only.",
    )
    parser.add_argument("--hawor-seq-root", required=True, help="Parent directory for HaWoR sequence folder.")
    parser.add_argument("--seq-name", required=True, help="Sequence folder name under hawor-seq-root.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite outputs even if extracted_images already exists.",
    )
    args = parser.parse_args()

    use_pinhole = bool(args.pinhole_image_dir and str(args.pinhole_image_dir).strip())
    if use_pinhole:
        if not args.cam_k:
            parser.error("--pinhole-image-dir requires --cam-k (JSON 3×3 intrinsics).")
        if args.clip_dir:
            print("[WARN] --clip-dir ignored when --pinhole-image-dir is set.", flush=True)
    else:
        if not args.clip_dir:
            parser.error("Provide --clip-dir (HOT3D) or --pinhole-image-dir + --cam-k (pinhole folder).")

    seq_root = os.path.join(os.path.abspath(args.hawor_seq_root), args.seq_name)
    img_out = os.path.join(seq_root, "extracted_images")
    os.makedirs(img_out, exist_ok=True)
    dummy_mp4 = os.path.join(os.path.abspath(args.hawor_seq_root), f"{args.seq_name}.mp4")

    if use_pinhole:
        focal_ref, n_out = _export_pinhole_flat_dir(
            args.pinhole_image_dir,
            seq_root,
            args.cam_k,
            args.pinhole_extensions,
            args.max_frames,
            args.overwrite,
        )
        if not os.path.isfile(dummy_mp4):
            open(dummy_mp4, "a").close()
        print("Done (pinhole folder).", flush=True)
        print(f"  seq_root:  {seq_root}", flush=True)
        print(f"  images:    {img_out}  ({n_out} jpgs)", flush=True)
        print(f"  est_focal: {os.path.join(seq_root, 'est_focal.txt')}  -> {focal_ref}", flush=True)
        print(f"  cam_K:     {os.path.join(seq_root, 'cam_K.json')}", flush=True)
        print(f"  frame map: {os.path.join(seq_root, 'hawor_clip_frame_map.json')}", flush=True)
        print(f"  dummy mp4: {dummy_mp4}", flush=True)
        return

    _ensure_clip_util(args.hot3d_module_path)

    clip_dir = os.path.abspath(args.clip_dir)
    frame_ids = _list_clip_frame_ids(clip_dir)
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise RuntimeError(
            f"No *.cameras.json under {clip_dir} (expected one untarred HOT3D clip dir, not a multi-seq parent)."
            f"{_hint_if_clip_parent_not_leaf(clip_dir)}"
        )

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(it, **_kwargs):
            return it

    n = len(frame_ids)
    print(f"Undistorting {n} frames (distorted raw→pinhole); full Aria res can take several minutes.", flush=True)

    map_records = []
    focal_ref: Optional[float] = None

    for i, fid in enumerate(tqdm(frame_ids, desc="prepare/warp", unit="frame")):
        out_jpg = os.path.join(img_out, f"{i:06d}.jpg")
        if os.path.isfile(out_jpg) and not args.overwrite:
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

    print("Done (HOT3D clip).", flush=True)
    print(f"  seq_root:     {seq_root}", flush=True)
    print(f"  images:       {img_out}  ({len(map_records)} files)", flush=True)
    print(f"  est_focal:    {est_path}  -> {focal_ref}", flush=True)
    print(f"  frame map:    {map_path}", flush=True)
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
