"""
Offline MANO solid mesh overlay on pinhole RGB using pyrender (OffscreenRenderer) + alpha blend.
Encode with ffmpeg (libx264). No moderngl / aitviewer window.

If headless rendering fails, try: export PYOPENGL_PLATFORM=osmesa
(requires Mesa/OSMesa libs in the image).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Sequence

import cv2
import numpy as np

LEFT_COLOR_BGR = (80, 220, 80)
RIGHT_COLOR_BGR = (255, 120, 60)


def _bgr_to_rgba01(color_bgr: tuple[int, int, int], alpha: float = 0.85) -> np.ndarray:
    b, g, r = color_bgr
    return np.array([r, g, b, int(alpha * 255)], dtype=np.uint8) / 255.0


def _T_world_cam_from_c2w(R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_c2w.astype(np.float32)
    T[:3, 3] = t_c2w.astype(np.float32).reshape(3)
    return T


def _apply_align_to_cam(
    R_c2w_slam: np.ndarray,
    t_c2w_slam: np.ndarray,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map SLAM camera c2w into the same aligned world frame as ``verts_aligned``."""
    Rf = R_align.astype(np.float64)
    tf = np.asarray(t_align, dtype=np.float64).reshape(3)
    Rc = R_c2w_slam.astype(np.float64)
    tc = np.asarray(t_c2w_slam, dtype=np.float64).reshape(3)
    R_out = (Rf @ Rc).astype(np.float32)
    t_out = (Rf @ tc + tf).astype(np.float32)
    return R_out, t_out


def render_solid_hands_overlay_pyrender(
    img_bgr: np.ndarray,
    verts_world: np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray,
    T_world_cam: np.ndarray,
    K: np.ndarray,
    faces_right: np.ndarray,
    alpha: float,
    renderer: object,
) -> np.ndarray:
    """
    verts_world: (2, V, 3). ``renderer``: pyrender.OffscreenRenderer(w, h).
    """
    import pyrender
    import trimesh

    h, w = img_bgr.shape[:2]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.06, 0.06, 0.06, 1.0])
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=50.0)
    scene.add(camera, pose=np.eye(4, dtype=np.float32))

    cv_to_gl = np.eye(4, dtype=np.float32)
    cv_to_gl[1, 1] = -1.0
    cv_to_gl[2, 2] = -1.0

    T_cam_world = np.linalg.inv(T_world_cam).astype(np.float32)
    colors = [LEFT_COLOR_BGR, RIGHT_COLOR_BGR]

    for side in range(2):
        if not bool(valid_mask[side]):
            continue
        vw = verts_world[side].astype(np.float32)
        n_v = vw.shape[0]
        verts_h = np.concatenate([vw, np.ones((n_v, 1), dtype=np.float32)], axis=1)
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

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    light_pose = np.eye(4, dtype=np.float32)
    scene.add(light, pose=light_pose)

    color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    overlay_bgr = cv2.cvtColor(color_rgba[..., :3], cv2.COLOR_RGB2BGR)
    out = img_bgr.copy()
    mask = depth > 0
    if np.any(mask):
        a = float(np.clip(alpha, 0.0, 1.0))
        out[mask] = ((1.0 - a) * out[mask] + a * overlay_bgr[mask]).astype(np.uint8)
    return out


def export_mesh_overlay_video_ffmpeg(
    image_paths: list[str],
    verts_aligned: np.ndarray,
    pred_valid: np.ndarray,
    R_c2w_slam: np.ndarray,
    t_c2w_slam: np.ndarray,
    R_align: np.ndarray,
    t_align: np.ndarray,
    fx: float,
    faces_right: np.ndarray,
    out_mp4: str,
    fps: float = 30.0,
    mesh_alpha: float = 0.75,
    work_dir: str | None = None,
    keep_frames: bool = False,
    cam_K: np.ndarray | None = None,
) -> None:
    if not image_paths:
        raise ValueError("no frames")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install ffmpeg to encode the overlay video.")

    import pyrender

    T = len(image_paths)
    if verts_aligned.shape[0] < T or R_c2w_slam.shape[0] < T or t_c2w_slam.shape[0] < T:
        raise ValueError(
            f"length mismatch: images={T}, verts={verts_aligned.shape[0]}, "
            f"cam R={R_c2w_slam.shape[0]}, cam t={t_c2w_slam.shape[0]}"
        )

    first = cv2.imread(image_paths[0])
    if first is None:
        raise RuntimeError(f"cv2.imread failed: {image_paths[0]}")
    h, w = first.shape[:2]
    if cam_K is not None:
        K = np.asarray(cam_K, dtype=np.float32).reshape(3, 3)
    else:
        cx, cy = w * 0.5, h * 0.5
        K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(os.path.abspath(out_mp4)), "_frames_mesh_overlay_tmp")
    os.makedirs(work_dir, exist_ok=True)
    for old in os.listdir(work_dir):
        os.remove(os.path.join(work_dir, old))

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        for i, imp in enumerate(image_paths):
            img = cv2.imread(imp)
            if img is None:
                raise RuntimeError(f"cv2.imread failed: {imp}")
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            R_c2w, t_c2w = _apply_align_to_cam(R_c2w_slam[i], t_c2w_slam[i], R_align, t_align)
            T_wc = _T_world_cam_from_c2w(R_c2w, t_c2w)
            vw = verts_aligned[i]
            vm = pred_valid[i]
            out_bgr = render_solid_hands_overlay_pyrender(
                img, vw, vm, T_wc, K, faces_right, mesh_alpha, renderer
            )
            cv2.imwrite(os.path.join(work_dir, f"{i:06d}.png"), out_bgr)
    finally:
        renderer.delete()

    out_mp4 = os.path.abspath(out_mp4)
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    pattern = os.path.join(work_dir, "%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_mp4,
    ]
    subprocess.run(cmd, check=True)

    if not keep_frames:
        for old in os.listdir(work_dir):
            os.remove(os.path.join(work_dir, old))
        try:
            os.rmdir(work_dir)
        except OSError:
            pass
