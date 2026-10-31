"""Camera utilities (numpy, float64).

Conventions
- OpenCV camera: x right, y down, looking +z (VGGT, DL3DV/3DGS cameras.json, gsplat viewmats).
- OpenGL camera: x right, y up, looking -z (FlashWorld cameras).
- c2w: 4x4 camera-to-world. Quaternions are (w, x, y, z), real part first (FlashWorld/PyTorch3D).
- FlashWorld camera vector (cam11): [qw, qx, qy, qz, tx, ty, tz, fx/W, fy/H, cx/W, cy/H] describing an
  OpenGL c2w pose; see FlashWorld utils.normalize_cameras / reconstruction_model.render.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- rotations
def quat_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z) with w >= 0 (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    m = R.reshape(-1, 3, 3)
    out = np.empty((m.shape[0], 4))
    for n, M in enumerate(m):
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q = [0.25 * S, (M[2, 1] - M[1, 2]) / S, (M[0, 2] - M[2, 0]) / S, (M[1, 0] - M[0, 1]) / S]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            S = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q = [(M[2, 1] - M[1, 2]) / S, 0.25 * S, (M[0, 1] + M[1, 0]) / S, (M[0, 2] + M[2, 0]) / S]
        elif M[1, 1] > M[2, 2]:
            S = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q = [(M[0, 2] - M[2, 0]) / S, (M[0, 1] + M[1, 0]) / S, 0.25 * S, (M[1, 2] + M[2, 1]) / S]
        else:
            S = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q = [(M[1, 0] - M[0, 1]) / S, (M[0, 2] + M[2, 0]) / S, (M[1, 2] + M[2, 1]) / S, 0.25 * S]
        q = np.asarray(q) / np.linalg.norm(q)
        out[n] = q if q[0] >= 0 else -q
    return out.reshape(R.shape[:-2] + (4,))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


# ----------------------------------------------------------------------------- poses
def to44(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.shape[-2:] == (4, 4):
        return m.copy()
    out = np.zeros(m.shape[:-2] + (4, 4))
    out[..., :3, :4] = m
    out[..., 3, 3] = 1.0
    return out


def invert(T: np.ndarray) -> np.ndarray:
    T = to44(T)
    R, t = T[..., :3, :3], T[..., :3, 3:]
    out = np.zeros_like(T)
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3:] = -np.swapaxes(R, -1, -2) @ t
    out[..., 3, 3] = 1.0
    return out


def flip_yz(c2w: np.ndarray) -> np.ndarray:
    """OpenCV <-> OpenGL camera axes (self-inverse): negate the camera y and z axes."""
    out = to44(c2w)
    out[..., :3, 1:3] *= -1
    return out


def interpolate_c2w(c2ws: np.ndarray, s_src: np.ndarray, s_dst: np.ndarray) -> np.ndarray:
    """Interpolate poses given at increasing positions s_src to positions s_dst (slerp + lerp)."""
    c2ws = to44(c2ws)
    s_src = np.asarray(s_src, float)
    qs = R_to_quat(c2ws[:, :3, :3])
    out = np.zeros((len(s_dst), 4, 4))
    for n, s in enumerate(s_dst):
        i = int(np.clip(np.searchsorted(s_src, s, side="right") - 1, 0, len(s_src) - 2))
        span = s_src[i + 1] - s_src[i]
        f = 0.0 if span <= 0 else float(np.clip((s - s_src[i]) / span, 0.0, 1.0))
        out[n, :3, :3] = quat_to_R(slerp(qs[i], qs[i + 1], f))
        out[n, :3, 3] = (1 - f) * c2ws[i, :3, 3] + f * c2ws[i + 1, :3, 3]
        out[n, 3, 3] = 1.0
    return out


def step_motion(c2w_a: np.ndarray, c2w_b: np.ndarray, K: np.ndarray, W: int, H: int,
                depth: float, grid: int = 7) -> float:
    """Median image displacement (px) of points at `depth` in view a when the camera moves a->b (OpenCV)."""
    us, vs = np.meshgrid(np.linspace(0.1 * W, 0.9 * W, grid), np.linspace(0.1 * H, 0.9 * H, grid))
    pix = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
    Xc = (np.linalg.inv(K) @ pix) * depth
    a, wb = to44(c2w_a), invert(c2w_b)
    Xw = a[:3, :3] @ Xc + a[:3, 3:]
    Xb = wb[:3, :3] @ Xw + wb[:3, 3:]
    ok = Xb[2] > 1e-6
    if not ok.any():
        return float("inf")
    uv = (K @ Xb[:, ok])[:2] / Xb[2:3, ok]
    return float(np.median(np.linalg.norm(uv - pix[:2, ok], axis=0)))


def path_motion(c2ws: np.ndarray, K: np.ndarray, W: int, H: int, depth: float) -> np.ndarray:
    return np.array([step_motion(c2ws[i], c2ws[i + 1], K, W, H, depth) for i in range(len(c2ws) - 1)])


def resample_by_profile(c2ws: np.ndarray, K: np.ndarray, W: int, H: int, depth: float, steps: np.ndarray,
                        start: int = 0) -> np.ndarray | None:
    """Poses along a key-pose path whose consecutive image motion follows `steps` (px/frame).

    Arc length is measured in image motion of points at `depth`; returns len(steps)+1 poses, or None if
    the path after `start` is too short.
    """
    c2ws = to44(c2ws)[start:]
    if len(c2ws) < 2:
        return None
    seg = path_motion(c2ws, K, W, H, depth)
    s_src = np.concatenate([[0.0], np.cumsum(seg)])
    s_dst = np.concatenate([[0.0], np.cumsum(np.asarray(steps, float))])
    if s_dst[-1] > s_src[-1] or not np.all(np.isfinite(s_src)):
        return None
    # strictly increasing source positions (drop stationary key poses)
    keep = np.concatenate([[True], np.diff(s_src) > 1e-9])
    return interpolate_c2w(c2ws[keep], s_src[keep], s_dst)


# ----------------------------------------------------------------------------- FlashWorld interop
def to_fw_cam11(c2w_cv: np.ndarray, K: np.ndarray, W: int, H: int) -> np.ndarray:
    """OpenCV c2w poses [N,4,4] + pixel intrinsics ([3,3] or [N,3,3]) -> FlashWorld camera vectors [N,11]."""
    c2w_gl = flip_yz(c2w_cv)
    N = len(c2w_gl)
    K = np.broadcast_to(np.asarray(K, float), (N, 3, 3))
    q = R_to_quat(c2w_gl[:, :3, :3])
    intr = np.stack([K[:, 0, 0] / W, K[:, 1, 1] / H, K[:, 0, 2] / W, K[:, 1, 2] / H], -1)
    return np.concatenate([q, c2w_gl[:, :3, 3], intr], -1)


def fw_c2w(cam11: np.ndarray) -> np.ndarray:
    cam11 = np.asarray(cam11, float)
    c2w = np.zeros((len(cam11), 4, 4))
    c2w[:, :3, :3] = quat_to_R(cam11[:, :4])
    c2w[:, :3, 3] = cam11[:, 4:7]
    c2w[:, 3, 3] = 1.0
    return c2w


def fw_normalize(cam11: np.ndarray, ref_w2c: np.ndarray | None = None, T_norm: float | None = None):
    """numpy port of FlashWorld utils.normalize_cameras(n_frame=None): returns (cam11_norm, ref_w2c, T_norm)."""
    c2w = fw_c2w(cam11)
    if ref_w2c is None:
        ref_w2c = invert(c2w[0])
    rel = to44(ref_w2c)[None] @ c2w
    if T_norm is None:
        T_norm = float(np.linalg.norm(rel[:, :3, 3], axis=-1).max())
    rel[:, :3, 3] /= (T_norm + 1e-2)
    out = np.concatenate([R_to_quat(rel[:, :3, :3]), rel[:, :3, 3], np.asarray(cam11, float)[:, 7:]], -1)
    return out, to44(ref_w2c), float(T_norm)


def fw_to_render(cam11_norm: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """FlashWorld (normalised) cameras -> OpenCV c2w [N,4,4] and pixel intrinsics [N,3,3] for gsplat."""
    c2w_cv = flip_yz(fw_c2w(cam11_norm))
    c = np.asarray(cam11_norm, float)
    K = np.zeros((len(c), 3, 3))
    K[:, 0, 0], K[:, 1, 1] = c[:, 7] * W, c[:, 8] * H
    K[:, 0, 2], K[:, 1, 2] = c[:, 9] * W, c[:, 10] * H
    K[:, 2, 2] = 1.0
    return c2w_cv, K
