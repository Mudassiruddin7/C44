"""Adaptive attacks that simulate camera image formation on rendered clips (for the robustness section)."""

from __future__ import annotations

import cv2
import numpy as np


def _line_kernel(length: float, direction: np.ndarray) -> np.ndarray:
    L = max(1, int(np.ceil(length)))
    size = 2 * L + 1
    k = np.zeros((size, size), np.float32)
    u = np.asarray(direction, float)
    u = u / (np.linalg.norm(u) + 1e-8)
    for t in np.linspace(-length / 2, length / 2, max(2, int(np.ceil(length * 4)))):
        x, y = L + t * u[0], L + t * u[1]
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= x0 + dx < size and 0 <= y0 + dy < size:
                k[y0 + dy, x0 + dx] += w
    return k / k.sum()


def motion_blur(frames: np.ndarray, m: np.ndarray, direction: np.ndarray, exposure: float = 0.5,
                max_len: float = 30.0) -> np.ndarray:
    """Blur each frame along its motion direction with length exposure * motion (a 'smart' attacker)."""
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        L = float(np.clip(exposure * m[t], 0.0, max_len))
        if L < 0.75 or not np.all(np.isfinite(direction[t])):
            out[t] = f
            continue
        out[t] = cv2.filter2D(f, -1, _line_kernel(L, direction[t]), borderType=cv2.BORDER_REFLECT)
    return out


def exposure_jitter(frames: np.ndarray, gain_sigma: float, wb_sigma: float, seed: int = 0) -> np.ndarray:
    """Fake auto-exposure / auto-white-balance: a random walk of the global gain (log step std `gain_sigma`) and of the
    red/blue balance (`wb_sigma`). An attacker who knows the CPS exposure cue sets the sigmas to the real clip's level."""
    rng = np.random.default_rng(seed)
    T = len(frames)
    g = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, gain_sigma, T - 1))])
    w = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, wb_sigma, T - 1))])
    gains = np.exp(np.stack([g + w / 2, g, g - w / 2], axis=1))[:, None, None, :]
    return np.clip(frames.astype(np.float32) * gains + 0.5, 0, 255).astype(np.uint8)


def scaled_noise(frames: np.ndarray, strength: float, seed: int = 0) -> np.ndarray:
    """Sensor noise whose whole model scales together: shot, read and fixed-pattern variances are all proportional to
    `strength` (1.0 = the default sensor_noise). Scaling only the shot-noise gain leaves a read-noise/PRNU floor that is
    already louder than many real clips, so the level could not be matched."""
    s = float(strength)
    return sensor_noise(frames, gain=0.004 * s, read_std=0.004 * np.sqrt(s), prnu_std=0.01 * np.sqrt(s), seed=seed)


def sensor_noise(frames: np.ndarray, gain: float = 0.004, read_std: float = 0.004, prnu_std: float = 0.01,
                 seed: int = 0) -> np.ndarray:
    """Poisson-Gaussian noise with a fixed PRNU pattern, applied in (approximately) linear light."""
    rng = np.random.default_rng(seed)
    H, W = frames.shape[1:3]
    prnu = 1.0 + prnu_std * rng.standard_normal((H, W, 1)).astype(np.float32)
    lin = (frames.astype(np.float32) / 255.0) ** 2.2
    out = np.empty_like(frames)
    for t in range(len(frames)):
        x = lin[t] * prnu
        x = rng.poisson(np.clip(x, 0, None) / gain) * gain + read_std * rng.standard_normal(x.shape)
        out[t] = np.clip(np.clip(x, 0, 1) ** (1 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out
