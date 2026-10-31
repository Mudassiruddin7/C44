"""Synthetic 'camera' vs 'renderer' clips of the same panning scene, for testing CPS feature code.

camera: temporal supersampling over the exposure (true motion blur), linear-light Poisson-Gaussian noise
        and a fixed PRNU pattern, then sRGB quantisation.
render: instantaneous pinhole crop, no blur, no noise.
This only checks that the feature code measures what it claims; it says nothing about real data.
"""

import cv2
import numpy as np


def texture(H: int, W: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((H, W, 3), np.float32)
    for sigma, amp in ((1.0, 0.6), (3.0, 1.0), (9.0, 1.4), (27.0, 1.8)):
        n = rng.standard_normal((H, W, 3)).astype(np.float32)
        acc += amp * cv2.GaussianBlur(n, (0, 0), sigma)
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    return 0.12 + 0.76 * acc


def _crop(tex: np.ndarray, x: float, y: float, h: int, w: int) -> np.ndarray:
    M = np.float32([[1, 0, -x], [0, 1, -y]])
    return cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def clip(kind: str, seed: int, T: int = 48, h: int = 192, w: int = 256, exposure: float = 0.6):
    rng = np.random.default_rng(seed)
    tex = texture(h + 64, w + 64 + 800, seed)
    speeds = 0.5 + 9.0 * np.abs(np.sin(np.linspace(0, rng.uniform(2, 5), T) + rng.uniform(0, 3)))
    xs = 8 + np.concatenate([[0.0], np.cumsum(speeds[:-1])])
    prnu = 1.0 + 0.02 * rng.standard_normal((h, w, 1)).astype(np.float32)
    frames = []
    for t in range(T):
        if kind == "camera":
            L = exposure * speeds[t]
            offs = np.linspace(-L / 2, L / 2, max(1, int(np.ceil(L * 3))))
            img = np.mean([_crop(tex, xs[t] + o, 16, h, w) for o in offs], axis=0)
            lin = img ** 2.2 * prnu
            gain = 0.003
            lin = rng.poisson(np.clip(lin, 0, None) / gain) * gain + 0.003 * rng.standard_normal(lin.shape)
            img = np.clip(lin, 0, 1) ** (1 / 2.2)
        else:
            img = _crop(tex, xs[t], 16, h, w)
        frames.append(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8))
    frames = np.stack(frames)
    direction = np.tile([1.0, 0.0], (T, 1))
    return frames, speeds, direction
