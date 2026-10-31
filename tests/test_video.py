import numpy as np

from c44 import video


def test_crop_resize_shapes():
    img = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    out = video.crop_resize(img, 704, 480)
    assert out.shape == (480, 704, 3)
    tall = np.zeros((1920, 1080, 3), np.uint8)
    assert video.crop_resize(tall, 704, 480).shape == (480, 704, 3)


def test_crop_resize_K_maps_projections():
    K = np.array([[1500.0, 0, 959.5], [0, 1500.0, 539.5], [0, 0, 1]])
    X = np.array([[0.3, -0.2, 4.0], [-1.0, 0.5, 6.0], [0.0, 0.0, 3.0]]).T
    uv = (K @ X)[:2] / X[2]
    K2 = video.crop_resize_K(K, (1920, 1080), (704, 480))
    uv2 = (K2 @ X)[:2] / X[2]
    x0, y0, cw, ch = video.crop_box(1920, 1080, 704, 480)
    expect = np.stack([(uv[0] - x0 + 0.5) * 704 / cw - 0.5, (uv[1] - y0 + 0.5) * 480 / ch - 0.5])
    assert np.allclose(uv2, expect, atol=1e-9)


def test_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    base = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
    import cv2
    base = cv2.resize(base, (704, 480), interpolation=cv2.INTER_CUBIC)
    frames = np.stack([np.roll(base, 2 * t, axis=1) for t in range(12)])
    p = video.write_mp4(frames, tmp_path / "x.mp4", fps=30, crf=12, preset="veryfast")
    info = video.probe(p)
    assert (info["width"], info["height"]) == (704, 480) and abs(info["fps"] - 30) < 0.01
    back = video.read_video(p)
    assert back.shape == frames.shape
    # luma only: yuv420p halves chroma resolution, which dominates RGB error on this random colour texture
    luma = lambda x: x.astype(float) @ np.array([0.299, 0.587, 0.114])
    mse = np.mean((luma(back) - luma(frames)) ** 2)
    assert 10 * np.log10(255 ** 2 / mse) > 35
