import numpy as np
import pytest

from c44 import cameras as cam


def _rand_R(rng):
    q = rng.standard_normal(4)
    return cam.quat_to_R(q / np.linalg.norm(q))


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = _rand_R(rng)
        assert np.allclose(cam.quat_to_R(cam.R_to_quat(R)), R, atol=1e-10)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_slerp_endpoints_and_norm():
    rng = np.random.default_rng(1)
    q0, q1 = cam.R_to_quat(_rand_R(rng)), cam.R_to_quat(_rand_R(rng))
    assert np.allclose(cam.quat_to_R(cam.slerp(q0, q1, 0.0)), cam.quat_to_R(q0))
    assert np.allclose(cam.quat_to_R(cam.slerp(q0, q1, 1.0)), cam.quat_to_R(q1))
    assert abs(np.linalg.norm(cam.slerp(q0, q1, 0.37)) - 1) < 1e-12


def test_invert_and_flip():
    rng = np.random.default_rng(2)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _rand_R(rng), rng.standard_normal(3)
    assert np.allclose(cam.invert(T) @ T, np.eye(4), atol=1e-12)
    assert np.allclose(cam.flip_yz(cam.flip_yz(T)), T)


def test_interpolate_hits_keys():
    rng = np.random.default_rng(3)
    poses = np.stack([np.eye(4) for _ in range(5)])
    for p in poses:
        p[:3, :3], p[:3, 3] = _rand_R(rng), rng.standard_normal(3)
    s = np.array([0.0, 1.0, 2.5, 3.0, 7.0])
    out = cam.interpolate_c2w(poses, s, s)
    assert np.allclose(out, poses, atol=1e-9)


def test_step_motion_translation_and_rotation():
    K = np.array([[500.0, 0, 352], [0, 500.0, 240], [0, 0, 1]])
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = 0.1                                # 10 cm sideways at 2 m depth -> 500 * 0.1 / 2 = 25 px
    assert abs(cam.step_motion(a, b, K, 704, 480, depth=2.0) - 25.0) < 1e-6
    c = np.eye(4)
    th = np.deg2rad(1.0)                          # 1 degree pan: f*th at the centre, f*th*(1+(x/f)^2) off-axis
    c[:3, :3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    m5 = cam.step_motion(a, c, K, 704, 480, depth=5.0)
    assert abs(m5 - cam.step_motion(a, c, K, 704, 480, depth=50.0)) < 1e-9   # pure rotation: depth-independent
    x_max = 0.4 * 704
    assert 500 * th <= m5 <= 500 * th * (1 + (x_max / 500) ** 2) * 1.05


def test_resample_by_profile_follows_steps():
    K = np.array([[400.0, 0, 352], [0, 400.0, 240], [0, 0, 1]])
    keys = np.stack([np.eye(4) for _ in range(10)])
    for i, p in enumerate(keys):
        p[0, 3] = 0.05 * i ** 1.3                 # non-uniform key spacing
    steps = np.abs(np.sin(np.linspace(0, 3, 30))) * 3 + 0.5
    out = cam.resample_by_profile(keys, K, 704, 480, depth=3.0, steps=steps)
    assert out is not None and len(out) == len(steps) + 1
    got = cam.path_motion(out, K, 704, 480, depth=3.0)
    assert np.allclose(got, steps, rtol=1e-6, atol=1e-6)
    assert cam.resample_by_profile(keys, K, 704, 480, depth=3.0, steps=steps * 100) is None


def test_flashworld_normalisation_matches_original():
    torch = pytest.importorskip("torch")
    from tests import fw_reference as ref
    rng = np.random.default_rng(4)
    N, W, H = 24, 704, 480
    c2w = np.stack([np.eye(4) for _ in range(N)])
    for i, p in enumerate(c2w):
        p[:3, :3], p[:3, 3] = _rand_R(rng), rng.standard_normal(3) * (0.2 + i / N)
    K = np.array([[420.0, 0, 350.0], [0, 418.0, 241.0], [0, 0, 1]])
    cam11 = cam.to_fw_cam11(c2w, K, W, H)
    ours, ref_w2c, T_norm = cam.fw_normalize(cam11)
    theirs, their_ref, their_T = ref.normalize_cameras(torch.tensor(cam11, dtype=torch.float32)[None], return_meta=True)
    theirs = theirs[0].numpy().astype(np.float64)
    assert abs(T_norm - float(their_T)) < 1e-4 * max(1.0, T_norm)
    assert np.allclose(ref_w2c, their_ref[0, 0].numpy(), atol=1e-4)
    # quaternions may differ by sign; compare rotations and the rest
    assert np.allclose(cam.quat_to_R(ours[:, :4]), cam.quat_to_R(theirs[:, :4]), atol=1e-4)
    assert np.allclose(ours[:, 4:], theirs[:, 4:], atol=1e-4)
    # render-time conversion preserves relative camera motion (FlashWorld's world frame is camera 0's
    # OpenGL frame, so absolute poses differ by that frame change; relative poses must be identical)
    c2w_cv, Ks = cam.fw_to_render(ours, W, H)
    for j in (1, 7, N - 1):
        rel_out = cam.invert(c2w_cv[0]) @ c2w_cv[j]
        rel_in = cam.invert(c2w[0]) @ c2w[j]
        rel_in[:3, 3] /= (T_norm + 1e-2)
        assert np.allclose(rel_out, rel_in, atol=1e-6)
    assert np.allclose(c2w_cv[0, :3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-9)
    assert np.allclose(Ks[0], K, atol=1e-9)
