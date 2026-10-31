import json

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from c44 import dl3dv, gs_render


def _write_3dgs_ply(path, N=50, deg=3, seed=0):
    rng = np.random.default_rng(seed)
    K = (deg + 1) ** 2
    cols = {"x": rng.normal(size=N), "y": rng.normal(size=N), "z": rng.normal(size=N) + 5,
            "nx": np.zeros(N), "ny": np.zeros(N), "nz": np.zeros(N)}
    dc = rng.normal(size=(N, 3))
    rest = rng.normal(size=(N, 3, K - 1))          # channel-major in the file
    for i in range(3):
        cols[f"f_dc_{i}"] = dc[:, i]
    flat = rest.reshape(N, -1)
    for i in range(flat.shape[1]):
        cols[f"f_rest_{i}"] = flat[:, i]
    cols["opacity"] = rng.normal(size=N)
    for i in range(3):
        cols[f"scale_{i}"] = rng.normal(size=N) - 3
    q = rng.normal(size=(N, 4))
    for i in range(4):
        cols[f"rot_{i}"] = q[:, i]
    arr = np.rec.fromarrays([cols[k].astype(np.float32) for k in cols], names=list(cols))
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return dc, rest, cols


def test_load_gs_ply_activations_and_sh_layout(tmp_path):
    dc, rest, cols = _write_3dgs_ply(tmp_path / "point_cloud.ply")
    gs = dl3dv.load_gs_ply(tmp_path / "point_cloud.ply")
    assert gs["sh_degree"] == 3 and gs["sh"].shape == (50, 16, 3)
    assert np.allclose(gs["sh"][:, 0], dc, atol=1e-6)
    assert np.allclose(gs["sh"][:, 1:], rest.transpose(0, 2, 1), atol=1e-6)
    assert np.allclose(gs["scales"], np.exp(np.stack([cols[f"scale_{i}"] for i in range(3)], -1)), rtol=1e-5)
    assert np.all((gs["opacities"] > 0) & (gs["opacities"] < 1))
    assert np.allclose(np.linalg.norm(gs["quats"], axis=-1), 1, atol=1e-5)


def test_load_cameras_json_and_locate(tmp_path):
    root = tmp_path / "scene" / "point_cloud" / "iteration_30000"
    root.mkdir(parents=True)
    _write_3dgs_ply(root / "point_cloud.ply")
    (tmp_path / "scene" / "point_cloud" / "iteration_7000").mkdir()
    _write_3dgs_ply(tmp_path / "scene" / "point_cloud" / "iteration_7000" / "point_cloud.ply", seed=1)
    R = np.eye(3)
    cams = [{"id": i, "img_name": f"frame_{i:05d}", "width": 960, "height": 540, "position": [0.1 * i, 0, 0],
             "rotation": R.tolist(), "fx": 800.0, "fy": 801.0} for i in (2, 0, 1)]
    (tmp_path / "scene" / "cameras.json").write_text(json.dumps(cams))
    files = dl3dv.locate_gs_files(tmp_path / "scene")
    assert "iteration_30000" in str(files["ply"])
    out = dl3dv.load_cameras(files["cameras"])
    assert [c["name"] for c in out] == ["frame_00000", "frame_00001", "frame_00002"]
    assert np.allclose(out[1]["c2w"][:3, 3], [0.1, 0, 0]) and out[0]["K"][0, 2] == 480
    gs = dl3dv.load_gs_ply(files["ply"])
    assert 1.0 < dl3dv.scene_depth(gs, out) < 10.0


def test_flashworld_param_split():
    torch = pytest.importorskip("torch")
    N, deg = 10, 2
    K = (deg + 1) ** 2
    p = torch.randn(N, 3 + 1 + 3 + 4 + 3 * K)
    gs = gs_render.flashworld_params_to_gs(p)
    assert gs["sh_degree"] == 2 and gs["colors"].shape == (N, K, 3)
    assert torch.allclose(gs["means"], p[:, :3]) and torch.allclose(gs["opacities"], p[:, 3])
    assert torch.allclose(gs["quats"].norm(dim=-1), torch.ones(N))
    assert torch.allclose(gs["colors"].reshape(N, -1), p[:, 11:])
    with pytest.raises(ValueError):
        gs_render.flashworld_params_to_gs(torch.randn(N, 20))


def test_index_uses_only_3dgs_zips(tmp_path, monkeypatch):
    """imgs_undist/<subset>/<hash>.zip shares the scene hash and sorts after 3DGS/: it must not override it."""
    import huggingface_hub
    h = "a" * 64
    files = {
        "DL3DV/DL3DV-ALL-video": [f"5K/{h}/video.mp4"],
        "DL3DV/DL3DV-GS-960P": [f"3DGS/5K/{h}.zip", f"imgs_undist/5K/{h}.zip", "README.md"],
    }

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo, repo_type=None):
            return files[repo]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    idx = dl3dv.build_index(token="x", cache=tmp_path / "idx.json")
    assert idx[h]["gs"] == f"3DGS/5K/{h}.zip"


def test_npr_state_dict_prefix():
    from c44.detectors.npr import strip_module_prefix
    assert strip_module_prefix({"module.conv1.weight": 1, "fc1.bias": 2}) == {"conv1.weight": 1, "fc1.bias": 2}
