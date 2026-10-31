"""DL3DV access: raw videos (RC source) and pretrained 3D Gaussian Splatting scenes (RR source).

Both repos are gated on Hugging Face ("auto" approval): accept the terms once on the dataset pages with
your account, then provide HF_TOKEN. Layouts (listed 2026-10-31):
  DL3DV/DL3DV-ALL-video : <subset>/<scene_hash>/video.mp4
  DL3DV/DL3DV-GS-960P   : 3DGS/<subset>/<scene_hash>.zip   (standard 3DGS output: point_cloud.ply, cameras.json)
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from . import config

REPO_VIDEO = "DL3DV/DL3DV-ALL-video"
REPO_GS = "DL3DV/DL3DV-GS-960P"


# ----------------------------------------------------------------------------- index
def build_index(token: str | None = None, cache: Path | None = None, refresh: bool = False) -> dict[str, dict]:
    """scene hash -> {'video': repo path, 'gs': repo path or None}; cached as JSON in the work dir."""
    cache = cache or (config.work_dir() / "dl3dv_index.json")
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    from huggingface_hub import HfApi
    api = HfApi(token=token or config.hf_token())
    vids = api.list_repo_files(REPO_VIDEO, repo_type="dataset")
    gss = api.list_repo_files(REPO_GS, repo_type="dataset")
    v = {p.split("/")[-2]: p for p in vids if p.endswith("/video.mp4")}
    g = {Path(p).stem: p for p in gss if p.endswith(".zip")}
    index = {h: {"video": v[h], "gs": g.get(h)} for h in sorted(v)}
    cache.write_text(json.dumps(index))
    return index


def sample_scenes(index: dict, n: int, seed: int = 0, require_gs: bool = True, exclude=()) -> list[str]:
    pool = sorted(h for h, r in index.items() if (r["gs"] or not require_gs) and h not in set(exclude))
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.permutation(len(pool))[:n]]


def fetch(repo: str, path_in_repo: str, token: str | None = None, dst: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    dst = dst or (config.work_dir() / "downloads")
    return Path(hf_hub_download(repo, path_in_repo, repo_type="dataset", token=token or config.hf_token(),
                                local_dir=str(dst)))


def fetch_video(scene: str, index: dict, token: str | None = None) -> Path:
    return fetch(REPO_VIDEO, index[scene]["video"], token)


def fetch_gs(scene: str, index: dict, token: str | None = None) -> dict[str, Path]:
    """Download and unpack a 3DGS scene; returns {'ply': ..., 'cameras': ...}."""
    zpath = fetch(REPO_GS, index[scene]["gs"], token)
    out = config.work_dir() / "gs" / scene
    if not out.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    return locate_gs_files(out)


def locate_gs_files(root: Path) -> dict[str, Path]:
    """Pick point_cloud.ply from the highest iteration_* folder (else the largest .ply) and cameras.json."""
    root = Path(root)
    plys = list(root.rglob("*.ply"))
    if not plys:
        raise FileNotFoundError(f"no .ply under {root}")

    def it(p: Path) -> int:
        for part in reversed(p.parts):
            if part.startswith("iteration_") and part[10:].isdigit():
                return int(part[10:])
        return -1

    named = [p for p in plys if p.name == "point_cloud.ply"]
    ply = max(named, key=it) if named else max(plys, key=lambda p: p.stat().st_size)
    cams = sorted(root.rglob("cameras.json")) or sorted(root.rglob("transforms*.json"))
    if not cams:
        raise FileNotFoundError(f"no cameras.json / transforms.json under {root}")
    return {"ply": ply, "cameras": cams[0]}


# ----------------------------------------------------------------------------- 3DGS file formats
def load_gs_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Standard 3DGS PLY -> activated parameters: means, quats (wxyz, unit), scales, opacities, sh [N,K,3]."""
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], -1).astype(np.float32)[:, None, :]
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    K = 1 + len(rest_names) // 3
    sh_degree = int(round(np.sqrt(K))) - 1
    if (sh_degree + 1) ** 2 != K:
        raise ValueError(f"unexpected number of SH coefficients: {len(rest_names)}")
    if rest_names:
        rest = np.stack([v[n] for n in rest_names], -1).astype(np.float32)          # [N, 3*(K-1)] channel-major
        rest = rest.reshape(len(means), 3, K - 1).transpose(0, 2, 1)                  # [N, K-1, 3]
        sh = np.concatenate([dc, rest], 1)
    else:
        sh = dc
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1)).astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12
    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float32)))
    return {"means": means, "quats": quats, "scales": scales, "opacities": opac, "sh": sh, "sh_degree": sh_degree}


def load_cameras(path: str | Path) -> list[dict]:
    """3DGS cameras.json (OpenCV c2w: 'rotation' 3x3, 'position' 3) or nerfstudio transforms.json (OpenGL).

    Returns cameras sorted by image name, each {'name','width','height','c2w' (OpenCV 4x4),'K' (3x3)}.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    out = []
    if isinstance(data, list):                                      # 3DGS cameras.json
        for c in data:
            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(c["rotation"], float)
            c2w[:3, 3] = np.asarray(c["position"], float)
            W, H = int(c["width"]), int(c["height"])
            K = np.array([[c["fx"], 0, c.get("cx", W / 2)], [0, c["fy"], c.get("cy", H / 2)], [0, 0, 1]], float)
            out.append({"name": c.get("img_name", str(c.get("id"))), "width": W, "height": H, "c2w": c2w, "K": K})
    else:                                                           # nerfstudio transforms.json
        from .cameras import flip_yz
        for fr in data["frames"]:
            W, H = int(fr.get("w", data.get("w"))), int(fr.get("h", data.get("h")))
            fx, fy = fr.get("fl_x", data.get("fl_x")), fr.get("fl_y", data.get("fl_y"))
            cx, cy = fr.get("cx", data.get("cx", W / 2)), fr.get("cy", data.get("cy", H / 2))
            c2w = flip_yz(np.asarray(fr["transform_matrix"], float))
            out.append({"name": Path(fr["file_path"]).name, "width": W, "height": H, "c2w": c2w,
                        "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)})
    return sorted(out, key=lambda c: c["name"])


def scene_depth(gs: dict, cams: list[dict], max_points: int = 200_000, seed: int = 0) -> float:
    """Median depth of Gaussian centres in front of the training cameras (sets the image-motion scale)."""
    rng = np.random.default_rng(seed)
    pts = gs["means"]
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    depths = []
    for c in cams[:: max(1, len(cams) // 20)]:
        w2c = np.linalg.inv(c["c2w"])
        z = (pts @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
        z = z[z > 1e-3]
        if z.size:
            depths.append(np.median(z))
    return float(np.median(depths)) if depths else 1.0
