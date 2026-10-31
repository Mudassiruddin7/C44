"""One Gaussian renderer for both render cells (RR: DL3DV 3DGS, GR: FlashWorld worlds).

Identical rasteriser settings across RR and GR (gsplat 'classic' mode, same background, resolution, near/far)
are what make the RR-vs-GR comparison isolate *generation* from *rendering*.
"""

from __future__ import annotations

import numpy as np


def gs_to_torch(gs: dict, device: str = "cuda") -> dict:
    import torch
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    return {"means": t(gs["means"]), "quats": t(gs["quats"]), "scales": t(gs["scales"]),
            "opacities": t(gs["opacities"]), "colors": t(gs["sh"]), "sh_degree": int(gs["sh_degree"])}


def flashworld_params_to_gs(params) -> dict:
    """FlashWorld scene params [N, 3+1+3+4+3K] (activated, as passed to gsplat in models/render.py) -> dict."""
    import torch
    C = params.shape[-1]
    K = (C - 11) // 3
    deg = int(round(K ** 0.5)) - 1
    if (deg + 1) ** 2 != K or 11 + 3 * K != C:
        raise ValueError(f"unexpected FlashWorld parameter width {C}")
    xyz, op, sc, rot, feat = params.float().split([3, 1, 3, 4, 3 * K], dim=-1)
    return {"means": xyz.contiguous(), "opacities": op.squeeze(-1).contiguous(), "scales": sc.contiguous(),
            "quats": torch.nn.functional.normalize(rot, dim=-1).contiguous(),
            "colors": feat.reshape(-1, K, 3).contiguous(), "sh_degree": deg}


def render(gs: dict, c2w_cv: np.ndarray, K: np.ndarray, width: int, height: int,
           background=(1.0, 1.0, 1.0), near: float = 0.01, far: float = 1000.0, chunk: int = 8):
    """Render OpenCV c2w poses [T,4,4] with intrinsics [3,3] or [T,3,3].

    Returns (frames uint8 [T,H,W,3], qc dict with per-frame 'coverage' = mean alpha and 'holes' = alpha<0.5 fraction).
    """
    import torch
    from gsplat import rasterization
    dev = gs["means"].device
    c2w = torch.as_tensor(np.asarray(c2w_cv), dtype=torch.float32, device=dev)
    viewmats = torch.linalg.inv(c2w)
    T = len(viewmats)
    Ks = torch.as_tensor(np.broadcast_to(np.asarray(K, np.float32), (T, 3, 3)).copy(), device=dev)
    frames, cov, holes = [], [], []
    with torch.no_grad():
        for i in range(0, T, chunk):
            vm, kk = viewmats[i:i + chunk], Ks[i:i + chunk]
            bg = torch.tensor(background, dtype=torch.float32, device=dev).expand(len(vm), 3)
            rgb, alpha, _ = rasterization(gs["means"], gs["quats"], gs["scales"], gs["opacities"], gs["colors"],
                                          vm, kk, width, height, sh_degree=gs["sh_degree"], near_plane=near,
                                          far_plane=far, backgrounds=bg, render_mode="RGB", rasterize_mode="classic")
            frames.append((rgb.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu())
            a = alpha[..., 0]
            cov.append(a.mean(dim=(1, 2)).cpu())
            holes.append((a < 0.5).float().mean(dim=(1, 2)).cpu())
    return (torch.cat(frames).numpy(),
            {"coverage": torch.cat(cov).numpy(), "holes": torch.cat(holes).numpy()})
