"""FlashWorld (ICLR 2026) wrapper: one 3DGS world per RC clip, rendered along that clip's camera path.

We call FlashWorld's own GenerationSystem.generate (app.py) with
  image   = the RC clip's first frame (704x480, so no cropping and no intrinsics rescaling is needed),
  cameras = 24 keyframes of the RC camera path (VGGT poses, OpenCV -> FlashWorld OpenGL cam11),
and render the returned Gaussians with c44.gs_render along all 48 RC poses, normalised exactly as FlashWorld
normalises its own render cameras (same ref_w2c and T_norm). The world therefore shares the RC clip's first
view and camera motion, and differs only in where the scene came from and how frames were formed.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import cameras, gs_render


class FlashWorld:
    def __init__(self, repo_dir: str | Path, ckpt_path: str | None = None, device: str = "cuda",
                 offload_t5: bool = False, offload_vae: bool = False, offload_transformer_during_vae: bool = False):
        import torch
        self.torch = torch
        repo_dir = str(Path(repo_dir).resolve())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        cwd = os.getcwd()
        os.chdir(repo_dir)                      # app.py imports models/, utils.py, quant.py from the repo root
        try:
            app = importlib.import_module("app")
        finally:
            os.chdir(cwd)
        # FlashWorld was written for an older gsplat; in gsplat 1.5.3 rasterization(packed=True, backgrounds=...) trips
        # an assertion (expects a vector, FlashWorld passes [C, D]). packed=False gives identical renders.
        import functools
        render_mod = importlib.import_module("models.render")
        render_mod.rasterization = functools.partial(render_mod.rasterization, packed=False)
        if ckpt_path is None:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download("imlixinyang/FlashWorld", "model.ckpt")
        self.system = app.GenerationSystem(ckpt_path=ckpt_path, device=torch.device(device), offload_t5=offload_t5,
                                           offload_vae=offload_vae,
                                           offload_transformer_during_vae=offload_transformer_during_vae)
        self.device = device

    def generate(self, image: np.ndarray, cam11: np.ndarray, text: str = "", image_index: int = 0, seed: int = 0):
        """image: uint8 [H,W,3]; cam11: FlashWorld cameras [N,11] with intrinsics normalised by this W,H.
        Returns (scene params tensor [G, C], ref_w2c 4x4, T_norm, seconds)."""
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        H, W = image.shape[:2]
        img = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1) / 255.0 * 2 - 1
        cams = torch.from_numpy(np.asarray(cam11, np.float32))
        t0 = time.time()
        scene, ref_w2c, T_norm = self.system.generate(cams, len(cam11), img, text, image_index, H, W, video_path=None)
        torch.cuda.synchronize()
        return scene, ref_w2c.reshape(4, 4).double().cpu().numpy(), float(T_norm.reshape(-1)[0]), time.time() - t0

    def render_path(self, scene, ref_w2c: np.ndarray, T_norm: float, cam11_path: np.ndarray, W: int, H: int,
                    background=(1.0, 1.0, 1.0)):
        cam_norm, _, _ = cameras.fw_normalize(cam11_path, ref_w2c=ref_w2c, T_norm=T_norm)
        c2w_cv, Ks = cameras.fw_to_render(cam_norm, W, H)
        return gs_render.render(gs_render.flashworld_params_to_gs(scene), c2w_cv, Ks, W, H, background)
