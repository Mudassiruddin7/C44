"""NPR (Tan et al., CVPR 2024), frame-averaged for video, with the authors' network code and NPR.pth.

Preprocessing follows the repo's test protocol for resized datasets: Resize((256, 256)) bilinear, no crop,
ImageNet normalisation; probability = sigmoid(model(x)).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .base import Detector, sample_frames


class NPR(Detector):
    name = "npr"

    def __init__(self, repo_dir: str | Path, device: str = "cuda", n_frames: int = 16):
        import torch
        from torchvision import transforms
        repo_dir = Path(repo_dir)
        spec = importlib.util.spec_from_file_location("npr_resnet", repo_dir / "networks" / "resnet.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.resnet50(num_classes=1)
        sd = torch.load(repo_dir / "NPR.pth", map_location="cpu")
        model.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd, strict=True)
        self.model = model.to(device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.torch, self.device, self.n = torch, device, n_frames

    def score(self, frames: np.ndarray, geom=None) -> float:
        from PIL import Image
        torch = self.torch
        x = torch.stack([self.tf(Image.fromarray(f)) for f in sample_frames(frames, self.n)]).to(self.device)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(x)).mean())
