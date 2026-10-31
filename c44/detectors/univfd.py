"""UnivFD (Ojha et al., CVPR 2023): linear probe on CLIP ViT-L/14 image embeddings, frame-averaged for video.

Uses the authors' released head (pretrained_weights/fc_weights.pth, pinned commit) on the HF port of the same
OpenAI CLIP ViT-L/14 weights; image_embeds equals CLIP's encode_image output used by the original code.
"""

from __future__ import annotations

import numpy as np

from .. import config
from .base import Detector, sample_frames


class UnivFD(Detector):
    name = "univfd"

    def __init__(self, device: str = "cuda", n_frames: int = 16):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        self.torch, self.device, self.n = torch, device, n_frames
        mid = "openai/clip-vit-large-patch14"
        self.proc = CLIPImageProcessor.from_pretrained(mid)
        self.model = CLIPVisionModelWithProjection.from_pretrained(mid, dtype=torch.float16).to(device).eval()
        sd = torch.hub.load_state_dict_from_url(config.UNIVFD_FC_URL, map_location="cpu", file_name="univfd_fc.pth")
        self.fc = torch.nn.Linear(768, 1)
        self.fc.load_state_dict(sd)
        self.fc.to(device).eval()

    def score(self, frames: np.ndarray, geom=None) -> float:
        torch = self.torch
        x = self.proc(images=list(sample_frames(frames, self.n)), return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            emb = self.model(pixel_values=x.to(self.device, torch.float16)).image_embeds.float()
            return float(torch.sigmoid(self.fc(emb)).mean())
