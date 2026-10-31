"""GV reference cell: image-to-video diffusion from the RC first frame (Wan2.2-TI2V-5B via diffusers).

EXPERIMENTAL - the diffusers image-to-video entry point for this checkpoint is not documented on its model card
(only text-to-video with WanPipeline is). We try WanImageToVideoPipeline and require an `image` argument;
if neither works we fail loudly rather than silently producing text-only videos. Verify on molab first.
The first generated frame is dropped because I2V models typically reproduce the conditioning (real) image.
"""

from __future__ import annotations

import inspect

import numpy as np

from .video import crop_resize

NEGATIVE = ("blurry, low quality, distorted, watermark, text, subtitles, static image, jpeg artifacts, "
            "deformed, extra objects")


class WanI2V:
    def __init__(self, model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", device: str = "cuda"):
        import diffusers
        import torch
        from diffusers import AutoencoderKLWan
        self.torch, self.device = torch, device
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe_cls = getattr(diffusers, "WanImageToVideoPipeline", None)
        if pipe_cls is None:
            raise RuntimeError("this diffusers build has no WanImageToVideoPipeline")
        self.pipe = pipe_cls.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16).to(device)
        if "image" not in inspect.signature(self.pipe.__call__).parameters:
            raise RuntimeError(f"{pipe_cls.__name__} takes no image input")

    def __call__(self, image: np.ndarray, prompt: str, n_out: int, out_wh: tuple[int, int], seed: int = 0,
                 gen_wh: tuple[int, int] = (1024, 704), steps: int = 50, guidance: float = 5.0) -> np.ndarray:
        from PIL import Image
        torch = self.torch
        num_frames = 4 * ((n_out + 1 + 3) // 4) + 1                  # Wan needs 4k+1 frames; +1 for the dropped frame
        res = self.pipe(image=Image.fromarray(crop_resize(image, *gen_wh)), prompt=prompt, negative_prompt=NEGATIVE,
                        width=gen_wh[0], height=gen_wh[1], num_frames=num_frames, guidance_scale=guidance,
                        num_inference_steps=steps, generator=torch.Generator(self.device).manual_seed(seed))
        frames = res.frames[0]
        frames = np.stack([np.asarray(f) for f in frames])
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0, 1) * 255 + 0.5).astype(np.uint8)
        frames = frames[1:1 + n_out]
        if len(frames) < n_out:
            raise RuntimeError(f"generator returned {len(frames) + 1} frames, need {n_out + 1}")
        return np.stack([crop_resize(f, *out_wh) for f in frames])
