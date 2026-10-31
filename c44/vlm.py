"""Vision-language model helper (captions for FlashWorld prompts; zero-shot real-vs-fake judge baseline)."""

from __future__ import annotations

import re

import numpy as np

CAPTION_PROMPT = ("Describe this place for a 3D scene generator in two or three sentences: the setting, the main "
                  "objects and their layout, materials and lighting. Do not mention the camera or the photo.")
JUDGE_PROMPT = ("These frames come from one short video. Is it real camera footage, or was it generated or "
                "rendered by a computer? Answer with a single integer from 0 to 100: the probability that it is "
                "NOT real camera footage.")


class VLM:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_id)
        try:
            m = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16)
        except TypeError:                          # older transformers
            m = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.model = m.to(device).eval()           # plain .to(): no `accelerate` needed
        self.model_id = model_id

    def ask(self, images: list[np.ndarray], prompt: str, max_new_tokens: int = 128) -> str:
        import torch
        from PIL import Image
        content = [{"type": "image", "image": Image.fromarray(im)} for im in images]
        content.append({"type": "text", "text": prompt})
        inputs = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def caption(self, image: np.ndarray) -> str:
        return self.ask([image], CAPTION_PROMPT)


def parse_probability(text: str) -> float:
    """First integer 0-100 in the reply, as a probability; NaN if none."""
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return float("nan")
    v = int(m.group(1))
    return v / 100.0 if 0 <= v <= 100 else float("nan")
