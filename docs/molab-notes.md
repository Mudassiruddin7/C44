# Notes: what had to be worked around (molab, RTX PRO 6000 Blackwell, torch 2.11 + CUDA 13)

Kept so a future run (or a reviewer) can see why the setup code looks the way it does.

## Environment
- **Fresh process per stage.** The notebook process imports `torch`/`transformers` before setup installs more packages; Python caches those imports and "is X installed" checks, so in-process stages fail with errors like "requires accelerate". Every GPU stage therefore runs as `python -m c44 <stage>`.
- **Do not let installs replace molab's torch.** FlashWorld's dependencies pull in a second torch (2.15) into the venv. `setup_env.remove_shadow_torch()` removes it. vggt and gsplat are installed with `--no-deps`.
- **gsplat JIT needs a matched CUDA toolkit.** molab ships nvcc 13.3 next to CUDA 13.0 headers; CCCL refuses that mix ("compiler and toolkit headers are incompatible") and the CUDA 13 layout puts `<nv/target>` under `include/cccl`. `setup_env` installs a private CUDA 13.0 nvcc + headers matching torch and points the build at them. A failed earlier build leaves a stale ninja state, so the JIT cache is cleared first.
- **gsplat 1.5.3 vs FlashWorld.** In `rasterization(packed=True, backgrounds=...)` gsplat 1.5.3 asserts a background vector, while FlashWorld passes `[C, D]`. `flashworld_gen.py` wraps FlashWorld's rasteriser with `packed=False` (same renders). The wrapper also composites backgrounds itself.
- **Qwen2.5-VL** is loaded with a plain `.to(device)` (no `accelerate` dependency).

## Data
- `DL3DV-GS-960P` holds both `3DGS/<subset>/<hash>.zip` (the Gaussian model) and `imgs_undist/<subset>/<hash>.zip` (images only). Indexing must use `3DGS/` only.
- The NPR weights were saved from `nn.DataParallel` (keys start with `module.`).
- 3DGS `cameras.json` stores the original resolution (~3850×2133) with fx/fy but no cx/cy.

## Benchmark design decisions
- **GR motion matching.** The real clip's camera path is in VGGT's arbitrary units and FlashWorld invents its own scene depth, so the same path gave 1.3–4.2× the real image motion. Rescaling translations moved render cameras off the generation path (holes, coverage 0.95). Instead the generated path is **played more slowly** (only its first fraction, resampled to 48 frames): poses stay on the path, coverage stays ≥ 0.98, and the per-clip speed is calibrated to match RC. Limitation: it can only slow a path down.
- **Attacks.** Blur exposure is calibrated once against the real clips' sharpness-vs-motion behaviour (the grid saturated at its upper end, 0.5). Noise is matched per clip to the paired real clip's noise level with *all* noise variances scaled together; scaling only the shot-noise gain left a read-noise/PRNU floor that was already louder than many real clips (117 of 288 clips pinned at the minimum gain, achieved noise 1.3–1.7× the real level), which is why the variants `noise`, `blur+noise`, `noisecal`, `blurcal+noisecal` are retired and ignored by the analysis.
- **Resolution control.** Real frames are shrunk from ~4K with an area filter, renders are produced at 704×480. The control re-renders GR/RR at 3× from the same worlds / paths / speeds and area-downscales them.
- **Calibration check.** `analysis.py` reports attacked-vs-real noise level ratios; use it before interpreting any attack result.
