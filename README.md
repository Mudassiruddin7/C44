# C44: videos rendered from AI-generated 3D worlds vs. video forensics

Benchmark build and evaluation code for CVPR 2027 idea C44. The design is a 2×2 benchmark (W2V) of scene origin × image formation:

| Cell | Scene | How the frames were made | Source |
|---|---|---|---|
| RC | real | camera | DL3DV raw video (48 frames) |
| RR | real | 3DGS render | DL3DV-GS-960P model of the same scene, along a path that reproduces RC's per-frame motion |
| GR | generated | 3DGS render | FlashWorld, conditioned on RC's first frame and RC's camera path, rendered along all 48 RC poses |
| GV | generated | video diffusion | Wan2.2-TI2V-5B image-to-video (reference cell, **experimental**) |

The proposed detector is the capture-physics score (CPS) in `c44/cps.py`. It is fitted on real footage only, and scored out-of-fold with folds split by scene.

## What to run

**One file: `c44_run.py`**, a marimo notebook.
1. Upload the contents of `c44_molab_upload.zip` to molab (the `c44/` folder and `c44_run.py`).
2. Attach the RTX PRO 6000 GPU.
3. Add `HF_TOKEN` and `C44_HF_REPO` in the Secrets panel.
4. Accept the terms of `DL3DV/DL3DV-ALL-video` and `DL3DV/DL3DV-GS-960P` on Hugging Face.
5. Press the three buttons in order.

| Step | What it does | Estimated time |
|---|---|---|
| 1 · Set up GPU stack | CUDA ≥12.8 compiler, gsplat compiled for Blackwell, VGGT, NPR, FlashWorld + model downloads (~30 GB) | 20–45 min first time; repeat every new session, because installs aren't persisted |
| 2 · Build clips, 6 scenes (pilot) | RC → captions → GR → RR | 30–45 min |
| 2 · Build clips, 150 scenes (full) | same, resumable | about 6–9 h, across one or two sessions |
| 3 · Detectors + CPS + report | UnivFD, NPR, multi-view consistency, CPS; AUC with 95% CIs | pilot 10–15 min, full 1–1.5 h |

These times are estimates; nothing has run on the GPU yet. The DL3DV video download speed and gsplat's first compile dominate. molab shuts down a notebook after 90 idle minutes and caps sessions at 12 h. Every step pushes to your Hugging Face repo, so after a restart run `python -m c44 pull` (or step 1 again) and press the step again to resume.

The command-line equivalent, for long runs in a terminal:
```
python -m c44 setup
python -m c44 rc --n 150
python -m c44 captions
python -m c44 gr
python -m c44 rr
python -m c44 detect
python -m c44 cps
python -m c44 report
```
Optional:
```
python -m c44 gv --limit 150        # video-diffusion reference cell (experimental)
```
For robustness and adaptive attacks, use the Python API: `pipeline.build_variants("crf28" | "blur" | "noise" | "blur+noise")`, then run `detect`, `cps` and `report` again.

## Status
- **Tested locally (CPU, 25 tests):**
  - camera math, checked against FlashWorld's own code;
  - video I/O;
  - metrics;
  - CPS on a simulated camera vs. renderer;
  - geometry and consistency on a synthetic scene;
  - 3DGS file parsing;
  - the whole pipeline with fake GPU components.

  Run the tests with: `uv venv && uv pip install -e .[dev] torch && pytest`.
- **Not yet run on the GPU:**
  - the gsplat build on Blackwell;
  - FlashWorld on torch ≥2.7 (it pins 2.6);
  - VGGT, UnivFD, NPR and Wan wrappers.

  Step 1 checks these first.
- **Missing baselines:** DeMamba publishes no weights; GenWorld's detector code is not public, so its multi-view-consistency criterion is re-implemented in `detectors/consistency.py`.
- **Licences:** DL3DV (non-commercial research terms on HF), ARKitScenes (Apple non-commercial), FlashWorld/gsplat (Apache-2.0), VGGT (check the model licence before any commercial use).
