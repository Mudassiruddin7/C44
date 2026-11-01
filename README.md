# C44: videos rendered from AI-generated 3D worlds vs. video forensics

Benchmark build and evaluation code (CVPR 2027 idea C44). The design is a 2×2 benchmark, **W2V**, of scene origin × image formation:

| Cell | Scene | How the frames were made | Source |
|---|---|---|---|
| RC | real | camera | DL3DV raw video (48 frames, 704×480) |
| RR | real | 3DGS render | DL3DV-GS-960P model of the same scene, along a path that reproduces RC's per-frame image motion |
| GR | generated | 3DGS render | FlashWorld, conditioned on RC's first frame, caption and camera path; rendered along that path, played at a speed calibrated so its image motion matches RC |
| GV | generated | video diffusion | Wan2.2-TI2V image-to-video (reference cell, **experimental**, not part of the notebook) |

Detectors compared: UnivFD, NPR, a multi-view reprojection-consistency detector (a re-implementation of the criterion GenWorld builds on), a one-class **capture-physics score** (CPS, `c44/cps.py`) and a supervised classifier on the same 8 physics features. Evaluation is by AUC with scene-level bootstrap CIs; every learned model is cross-validated with folds split **by scene**.

## Run it (one notebook)

1. On Hugging Face, accept the terms of `DL3DV/DL3DV-ALL-video` and `DL3DV/DL3DV-GS-960P`, and create a token that can read them (and write to your results repo, if you use one).
2. Upload **`c44_molab.py`** to [molab](https://molab.marimo.io) and attach the GPU (built and tested on an RTX PRO 6000 Blackwell, 96 GB).
3. Fill in the form: the token (password field), the number of scenes (6 = pilot, 150 = full) and, optionally, a private Hugging Face dataset repo (`user/name`, or its URL) that keeps your work if molab restarts. Press **Start the whole pipeline**.
4. Wait for `ALL DONE`, then use the download button for `c44_results.zip`.

The notebook downloads this repository's code from GitHub on every run, so fixing the code never requires re-uploading the notebook (only changes to the notebook file itself do). One cell runs everything:

| Step | Command | What it does | Measured on 150 scenes |
|---|---|---|---|
| 1–2 | `check`, `setup` | environment report; VGGT, NPR, FlashWorld, a CUDA 13.0 toolkit matched to torch, gsplat compiled for sm_120 | ~10 min per session |
| 3 | smoke test + `rc`, `captions`, `gr`, `rr` | one scene through every component first (fails fast with all errors listed), then the full build; resumes from the HF repo | RC 133 min, captions 5, GR 38, RR 165 |
| 4 | `evaluate` | UnivFD, NPR, consistency, CPS, report | 17 min |
| 5 | `robust` | recompression (CRF 23/28/35) and camera-aware attacks (calibrated blur, noise matched per clip to its real pair) | ~1–2 h |
| 6 | `control` | GR/RR re-rendered at 3× and area-downscaled (resolution/resampling control) on a scene subset | 60 min |
| 7 | `analyse` | final tables in `results/analysis/` (see `c44/analysis.py`) and the results zip | 3 min |

Every GPU stage runs in a fresh subprocess (`python -m c44 <stage>`), so each is also usable from a terminal and is resumable (finished clips are skipped, errored ones retried). molab limits: sessions last 12 h, notebooks stop after 90 idle minutes, and files are not kept, so use the HF repo for long runs.

```
python -m c44 setup | pull | rc --n 150 | captions | gr | rr | detect | features | cps | evaluate
python -m c44 robust --n 75 | control --n 40 | analyse | report
python -m c44 gv --limit 150        # experimental video-diffusion cell
```

## Repository layout

```
c44_molab.py        the notebook (upload this to molab)
c44/
  config.py         benchmark spec (704×480, 48 frames), pinned upstream commits, .env / HF settings
  dl3dv.py          DL3DV index, downloads, 3DGS PLY / cameras.json parsing
  video.py          decode / encode (H.264) / resize
  cameras.py        pose conversions and FlashWorld camera conventions
  geometry.py       VGGT wrapper, per-frame image motion, reprojection error
  gs_render.py      gsplat rasteriser wrapper
  flashworld_gen.py FlashWorld wrapper;  gen_video.py  Wan wrapper (experimental)
  vlm.py            Qwen2.5-VL captions (FlashWorld prompts)
  cps.py            capture-physics features and the one-class score
  attacks.py        motion blur and sensor-noise attacks
  detectors/        UnivFD, NPR, reprojection consistency, VLM judge
  pipeline.py       build and evaluation stages (resumable)
  analysis.py       final tables and supervised physics score
  metrics.py        AUC, bootstrap CIs
  manifest.py, storage.py   clip bookkeeping; optional Hugging Face persistence
  setup_env.py, molab.py, cli.py   environment setup, notebook stages, command line
tests/              33 CPU tests with fake GPU components (no GPU, network or data needed)
docs/molab-notes.md what had to be worked around on molab / Blackwell, and why
```

## Tests

```
uv venv && uv pip install -e ".[dev]" torch && pytest
```
They cover camera math (checked against FlashWorld's own code), video I/O, metrics, CPS on a simulated camera vs. renderer, geometry and consistency on a synthetic scene, 3DGS parsing, attack calibration, the supersampled control, and the whole pipeline end to end with fake GPU components. Nothing in the tests needs a GPU; the GPU path is exercised by the notebook's smoke test.

## Limitations

- GenWorld's detector code is not public and DeMamba publishes no weights, so the consistency criterion is re-implemented (`detectors/consistency.py`) and DeMamba is not included.
- The GV cell is experimental and not in the notebook.
- The resolution control uses 3× supersampling (the real clips are shrunk about 5× from ~4K) on a subset of scenes.
- Licences: DL3DV (non-commercial research terms on HF), FlashWorld and gsplat (Apache-2.0), VGGT (check the model licence before commercial use).

## Security

Never commit a Hugging Face token. `.env` is git-ignored; the notebook takes the token through a password field and only places it in the process environment.
