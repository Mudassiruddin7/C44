# C44: videos rendered from AI-generated 3D worlds vs. video forensics

Benchmark build and evaluation code (CVPR 2027 idea C44). The design is a 2×2 benchmark, **W2V**, of scene origin × image formation:

| Cell | Scene | How the frames were made | Source |
|---|---|---|---|
| RC | real | camera | DL3DV raw video (48 frames, 704×480) |
| RR | real | 3DGS render | DL3DV-GS-960P model of the same scene, along a path that reproduces RC's per-frame image motion |
| GR | generated | 3DGS render | FlashWorld, conditioned on RC's first frame, caption and camera path; rendered along that path, played at a speed calibrated so its image motion matches RC |
| GV | generated | video diffusion | Wan2.2-TI2V image-to-video (reference cell, **experimental**, not part of the notebook) |

Detectors compared: UnivFD, NPR, a multi-view reprojection-consistency detector (a re-implementation of the criterion GenWorld builds on), **CPS-1** (a one-class capture-physics score on 8 VGGT-motion-based features, `c44/cps.py`) and **CPS-2** (`c44/physics.py`): 23 image-formation cues of a physical camera measured from the pixels alone (optical-flow motion, no depth model), in a supervised and a one-class form. The proposed detector is **CPS-2 + CLIP**: CPS-2 fused (equal-weight z-scores, normalised with out-of-fold scores) with a probe on frozen CLIP ViT-L/14 embeddings trained on the dev scenes (`c44/learned.py`, `c44/embed.py`). The generic detectors are reported both zero-shot (released weights) and as probes retrained on the same dev scenes, which is the fair comparison. The CPS-2 cues that a renderer lacks by construction: auto-exposure / white-balance dynamics (motion-compensated frame-to-frame gain jitter), temporal vs. spatial noise statistics, noise whiteness and chroma noise, noise-free patches, power-spectrum shape, edge acutance, and blur-motion coupling.

**Protocol.** Scenes are split once by a hash into dev (70%) and held-out test (30%); no design decision uses the test scenes. Models train on dev scenes only; AUCs are on test scenes with scene-level bootstrap CIs. The "aug" variant of CPS-2 trains on recompressed and attacked fakes and is reported under three defender assumptions: the attack was **seen**; it is a **new combination** of components seen in other attacks; its **components are all unseen** (no training attack shares blur / noise / exposure jitter with it). **RAT** (randomised-attacker training) adds fakes processed by the same blur / noise / exposure operators at random strengths (`attacks.nuisance`, kinds `nz-*`); it never sees a calibrated attack. **SYM** applies the operators to real clips too (labelled real). Because RAT may only learn "processed = fake", the analysis also reports the false-alarm rate on real clips that went through the same operators (`cps2_operators`), next to the detection rate of the full attacker at the same threshold: that is the price of the defence. Besides the held-out split, every setting is also evaluated with a pre-specified 5-fold scene-grouped cross-validation over *all* scenes (nothing tuned), which triples the sample.

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
| 5 | `robust` | recompression (CRF 23/28/35) and camera-aware attacks, all 150 scenes, in parallel over CPU cores: calibrated blur, noise matched per clip to its real pair, and exposure / white-balance jitter calibrated to the real clip's measured level; plus 2,250 randomised-operator clips (`nz-*`: GR, RR and RC; training sets and the false-alarm test) | 90 min on 75 scenes serially (v15); parallel, 150 scenes + `nz-*`: estimate ~1.5-2 h |
| 6 | `control` | GR/RR re-rendered at 3× and area-downscaled (resolution/resampling control), 100 scenes | 60 min for 40 scenes (v15); estimate ~2.5 h for 100 |
| 7 | `physics` (+ `embed`) | CPS-2 features of every clip, parallel over CPU cores; then CLIP ViT-L/14 and NPR-network embeddings of every clip (GPU, resumable) | 47 min for 2.5k clips (v15); ~70 min for ~3.5k; embeddings ~20-30 min |
| 8 | `analyse` | final tables in `results/analysis/` (`summary.md`, CSVs, and `tables.tex` for the paper; see `c44/analysis.py`) and the results zip | ~10 min |

Every GPU stage runs in a fresh subprocess (`python -m c44 <stage>`), so each is also usable from a terminal and is resumable (finished clips are skipped, errored ones retried). molab limits: sessions last 12 h, notebooks stop after 90 idle minutes, and files are not kept, so use the HF repo for long runs.

```
python -m c44 setup | pull | rc --n 150 | captions | gr | rr | detect | features | cps | evaluate
python -m c44 robust --n 150 | control --n 100 | physics | analyse | report
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
  cps.py            CPS-1: first-generation capture-physics features and one-class score
  physics.py        CPS-2: 23 pixel-only camera cues, supervised and one-class detectors
  attacks.py        motion blur, sensor-noise and exposure-jitter attacks
  detectors/        UnivFD, NPR, reprojection consistency, VLM judge
  pipeline.py       build and evaluation stages (resumable)
  embed.py          CLIP / NPR backbone embeddings of every clip (frozen), for the retrained probes
  learned.py        probe + physics/CLIP fusion classifiers
  analysis.py       final tables (held-out scenes): detectors, CPS-2, cue families, transfer, attacks, control
  metrics.py        AUC, bootstrap CIs
  manifest.py, storage.py   clip bookkeeping; optional Hugging Face persistence
  setup_env.py, molab.py, cli.py   environment setup, notebook stages, command line
tests/              38 CPU tests with fake GPU components (no GPU, network or data needed)
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
