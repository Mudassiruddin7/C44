"""molab environment setup for the GPU stack (RTX PRO 6000 Blackwell = compute capability 12.0).

Order of operations (each step logs and can be re-run):
  report()             what is installed (GPU, torch/CUDA, nvcc, gcc, disk, RAM)
  ensure_nvcc()        CUDA >= 12.8 compiler for gsplat's JIT build: existing -> pip wheels -> user-space toolkit
  install_gsplat()     gsplat from PyPI + a rasterisation smoke test (triggers the CUDA build, several minutes)
  fetch_repo()         pinned upstream code as GitHub archives (no git needed): FlashWorld, NPR
  install_flashworld() FlashWorld python deps without touching torch, then its checkpoint
  install_vggt()
FlashWorld pins torch 2.6, which cannot run on Blackwell; we keep molab's torch (>= 2.7, CUDA >= 12.8).
"""

from __future__ import annotations

import io
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from . import config

CUDA_RUNFILE = ("https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/"
                "cuda_12.8.1_570.124.06_linux.run")


def run(cmd: list[str], env: dict | None = None, check: bool = True, tail: int = 40) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env={**os.environ, **(env or {})}, capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().splitlines()
    if out:
        print("\n".join(out[-tail:]), flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p


def pip_install(*args: str, env: dict | None = None) -> None:
    """Install into *this* interpreter (molab notebooks use uv-managed environments)."""
    if shutil.which("uv"):
        run(["uv", "pip", "install", "--python", sys.executable, *args], env=env)
    else:
        run([sys.executable, "-m", "pip", "install", *args], env=env)


# ----------------------------------------------------------------------------- report
def _version(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:                       # noqa: BLE001 - report anything
        return f"missing ({type(e).__name__})"


def find_nvcc() -> tuple[str | None, tuple[int, int] | None]:
    cands = [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", str(Path.home() / ".c44" / "cuda"),
             str(Path.home() / ".c44" / "cuda-12.8")]
    paths = [shutil.which("nvcc")] + [str(Path(c) / "bin" / "nvcc") for c in cands if c]
    for p in paths:
        if p and Path(p).exists():
            out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout
            m = re.search(r"release (\d+)\.(\d+)", out)
            if m:
                return p, (int(m.group(1)), int(m.group(2)))
    return None, None


def report() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": _version("torch"), "torchvision": _version("torchvision"), "transformers": _version("transformers"),
            "diffusers": _version("diffusers"), "gsplat": _version("gsplat"), "numpy": _version("numpy"),
            "cv2": _version("cv2"), "av": _version("av")}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
            info["arch_list"] = torch.cuda.get_arch_list()
    except Exception as e:                       # noqa: BLE001
        info["torch_error"] = repr(e)
    nvcc, ver = find_nvcc()
    info["nvcc"] = f"{nvcc} ({ver[0]}.{ver[1]})" if nvcc else "not found"
    info["gcc"] = (subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines() or ["?"])[0] \
        if shutil.which("gcc") else "not found"
    du = shutil.disk_usage(config.work_dir())
    info["work_dir"] = str(config.work_dir())
    info["disk_free_gb"] = round(du.free / 2**30, 1)
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        info["cpus"] = os.cpu_count()
    except (ValueError, AttributeError, OSError):
        pass
    info["hf_token"] = "set" if config.hf_token() else "missing (add HF_TOKEN in the Secrets panel)"
    info["hf_repo"] = config.hf_repo() or "unset (set C44_HF_REPO to persist outputs)"
    return info


# ----------------------------------------------------------------------------- CUDA compiler
def gpu_arch() -> str:
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def _cuda_home_from_wheels(dst: Path) -> Path:
    """Assemble a CUDA_HOME from NVIDIA's pip wheels (nvcc, runtime headers/libs, CCCL headers)."""
    import importlib
    roots = {}
    for name in ("nvidia.cuda_nvcc", "nvidia.cuda_runtime", "nvidia.cuda_cccl"):
        importlib.invalidate_caches()
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"{name} not importable after install")
        roots[name.split(".")[1]] = Path(list(spec.submodule_search_locations)[0])
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "include").mkdir(parents=True)
    (dst / "lib64").mkdir()
    (dst / "bin").symlink_to(roots["cuda_nvcc"] / "bin", target_is_directory=True)
    if (roots["cuda_nvcc"] / "nvvm").exists():
        (dst / "nvvm").symlink_to(roots["cuda_nvcc"] / "nvvm", target_is_directory=True)
    for key in ("cuda_nvcc", "cuda_runtime", "cuda_cccl"):
        inc = roots[key] / "include"
        if inc.exists():
            for item in inc.iterdir():
                link = dst / "include" / item.name
                if not link.exists():
                    link.symlink_to(item, target_is_directory=item.is_dir())
    lib = roots["cuda_runtime"] / "lib"
    for item in lib.iterdir():
        (dst / "lib64" / item.name).symlink_to(item)
    so = sorted((dst / "lib64").glob("libcudart.so.*"))
    if so and not (dst / "lib64" / "libcudart.so").exists():
        (dst / "lib64" / "libcudart.so").symlink_to(so[0])
    return dst


def _activate_cuda_home(home: Path) -> None:
    os.environ["CUDA_HOME"] = str(home)
    os.environ["PATH"] = f"{home / 'bin'}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{home / 'lib64'}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def ensure_nvcc(min_version: tuple[int, int] = (12, 8), allow_runfile: bool = True) -> str:
    nvcc, ver = find_nvcc()
    if nvcc and ver >= min_version:
        _activate_cuda_home(Path(nvcc).resolve().parent.parent)
        print(f"nvcc {ver} found at {nvcc}")
        return os.environ["CUDA_HOME"]
    print(f"nvcc >= {min_version} not found (have {ver}); trying NVIDIA pip wheels")
    try:
        v = f"{min_version[0]}.{min_version[1]}.*"
        pip_install(f"nvidia-cuda-nvcc-cu12=={v}", f"nvidia-cuda-runtime-cu12=={v}", f"nvidia-cuda-cccl-cu12=={v}")
        home = _cuda_home_from_wheels(Path.home() / ".c44" / "cuda")
        _activate_cuda_home(home)
        nvcc, ver = find_nvcc()
        if nvcc and ver >= min_version:
            print(f"using pip CUDA toolkit {ver} at {home}")
            return str(home)
    except Exception as e:                       # noqa: BLE001 - fall through to the runfile
        print("pip toolkit route failed:", repr(e))
    if not allow_runfile:
        raise RuntimeError("no CUDA compiler; re-run with allow_runfile=True")
    home = Path.home() / ".c44" / "cuda-12.8"
    if not (home / "bin" / "nvcc").exists():
        runfile = config.work_dir() / "cuda_12.8.1.run"
        print("downloading the CUDA 12.8.1 toolkit (5.4 GB) - several minutes")
        urllib.request.urlretrieve(CUDA_RUNFILE, runfile)
        run(["sh", str(runfile), "--silent", "--toolkit", f"--toolkitpath={home}", "--no-man-page", "--override",
             f"--tmpdir={config.work_dir()}"])
        runfile.unlink(missing_ok=True)
    _activate_cuda_home(home)
    nvcc, ver = find_nvcc()
    if not nvcc:
        raise RuntimeError("CUDA toolkit install failed")
    return str(home)


# ----------------------------------------------------------------------------- gsplat
def _cccl_include_dirs() -> list[str]:
    """Directories that contain <nv/target> (CUDA 13 moved CCCL headers into include/cccl)."""
    import glob
    import site
    roots = [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", *site.getsitepackages(), sys.prefix]
    found = []
    for r in roots:
        if not r:
            continue
        for hit in glob.glob(os.path.join(r, "**", "nv", "target"), recursive=True):
            d = os.path.dirname(os.path.dirname(hit))
            if d not in found:
                found.append(d)
    return found


def install_base() -> None:
    """Small pure-python deps the pipeline needs (never touches torch)."""
    pip_install("av", "imageio-ffmpeg", "plyfile", "scipy", "scikit-learn", "pandas", "huggingface_hub", "einops", "safetensors")


def install_gsplat() -> None:
    env = {"TORCH_CUDA_ARCH_LIST": gpu_arch(), "MAX_JOBS": str(max(2, (os.cpu_count() or 4)))}
    inc = _cccl_include_dirs()
    if inc:
        print("CCCL headers:", inc)
        for var in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
            env[var] = ":".join(inc + [os.environ.get(var, "")]).rstrip(":")
    os.environ.update(env)
    pip_install("ninja", "jaxtyping", "rich", env=env)
    pip_install("--no-deps", f"gsplat=={config.GSPLAT_VERSION}", env=env)   # --no-deps: keep molab's torch
    smoke_test_gsplat()


def smoke_test_gsplat() -> float:
    """Render 10k random Gaussians (first call JIT-compiles gsplat's CUDA kernels). Returns mean alpha."""
    import numpy as np
    import torch
    from . import gs_render
    rng = np.random.default_rng(0)
    N = 10_000
    gs = {"means": rng.normal(0, 1, (N, 3)) + [0, 0, 6], "quats": rng.normal(size=(N, 4)),
          "scales": np.full((N, 3), 0.05), "opacities": np.full(N, 0.8), "sh": rng.normal(size=(N, 1, 3)) * 0.3,
          "sh_degree": 0}
    gs["quats"] /= np.linalg.norm(gs["quats"], axis=1, keepdims=True)
    K = np.array([[300.0, 0, 176], [0, 300.0, 120], [0, 0, 1]])
    frames, qc = gs_render.render(gs_render.gs_to_torch(gs), np.eye(4)[None], K, 352, 240)
    torch.cuda.synchronize()
    print("gsplat OK:", frames.shape, "mean coverage", float(qc["coverage"].mean()))
    return float(qc["coverage"].mean())


# ----------------------------------------------------------------------------- upstream code
def fetch_repo(repo_url: str, commit: str, dst: Path) -> Path:
    """Download a pinned GitHub commit as a zip archive (no git required); returns the extracted folder."""
    if dst.exists() and any(dst.iterdir()):
        return dst
    owner_repo = repo_url.removeprefix("https://github.com/").removesuffix(".git")
    url = f"https://codeload.github.com/{owner_repo}/zip/{commit}"
    print("downloading", url)
    data = urllib.request.urlopen(url, timeout=600).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(dst.parent)
    shutil.move(str(dst.parent / top), str(dst))
    return dst


def code_dir() -> Path:
    p = config.work_dir() / "code"
    p.mkdir(parents=True, exist_ok=True)
    return p


def install_flashworld() -> Path:
    repo = fetch_repo(config.FLASHWORLD_REPO, config.FLASHWORLD_COMMIT, code_dir() / "FlashWorld")
    pip_install("omegaconf", "jaxtyping", "rich", "einops", "moviepy==1.0.3", "av", "plyfile", "ftfy", "accelerate",
                "nanobind", "gradio", "fastapi", "uvicorn", "imageio", "imageio-ffmpeg",
                "transformers>=4.57", config.FLASHWORLD_DIFFUSERS)
    from huggingface_hub import hf_hub_download
    hf_hub_download("imlixinyang/FlashWorld", "model.ckpt")
    return repo


def install_vggt() -> None:
    pip_install("--no-deps", config.VGGT_PIP)   # keep molab's torch; vggt only needs torch/numpy/pillow/hf_hub


def install_npr() -> Path:
    return fetch_repo(config.NPR_REPO, config.NPR_COMMIT, code_dir() / "NPR")
