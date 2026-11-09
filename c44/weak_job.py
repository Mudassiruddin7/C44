"""A detached runner for the weakness pipeline, so that refreshing the notebook (or restarting its kernel) does not kill a multi-hour job.

The notebook starts `python -m c44.weak_job` in its own session (not a child of the kernel's process group), with all output going to
`<work>/weak_job.log` and the state in `<work>/weak_job.json`. Pressing the button again finds the live process and just follows its log
instead of starting a second one. If the whole container is recycled the job is gone, but every stage pushes its progress to the HF repo
(clips and features every 25-100 items), so the next run resumes from there.

    state file   {"state": "running" | "done" | "failed", "pid", "step", "started", "n_scenes", "zip", "upload", "error"}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config

STEPS = [("weak_attack", True), ("weak_features", True), ("weak_embed", False), ("weak_analyse", True)]     # (stage, fatal)


def paths() -> tuple[Path, Path]:
    w = config.work_dir()
    return w / "weak_job.log", w / "weak_job.json"


def read_state() -> dict:
    _, js = paths()
    try:
        return json.loads(js.read_text(encoding="utf-8"))
    except Exception:                                                     # noqa: BLE001 - missing or half-written file
        return {}


def write_state(**kw) -> dict:
    _, js = paths()
    st = {**read_state(), **kw}
    tmp = js.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, default=str), encoding="utf-8")
    tmp.replace(js)
    return st


def alive(pid: int | None) -> bool:
    """Is process `pid` running (a zombie counts as finished)?"""
    if not pid:
        return False
    try:
        stat = Path(f"/proc/{int(pid)}/stat")
        if stat.exists():
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def is_running() -> bool:
    st = read_state()
    return st.get("state") == "running" and alive(st.get("pid"))


def start(n_scenes: int = 150, cmd: list[str] | None = None) -> dict:
    """Start the detached job, or return the live one. `cmd` is for tests."""
    if is_running():
        return read_state()
    log, _ = paths()
    code_root = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (code_root, os.environ.get("PYTHONPATH", "")) if p)}
    cmd = cmd or [sys.executable, "-u", "-m", "c44.weak_job", "--n", str(n_scenes)]
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== weak job started {time.strftime('%Y-%m-%d %H:%M:%S')} (n_scenes={n_scenes}) =====\n")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(Path.cwd()), start_new_session=True)
    return write_state(state="running", pid=p.pid, step="starting", started=time.time(), n_scenes=n_scenes, error="", zip="", upload=[])


def follow(poll: float = 5.0, tail_lines: int = 40, out=print, max_wait: float | None = None) -> dict:
    """Print the job's log as it grows until the process ends; returns the final state ('done', 'failed', or 'ended' if it died silently)."""
    log, _ = paths()
    pos = 0
    if log.exists():                                                      # attaching: show the last lines, then follow
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        out("\n".join(lines[-tail_lines:]))
        pos = log.stat().st_size
    t0 = time.time()
    while True:
        running = is_running()
        if log.exists() and log.stat().st_size > pos:
            with open(log, "rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
            pos += len(chunk)
            out(chunk.decode("utf-8", errors="replace").rstrip("\n"))
        if not running:
            break
        if max_wait is not None and time.time() - t0 > max_wait:
            return {**read_state(), "state": "running"}
        time.sleep(poll)
    st = read_state()
    if st.get("state") == "running":                                      # the process is gone but never wrote done/failed: killed from outside
        st = write_state(state="ended", error="the job process ended without finishing (the session or container was probably recycled); "
                                              "press the button again - progress is on the HF repo")
    return st


def run_steps(runner, steps=STEPS, log=print) -> dict:
    """Run the stages in order; a non-fatal stage that fails is logged and skipped. `runner(stage, **kw)` is molab.run in production."""
    n = read_state().get("n_scenes") or 150
    for stage, fatal in steps:
        write_state(step=stage)
        log(f"\n===== {stage} =====")
        try:
            res = runner(stage, n_scenes=n) if stage == "weak_attack" else runner(stage)
            if stage == "weak_analyse":
                write_state(zip=str(res or ""))
        except Exception as e:                                            # noqa: BLE001
            log(f"!!! {stage} FAILED: {str(e)[-1800:]}")
            if fatal:
                return write_state(state="failed", error=f"{stage}: {str(e)[-1800:]}")
    return write_state(state="done", step="finished")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args(argv)
    write_state(pid=os.getpid(), state="running", n_scenes=a.n)
    from . import molab
    st = run_steps(molab.run)
    write_state(upload=list(molab.LAST_UPLOAD))
    print(f"\nJOB {st.get('state', '?').upper()}", flush=True)
    return 0 if st.get("state") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
