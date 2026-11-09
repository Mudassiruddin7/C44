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
    remote = f"logs/weak_job_{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.log"        # one file per run: a new run never overwrites a crash's log
    env["C44_PUSH_LOG"] = f"{log}|{remote}"
    cmd = cmd or [sys.executable, "-u", "-m", "c44.weak_job", "--n", str(n_scenes)]
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== weak job started {time.strftime('%Y-%m-%d %H:%M:%S')} (n_scenes={n_scenes}) =====\n")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(Path.cwd()), start_new_session=True)
    return write_state(state="running", pid=p.pid, step="starting", started=time.time(), n_scenes=n_scenes, error="", zip="", upload=[],
                       remote_log=remote)


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


def _read_first(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:                                                     # noqa: BLE001
        return ""


def resource_line() -> str:
    """One line: RAM, container limit, memory held by this job and all its workers, CPU, load, disk. Never raises."""
    parts = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"RAM {vm.used / 2**30:.1f}/{vm.total / 2**30:.0f} GiB ({vm.percent:.0f}%)")
        me = psutil.Process()
        kids = me.children(recursive=True)
        rss = []
        for q in [me, *kids]:
            try:
                rss.append(q.memory_info().rss)
            except Exception:                                             # noqa: BLE001 - a worker exited between listing and reading
                pass
        parts.append(f"job tree {len(kids)} child procs, {sum(rss) / 2**30:.1f} GiB RSS (largest {max(rss, default=0) / 2**30:.2f} GiB)")
        parts.append(f"cpu {psutil.cpu_percent(None):.0f}%")
    except Exception:                                                     # noqa: BLE001
        pass
    for cur, lim in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                     ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        c, m = _read_first(cur), _read_first(lim)
        if c.isdigit():
            parts.append(f"cgroup {int(c) / 2**30:.1f}/" + (f"{int(m) / 2**30:.0f}" if m.isdigit() and int(m) < 2 ** 50 else "unlimited") + " GiB")
            break
    try:
        parts.append(f"load {os.getloadavg()[0]:.1f}")
    except (OSError, AttributeError):
        pass
    free = config.disk_free_gb()
    if free is not None:
        parts.append(f"disk free {free:.0f} GiB")
    return " | ".join(parts) or "(no resource info available)"


def push_log(why: str = "") -> bool:
    """Upload the job log to the HF repo (never raises)."""
    try:
        from .storage import Store, backing_off
        local, _, remote = os.environ.get("C44_PUSH_LOG", "").partition("|")
        if not local or not remote or not Path(local).exists() or (backing_off() and why != "end"):
            return False
        ok, msg = Store().upload_file(local, remote, retries=1, max_wait=0)        # best effort: never wait for the Hub on behalf of a log
        if not ok:
            print(f"[log push] {msg}", flush=True)
        return ok
    except Exception as e:                                                # noqa: BLE001
        print(f"[log push] failed: {e!r}", flush=True)
        return False


def monitor(stop, interval: float = 30.0, push_every: float = 600.0, now=time.time) -> None:
    """Thread body: a resource line every `interval` s (so the log shows memory right up to a crash), a log push every `push_every` s."""
    last_push = now()
    while not stop.wait(interval):
        print(f"[mon {time.strftime('%H:%M:%S')}] {resource_line()}", flush=True)
        if now() - last_push >= push_every:
            push_log()
            last_push = now()


def banner() -> str:
    import platform
    return "\n".join([f"[job] python {platform.python_version()} on {platform.platform()}, pid {os.getpid()}, cpus {os.cpu_count()}",
                      f"[job] workers for features: C44_WEAK_WORKERS={os.environ.get('C44_WEAK_WORKERS', '6 (default)')}",
                      f"[job] start: {resource_line()}"])


def main(argv=None) -> int:
    import argparse
    import signal
    import threading
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args(argv)
    write_state(pid=os.getpid(), state="running", n_scenes=a.n)
    print(banner(), flush=True)

    def on_signal(signum, _frame):                                        # a shutdown leaves a trace on the Hub
        print(f"[job] received signal {signum} at {time.strftime('%H:%M:%S')} - {resource_line()}", flush=True)
        write_state(state="ended", error=f"killed by signal {signum}")
        push_log("signal")
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", signal.SIGTERM)):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop,), daemon=True).start()
    try:
        from . import molab
        st = run_steps(molab.run)
        write_state(upload=list(molab.LAST_UPLOAD))
        print(f"\nJOB {st.get('state', '?').upper()}", flush=True)
        return 0 if st.get("state") == "done" else 1
    except BaseException as e:                                            # noqa: BLE001 - log it, then re-raise
        print(f"[job] crashed: {e!r}", flush=True)
        write_state(state="failed", error=f"job crashed: {e!r}")
        raise
    finally:
        stop.set()
        print(f"[job] end {time.strftime('%H:%M:%S')}: {resource_line()}", flush=True)
        push_log("end")


if __name__ == "__main__":
    sys.exit(main())
