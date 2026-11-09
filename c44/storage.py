"""Persist work across 12-hour molab sessions in a private Hugging Face dataset repo.

molab only keeps uploaded/source files below 1 GB, so clips, metadata and score tables are pushed to
HF in batched commits and pulled back on the next session. Everything works without a repo (local only).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from . import config

# The Hub allows 128 commits per hour per repo (free plan). A progress push every 25 clips used up that budget in ~50 minutes and every
# later push failed, each one blocking the main thread for ~35 s of useless retries. So: progress pushes are spaced out and capped per hour,
# the files of a skipped push simply stay staged for the next one, a 429 backs the pushes off for the time the Hub asks (capped, so a freed
# slot is noticed within minutes), and only the end of a stage (force=True) is allowed to wait.
PUSH_MIN_GAP = float(os.environ.get("C44_PUSH_MIN_GAP", "150"))        # s between two progress pushes
PUSH_PER_HOUR = int(os.environ.get("C44_PUSH_PER_HOUR", "50"))         # progress pushes per rolling hour (the Hub's cap is 128)
PUSH_BACKOFF_CAP = float(os.environ.get("C44_PUSH_BACKOFF_CAP", "300"))    # s: longest pause after one 429 (slots free up one by one)
PUSH_MAX_WAIT = float(os.environ.get("C44_PUSH_MAX_WAIT", "900"))      # s: longest a stage-end push may sleep for the Hub
_PUSH = {"last": 0.0, "next_ok": 0.0, "times": []}                     # shared by every Store of this process


def _reset_push_state() -> None:
    _PUSH.update(last=0.0, next_ok=0.0, times=[])


def backing_off() -> bool:
    """True while the Hub told us to slow down (best-effort pushes such as the log should then stay quiet)."""
    return time.time() < _PUSH["next_ok"]


def retry_after(e: BaseException) -> float | None:
    """Seconds a Hub rate-limit error (HTTP 429) asks us to wait, or None when `e` is some other error.
    The Hub says e.g. 'You can retry this action in about 1 hour' / 'in 44 minutes'."""
    text = str(e)
    resp = getattr(e, "response", None)
    if getattr(resp, "status_code", None) != 429 and "429" not in text and "rate limit" not in text.lower():
        return None
    try:
        hdr = float((getattr(resp, "headers", None) or {}).get("Retry-After"))
        if hdr > 0:
            return hdr
    except (TypeError, ValueError):
        pass
    m = re.search(r"retry[^.]*?\bin (?:about )?(\d+|an?)\s*(second|minute|hour)", text, re.I)
    if m:
        n = 1 if m.group(1).lower().startswith("a") else int(m.group(1))
        return n * {"second": 1, "minute": 60, "hour": 3600}[m.group(2).lower()]
    return 600.0


class Store:
    def __init__(self, repo_id: str | None = None, token: str | None = None):
        self.repo_id = repo_id or config.hf_repo()
        self.token = token or config.hf_token()
        self.enabled = bool(self.repo_id and self.token)
        self._pending: list[tuple[Path, str]] = []

    def ensure_repo(self) -> None:
        if not self.enabled:
            return
        from huggingface_hub import HfApi
        HfApi(token=self.token).create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)

    def stage(self, local: str | Path, path_in_repo: str | None = None) -> None:
        """Queue a file; path_in_repo defaults to its path relative to the work dir."""
        local = Path(local)
        if path_in_repo is None:
            path_in_repo = local.resolve().relative_to(config.work_dir().resolve()).as_posix()
        self._pending.append((local, path_in_repo))

    def flush(self, message: str = "c44: add outputs") -> int:
        """Upload all staged files in one commit. Returns the number of files pushed."""
        if not self.enabled or not self._pending:
            n = len(self._pending)
            self._pending.clear()
            return 0 if not self.enabled else n
        from huggingface_hub import CommitOperationAdd, HfApi
        latest = {p: l for l, p in self._pending if l.exists()}              # a retried push may have queued a path twice: last one wins
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(l)) for p, l in latest.items()]
        if ops:
            HfApi(token=self.token).create_commit(self.repo_id, repo_type="dataset", operations=ops,
                                                  commit_message=message)
        n = len(ops)
        self._pending.clear()
        return n

    def push(self, message: str = "c44: add outputs", force: bool = False, log=print) -> str:
        """flush() under the Hub's commit limit. NEVER raises; staged files are kept whenever nothing is sent.
        Returns 'pushed', 'nothing', 'throttled' (too soon / hourly budget used), 'backoff' (the Hub said 429) or 'failed'.
        Progress pushes (force=False) are skipped when too frequent; force=True (end of a stage) ignores the gap and the budget and
        waits out a 429 if that takes at most PUSH_MAX_WAIT seconds."""
        if not self.enabled:
            self._pending.clear()                                           # local-only run: nothing is ever sent
            return "nothing"
        if not self._pending:
            return "nothing"
        st, now = _PUSH, time.time()
        if not force:
            st["times"] = [t for t in st["times"] if now - t < 3600]
            if now < st["next_ok"]:
                return "backoff"
            if now - st["last"] < PUSH_MIN_GAP or len(st["times"]) >= PUSH_PER_HOUR:
                return "throttled"
        elif now < st["next_ok"]:
            wait = st["next_ok"] - now
            if wait > PUSH_MAX_WAIT:
                log(f"[push] the Hub asked for a pause of {wait / 60:.0f} min: {len(self._pending)} files stay queued (safe on local disk)")
                return "backoff"
            log(f"[push] waiting {wait:.0f} s for the Hub's rate limit before the final push of this stage")
            time.sleep(wait)
        for attempt in (1, 2):
            try:
                self.flush(message)
                st["last"] = time.time()
                st["times"].append(st["last"])
                st["next_ok"] = 0.0
                return "pushed"
            except Exception as e:                                          # noqa: BLE001 - hours of CPU work must not die on a Hub hiccup
                wait = retry_after(e)
                if wait is None:                                            # a network hiccup: one quick second try
                    if attempt == 1:
                        time.sleep(5)
                        continue
                    log(f"[push] HF commit failed ({e!r}); {len(self._pending)} files stay queued (safe on local disk)")
                    return "failed"
                pause = min(wait, PUSH_BACKOFF_CAP)
                st["next_ok"] = time.time() + pause
                log(f"[push] HF rate limit (429, asks for {wait / 60:.0f} min): pausing pushes for {pause / 60:.0f} min; "
                    f"{len(self._pending)} files stay queued (safe on local disk)")
                if force and attempt == 1 and pause <= PUSH_MAX_WAIT:
                    time.sleep(pause)
                    continue
                return "backoff"
        return "failed"

    def upload_file(self, local: str | Path, path_in_repo: str, retries: int = 3, max_wait: float = 1800.0) -> tuple[bool, str]:
        """Upload one (possibly large) file, e.g. the results zip, and verify that it is listed in the repo afterwards.
        Never raises: returns (ok, message), so a Hub problem cannot lose the local copy or abort a notebook.
        A rate-limit answer (429) is waited out (up to `max_wait` s in total, 0 = do not wait) and does not count as a failed attempt."""
        local = Path(local)
        if not self.enabled:
            return False, "no HF repo/token set: the file stays local only"
        if not local.exists():
            return False, f"{local} does not exist"
        last, attempt, waited = "", 0, 0.0
        while attempt < retries:
            attempt += 1
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=self.token)
                api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
                api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo, repo_id=self.repo_id, repo_type="dataset",
                                commit_message=f"c44: {path_in_repo}")
                _PUSH["times"].append(time.time())
                if path_in_repo in set(api.list_repo_files(self.repo_id, repo_type="dataset")):
                    return True, f"uploaded {local.name} ({local.stat().st_size / 2**20:.1f} MiB) to hf://datasets/{self.repo_id}/{path_in_repo}"
                last = "upload returned but the file is not listed in the repo"
            except Exception as e:                                          # noqa: BLE001
                last = repr(e)
                wait = retry_after(e)
                if wait is not None:
                    _PUSH["next_ok"] = time.time() + min(wait, PUSH_BACKOFF_CAP)
                    nap = min(wait, PUSH_BACKOFF_CAP) + 5
                    if waited + nap <= max_wait:
                        waited += nap
                        attempt -= 1                                        # waiting for a free slot is not a failed attempt
                        print(f"[upload] HF rate limit: waiting {nap:.0f} s for a free slot ({waited / 60:.0f} of at most {max_wait / 60:.0f} min)", flush=True)
                        time.sleep(nap)
                        continue
            if attempt < retries:
                time.sleep(5 * attempt)
        return False, f"upload failed after {retries} attempts: {last}"

    def pull(self, patterns: list[str] | None = None) -> Path:
        """Download repo contents (optionally filtered) into the work dir; returns the work dir."""
        if not self.enabled:
            return config.work_dir()
        from huggingface_hub import snapshot_download
        snapshot_download(self.repo_id, repo_type="dataset", token=self.token,
                          local_dir=str(config.work_dir()), allow_patterns=patterns)
        return config.work_dir()
