"""Persist work across 12-hour molab sessions in a private Hugging Face dataset repo.

molab only keeps uploaded/source files below 1 GB, so clips, metadata and score tables are pushed to
HF in batched commits and pulled back on the next session. Everything works without a repo (local only).
"""

from __future__ import annotations

from pathlib import Path

from . import config


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
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(l)) for l, p in self._pending if l.exists()]
        if ops:
            HfApi(token=self.token).create_commit(self.repo_id, repo_type="dataset", operations=ops,
                                                  commit_message=message)
        n = len(ops)
        self._pending.clear()
        return n

    def pull(self, patterns: list[str] | None = None) -> Path:
        """Download repo contents (optionally filtered) into the work dir; returns the work dir."""
        if not self.enabled:
            return config.work_dir()
        from huggingface_hub import snapshot_download
        snapshot_download(self.repo_id, repo_type="dataset", token=self.token,
                          local_dir=str(config.work_dir()), allow_patterns=patterns)
        return config.work_dir()
