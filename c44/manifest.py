"""Append-only JSONL manifest of benchmark clips (one record per clip; last record per id wins)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Clip:
    clip_id: str
    cell: str                 # RC / RR / GR / GV
    scene: str                # DL3DV scene hash (shared by all clips derived from one scene)
    path: str                 # path relative to the work dir, e.g. clips/RC/<id>.mp4
    n_frames: int
    fps: float
    width: int
    height: int
    generator: str = ""       # camera | dl3dv-3dgs | flashworld | wan2.2-ti2v ...
    parent: str = ""          # RC clip this one was derived from (GR/GV/RR pairing)
    status: str = "ok"        # ok | qc_fail
    meta: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)


class Manifest:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Clip]:
        clips: dict[str, Clip] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    clips[d["clip_id"]] = Clip(**d)
        return clips

    def add(self, clip: Clip) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(clip), default=_json_default) + "\n")

    def ids(self, cell: str | None = None, status: str | None = "ok") -> set[str]:
        return {c.clip_id for c in self.load().values()
                if (cell is None or c.cell == cell) and (status is None or c.status == status)}

    def by_cell(self, cell: str, status: str | None = "ok") -> list[Clip]:
        return sorted((c for c in self.load().values()
                       if c.cell == cell and (status is None or c.status == status)), key=lambda c: c.clip_id)

    def to_frame(self):
        import pandas as pd
        rows = [{**{k: v for k, v in asdict(c).items() if k != "meta"}, **{f"meta.{k}": v for k, v in c.meta.items()
                if not isinstance(v, (list, dict))}} for c in self.load().values()]
        return pd.DataFrame(rows)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
