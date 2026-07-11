"""Cross-terminal run progress (JSON file + watch command).

Long LLM runs update a small JSON status file on every completed trial so a
second terminal can `watch-progress` without attaching to the runner process.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import PATHS

DEFAULT_PROGRESS_FILE = PATHS.output_dir / ".run_progress.json"


@dataclass
class ProgressReporter:
    """Write incremental run status to `path` (atomic replace)."""

    path: Path
    desc: str = "trials"
    total: int = 0
    pid: int = field(default_factory=os.getpid)
    started_at: float = field(default_factory=time.time)

    done: int = 0
    in_flight: int = 0
    cached: int = 0
    selected: int = 0
    last_domain: str = ""
    last_model: str = ""

    def set_in_flight(self, n: int) -> None:
        self.in_flight = n
        self._flush()

    def tick(
        self,
        *,
        cached: bool = False,
        selected: bool = False,
        domain: str = "",
        model: str = "",
    ) -> None:
        self.done += 1
        if cached:
            self.cached += 1
        if selected:
            self.selected += 1
        if domain:
            self.last_domain = domain
        if model:
            self.last_model = model
        self._flush()

    def finish(self, status: str = "done") -> None:
        self._flush(status=status)

    def _flush(self, status: str = "running") -> None:
        now = time.time()
        elapsed = max(now - self.started_at, 1e-6)
        payload = {
            "status": status,
            "pid": self.pid,
            "desc": self.desc,
            "total": self.total,
            "done": self.done,
            "in_flight": self.in_flight,
            "cached": self.cached,
            "selected": self.selected,
            "pct": round(100.0 * self.done / self.total, 2) if self.total else 0.0,
            "rate_per_sec": round(self.done / elapsed, 2),
            "eta_sec": int((self.total - self.done) / max(self.done / elapsed, 1e-9))
            if self.total > self.done
            else 0,
            "last_domain": self.last_domain,
            "last_model": self.last_model,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started_at)),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "elapsed_sec": int(elapsed),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def load_progress(path: Optional[Path] = None) -> Optional[dict]:
    p = path or DEFAULT_PROGRESS_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_progress_line(data: dict) -> str:
    eta = data.get("eta_sec", 0)
    eta_s = f"{eta // 3600}h{(eta % 3600) // 60}m" if eta >= 3600 else f"{eta // 60}m{eta % 60}s"
    return (
        f"[{data.get('status', '?')}] {data.get('desc', '')} "
        f"{data.get('done', 0)}/{data.get('total', 0)} "
        f"({data.get('pct', 0):.1f}%) "
        f"in_flight={data.get('in_flight', 0)} "
        f"rate={data.get('rate_per_sec', 0):.2f}/s "
        f"eta~{eta_s} | "
        f"sel={data.get('selected', 0)} cache={data.get('cached', 0)} | "
        f"{data.get('last_domain', '')} {data.get('last_model', '')}"
    )
