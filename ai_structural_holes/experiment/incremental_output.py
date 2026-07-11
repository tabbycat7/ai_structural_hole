"""Thread-safe incremental CSV writes and periodic analysis refresh."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence


class IncrementalCsvWriter:
    """Append one row at a time to a CSV under a lock."""

    def __init__(
        self,
        path: Path,
        fieldnames: Optional[Sequence[str]] = None,
        *,
        truncate_on_init: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fieldnames: Optional[list[str]] = list(fieldnames) if fieldnames else None
        self._initialized = False
        if truncate_on_init and self.path.exists():
            self.path.unlink()
        elif not truncate_on_init and self.path.exists():
            # Resume: keep existing rows and append. Adopt the on-disk header as
            # the field order so appended rows stay column-aligned; extra keys on
            # new rows are dropped safely by DictWriter(extrasaction="ignore").
            header = self._read_header()
            if header:
                self._fieldnames = header
                self._initialized = True

    def _read_header(self) -> Optional[list[str]]:
        try:
            with open(self.path, encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                return next(reader, None)
        except OSError:
            return None

    @property
    def n_rows(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, encoding="utf-8-sig") as f:
            return max(sum(1 for _ in f) - 1, 0)

    def read_dataframe(self) -> "pd.DataFrame":
        import io

        import pandas as pd

        with self._lock:
            if not self._initialized or not self.path.exists():
                return pd.DataFrame()
            text = self.path.read_text(encoding="utf-8-sig")
        return pd.read_csv(io.StringIO(text))

    def append_row(self, row: dict) -> None:
        with self._lock:
            if self._fieldnames is None:
                self._fieldnames = list(row.keys())
            if not self._initialized:
                with self.path.open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=self._fieldnames, extrasaction="ignore"
                    )
                    writer.writeheader()
                self._initialized = True
            with self.path.open("a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self._fieldnames, extrasaction="ignore"
                )
                writer.writerow(row)


class PeriodicAnalysisRefresher:
    """Trigger a callback when enough trials complete or enough time passes."""

    def __init__(
        self,
        callback: Callable[[], None],
        *,
        refresh_every: int = 100,
        refresh_sec: float = 300.0,
    ):
        self.callback = callback
        self.refresh_every = refresh_every
        self.refresh_sec = refresh_sec
        self._count = 0
        self._last_refresh_count = 0
        self._last_refresh_time = time.time()
        self._lock = threading.Lock()

    def tick(self, n: int = 1) -> None:
        should_refresh = False
        with self._lock:
            self._count += n
            should_refresh = self._should_refresh_unlocked()
            if should_refresh:
                self._last_refresh_count = self._count
                self._last_refresh_time = time.time()
        if should_refresh:
            self.callback()

    def maybe_refresh(self, *, force: bool = False) -> None:
        should_refresh = False
        with self._lock:
            if force or self._should_refresh_unlocked():
                self._last_refresh_count = self._count
                self._last_refresh_time = time.time()
                should_refresh = True
        if should_refresh:
            self.callback()

    def _should_refresh_unlocked(self) -> bool:
        now = time.time()
        by_count = self._count - self._last_refresh_count >= self.refresh_every
        by_time = now - self._last_refresh_time >= self.refresh_sec
        return by_count or by_time
