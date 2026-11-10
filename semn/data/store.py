"""
semn/data/store.py
==========================
Dataset storage behind a small protocol; Parquet + Polars by default (decision D2).

Layout: ``<root>/<name>/year=YYYY/part.parquet`` for tables with a ``date``
column, ``<root>/<name>/part.parquet`` otherwise.  A write replaces the whole
dataset: it is written to a temporary folder first and swapped in, so a failed
write never leaves a half-written dataset behind.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

import polars as pl


class DataStore(Protocol):
    def write(self, name: str, df: pl.DataFrame) -> None: ...
    def scan(self, name: str) -> pl.LazyFrame: ...


class ParquetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, name: str, df: pl.DataFrame) -> None:
        target = self.root / name
        tmp, old = target.with_name(name + ".tmp"), target.with_name(name + ".old")
        shutil.rmtree(tmp, ignore_errors=True)
        if "date" in df.columns and not df.is_empty():
            for (year,), part in df.group_by(pl.col("date").dt.year(), maintain_order=True):
                (tmp / f"year={year}").mkdir(parents=True)
                part.write_parquet(tmp / f"year={year}" / "part.parquet")
        else:
            tmp.mkdir(parents=True)
            df.write_parquet(tmp / "part.parquet")
        if target.exists():
            target.rename(old)
        tmp.rename(target)
        shutil.rmtree(old, ignore_errors=True)

    def scan(self, name: str) -> pl.LazyFrame:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(f"dataset {name!r} not found under {self.root}")
        # hive_partitioning=False: `year` is a storage detail, not a column.
        return pl.scan_parquet(path / "**" / "*.parquet", hive_partitioning=False)

    def files(self, name: str) -> list[Path]:
        """Parquet files of a dataset, sorted (for manifests)."""
        return sorted((self.root / name).rglob("*.parquet"))
