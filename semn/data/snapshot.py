"""
semn/data/snapshot.py
==========================
Versioned, content-addressed snapshots with a JSON manifest (Phase 1).

A snapshot lives at ``<root>/<name>/<hash12>/`` and holds ``data.parquet`` plus
``manifest.json``: input file hashes, vendor and version, build parameters,
code revision and the data's content hash.  The same data always gets the same
hash and folder, so a snapshot is never silently overwritten with different data.
Later phases record the content hash in the experiment ledger.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def content_hash(df: pl.DataFrame) -> str:
    """SHA-256 of the data: schema plus every value in row order (sort before hashing if order is not fixed)."""
    h = hashlib.sha256(json.dumps({k: str(v) for k, v in df.schema.items()}).encode())
    h.update(df.write_csv().encode())
    return h.hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_snapshot(
    df: pl.DataFrame,
    root: str | Path,
    name: str,
    *,
    params: dict,
    inputs: list[str | Path],
    vendor: str,
    vendor_version: str,
) -> Path:
    """Write *df* and its manifest; return the snapshot folder.  Re-writing identical data is a no-op."""
    digest = content_hash(df)
    folder = Path(root) / name / digest[:12]
    if (folder / "manifest.json").exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    df.write_parquet(folder / "data.parquet")
    manifest = {
        "name": name,
        "content_hash": digest,
        "rows": df.height,
        "schema": {k: str(v) for k, v in df.schema.items()},
        "params": params,
        "inputs": {str(p): file_hash(p) for p in sorted(map(str, inputs))},
        "vendor": vendor,
        "vendor_version": vendor_version,
        "code_revision": _git_revision(),
        "polars_version": pl.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Manifest last: a folder without one is an incomplete write and gets redone.
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return folder


def load_snapshot(folder: str | Path) -> tuple[pl.DataFrame, dict]:
    """Read a snapshot and check its data still matches the manifest hash."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    df = pl.read_parquet(folder / "data.parquet")
    if content_hash(df) != manifest["content_hash"]:
        raise ValueError(f"snapshot {folder} does not match its manifest hash")
    return df, manifest


def _git_revision() -> str:
    """HEAD commit, with ``-dirty`` if tracked files have uncommitted changes."""
    try:
        here = Path(__file__).parent
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return rev + ("-dirty" if dirty else "")
