"""Lightweight experiment logging.

Two helpers so that documenting a run is nearly free and nothing is silently
overwritten:

- ``run_dir(name)`` -> a fresh, timestamped output directory
  ``runs/<YYYYMMDD-HHMMSS>_<name>/`` for that run's artifacts (non-clobbering).
- ``log_run(name, params, metrics, outputs, notes)`` -> append one JSON record
  (timestamp, params, metrics, output paths, notes) to ``runs/index.jsonl``.

The machine-readable ``index.jsonl`` is the automatic record; the curated
narrative (what each run *meant*, which results are real vs artifact) lives in
``EXPERIMENTS.md`` and is maintained by hand.

Convention: new and edited scripts call ``run_dir`` for their outputs and
``log_run`` at the end. Example::

    from diffrt.diamond.explog import run_dir, log_run
    out = run_dir("fire_differential")
    ...save out/"fire.png"...
    log_run("fire_differential", params={...}, metrics={...},
            outputs=[out / "fire.png"], notes="...")
"""

from __future__ import annotations

import datetime
import json
import pathlib


def _runs_dir() -> pathlib.Path:
    # src/diffrt/diamond/explog.py -> parents[3] == repo root
    return pathlib.Path(__file__).resolve().parents[3] / "runs"


def _clean(o):
    """Make params/metrics JSON-serializable (handles numpy scalars/arrays)."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, pathlib.Path):
        return str(o)
    return o


def run_dir(name: str, ts: str | None = None) -> pathlib.Path:
    """Create and return ``runs/<timestamp>_<name>/`` for this run's artifacts."""
    ts = ts or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = _runs_dir() / f"{ts}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_run(name: str, params: dict, metrics: dict, outputs, notes: str = "") -> dict:
    """Append one JSON line to ``runs/index.jsonl`` and return the record."""
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "params": _clean(params),
        "metrics": _clean(metrics),
        "outputs": [str(o) for o in outputs],
        "notes": notes,
    }
    idx = _runs_dir()
    idx.mkdir(parents=True, exist_ok=True)
    with open(idx / "index.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


__all__ = ["run_dir", "log_run"]
