"""Export a recorded run as compact JSON for the browser replay viewer.

Per-frame arrays are packed as little-endian binary and base64-encoded:
  pos     uint16, positions quantised to the box: (frames, N, dim)
  labels  uint16, cluster label per particle:     (frames, N)
  bonds   uint16, bound pairs (i, j), all frames concatenated; `bond_offsets`
          gives each frame's start index into the pair list.
"""
from __future__ import annotations

import base64
import json
import math

import numpy as np

from .sim import Simulation


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<u2").tobytes()).decode()


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return _clean(v.item())
    return v


def export_viewer(sim: Simulation, path: str, label: str, rules: list[str],
                  notes: str = "") -> None:
    if not sim.frames:
        raise ValueError("run the Simulation with record=True to export a replay")
    s, box = sim.state, sim.box.size
    pos = np.stack([f[0] for f in sim.frames])
    q = np.clip(np.round(pos / box * 65535), 0, 65535)
    labels = np.stack([f[2] for f in sim.frames])
    bonds = [f[1] for f in sim.frames]
    offsets = np.cumsum([0] + [len(b) for b in bonds]).tolist()
    flat = np.concatenate(bonds) if offsets[-1] else np.zeros((0, 2), int)

    keys = list(sim.rows[0])
    series = {k: [_clean(r[k]) if r[k] != "" else None for r in sim.rows] for k in keys}
    out = {
        "label": label,
        "notes": notes,
        "meta": {
            "dim": s.dim, "n": s.n, "box": box, "dt": sim.integ.dt,
            "log_every": sim.log_every, "r_bond": sim.r_bond, "rules": rules,
            "species": [{"name": sp.name, "mass": sp.mass, "charge": sp.charge}
                        for sp in s.species],
            "kind": s.kind.tolist(),
        },
        "frames": {"count": len(sim.frames), "pos": _b64(q), "labels": _b64(labels),
                   "bonds": _b64(flat), "bond_offsets": offsets},
        "series": series,
        "anomalies": sim.monitor.events,
    }
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
