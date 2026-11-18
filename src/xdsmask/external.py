from __future__ import annotations

import numpy as np

from .voxelize import frame_count, geometry


def load_external_masks(path, header, records, key="mask", flip_x=False, flip_y=False,
                        invert=True, deg_per_entry=None):
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path}: no array named {key!r} (available: {sorted(data.files)})")
        arr = np.asarray(data[key])
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D array (z, y, x), got shape {arr.shape}")
    nz, h, w = arr.shape
    nx, ny, sf = geometry(header)
    if (h, w) == (nx, ny) and (h, w) != (ny, nx):
        arr = arr.transpose(0, 2, 1)
    if arr.shape[1:] != (ny, nx):
        raise ValueError(
            f"per-frame mask shape {arr.shape[1:]} does not match detector (ny, nx)=({ny}, {nx})"
        )
    if flip_x:
        arr = arr[:, :, ::-1]
    if flip_y:
        arr = arr[:, ::-1, :]
    keep = np.ascontiguousarray(~arr if invert else arr, dtype=bool)
    osc = float(header.get("oscillation_range", "0") or 0.0)
    if osc <= 0:
        raise ValueError("OSCILLATION_RANGE missing or zero in header; needed to map degree-indexed masks to frames")
    sa = float(header.get("starting_angle", "0") or 0.0)
    dpe = float(deg_per_entry) if deg_per_entry else 360.0 / nz
    nf, sf = frame_count(header, records)
    ang = sa + np.arange(nf) * osc
    iz = np.round(ang / dpe).astype(np.int64) % nz
    return keep[iz], 1
