from __future__ import annotations

import numpy as np

from .voxelize import frame_count, geometry


def load_external_masks(path, header, key="mask", flip_x=False, flip_y=False, deg_per_entry=None):
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path}: no array named {key!r} (available: {sorted(data.files)})")
        arr = np.asarray(data[key])
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D array (z, y, x), got shape {arr.shape}")
    nz, h, w = arr.shape
    nx, ny, _ = geometry(header)
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
    dpe = float(deg_per_entry) if deg_per_entry else 360.0 / nz
    return arr, dpe


def z_entries(header, records, nz, deg_per_entry):
    osc = float(header.get("oscillation_range", "0") or 0.0)
    if osc <= 0:
        raise ValueError("OSCILLATION_RANGE missing or zero in header; needed to map degree-indexed masks to frames")
    sa = float(header.get("starting_angle", "0") or 0.0)
    nf, sf = frame_count(header, records)
    ang = sa + np.arange(nf) * osc
    return np.round(ang / deg_per_entry).astype(np.int64) % nz, nf


def block_reduce_any(arr, block):
    nz, ny, nx = arr.shape
    b = max(int(block), 1)
    nyb = int(np.ceil(ny / b))
    nxb = int(np.ceil(nx / b))
    out = np.zeros((nz, nyb, nxb), dtype=bool)
    for iy in range(nyb):
        ysl = slice(iy * b, min((iy + 1) * b, ny))
        for ix in range(nxb):
            xsl = slice(ix * b, min((ix + 1) * b, nx))
            out[:, iy, ix] = arr[:, ysl, xsl].any(axis=(1, 2))
    return out
