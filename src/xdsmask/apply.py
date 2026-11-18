from __future__ import annotations

from pathlib import Path

import numpy as np

from .hkl import write_filter_hkl
from .voxelize import geometry


def masked_flags(records, masks, header):
    nx, ny, sf = geometry(header)
    nf = masks.shape[0]
    f = np.round(records["zcal"]).astype(np.int64) - sf
    ix = np.round(records["xcal"]).astype(np.int64)
    iy = np.round(records["ycal"]).astype(np.int64)
    inside = (f >= 0) & (f < nf) & (ix >= 1) & (ix <= nx) & (iy >= 1) & (iy <= ny)
    masked = np.ones(records.shape[0], dtype=bool)
    idx = np.nonzero(inside)[0]
    masked[idx] = ~masks[f[idx], iy[idx] - 1, ix[idx] - 1]
    return masked


def save_masks(path, masks):
    path = Path(path)
    with path.open("wb") as fh:
        np.savez(fh, masks=np.asarray(masks, dtype=bool))


def load_masks(path):
    with np.load(path) as data:
        return data["masks"]


def apply_masks(hkl, masks, filter_out=None, out=None):
    masked = masked_flags(hkl.records, masks, hkl.header)
    if filter_out is not None:
        write_filter_hkl(filter_out, hkl.records, masked)
    if out is not None:
        hkl.write(out, records=hkl.records[~masked])
    return masked
