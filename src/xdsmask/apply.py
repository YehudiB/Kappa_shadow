from __future__ import annotations

from pathlib import Path

import numpy as np

from .hkl import write_filter_hkl
from .voxelize import frame_indices, geometry, pixel_indices


def masked_flags(records, masks, header, block=1, flip_x=False, flip_y=False, z_offset_deg=0.0):
    _, _, sf = geometry(header)
    nf, nyb, nxb = masks.shape
    f = frame_indices(records, header, nf, z_offset_deg)
    bx, by = pixel_indices(records, nxb, nyb, block, flip_x, flip_y)
    inside = (f >= 0) & (f < nf) & (bx >= 0) & (bx < nxb) & (by >= 0) & (by < nyb)
    masked = np.ones(records.shape[0], dtype=bool)
    idx = np.nonzero(inside)[0]
    masked[idx] = ~masks[f[idx], by[idx], bx[idx]]
    return masked


def save_masks(path, masks, block=1):
    path = Path(path)
    with path.open("wb") as fh:
        np.savez(fh, masks=np.asarray(masks, dtype=bool), block=np.int64(block))


def load_masks(path):
    with np.load(path) as data:
        return data["masks"], int(data["block"]) if "block" in data else 1


def apply_masks(hkl, masks, block=1, flip_x=False, flip_y=False, z_offset_deg=0.0, filter_out=None, out=None):
    masked = masked_flags(hkl.records, masks, hkl.header, block=block, flip_x=flip_x, flip_y=flip_y,
                          z_offset_deg=z_offset_deg)
    if filter_out is not None:
        write_filter_hkl(filter_out, hkl.records, masked)
    if out is not None:
        hkl.write(out, records=hkl.records[~masked])
    return masked
