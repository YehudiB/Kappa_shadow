from __future__ import annotations

from pathlib import Path

import numpy as np

from .hkl import write_filter_hkl
from .voxelize import frame_indices, geometry, pixel_indices


def masked_flags(records, masks, header, block=1, flip_x=False, flip_y=False, z_offset_deg=0.0,
                 invert=False, deg_per_entry=None):
    _, _, sf = geometry(header)
    nf, nyb, nxb = masks.shape
    if deg_per_entry is not None:
        osc = float(header.get("oscillation_range", "0") or 0.0)
        if osc <= 0:
            raise ValueError("deg_per_entry requires OSCILLATION_RANGE in the header")
        sa = float(header.get("starting_angle", "0") or 0.0)
        idt = np.int32 if nf < 2 ** 31 else np.int64
        ang = sa + (records["zcal"] - sf) * osc
        f = np.round(ang / deg_per_entry).astype(idt) % nf
    else:
        f = frame_indices(records, header, nf, z_offset_deg)
    bx, by = pixel_indices(records, nxb, nyb, block, flip_x, flip_y)
    inside = (f >= 0) & (f < nf) & (bx >= 0) & (bx < nxb) & (by >= 0) & (by < nyb)
    masked = np.ones(records.shape[0], dtype=bool)
    idx = np.nonzero(inside)[0]
    val = masks[f[idx], by[idx], bx[idx]]
    masked[idx] = val if invert else ~val
    return masked


def save_masks(path, masks, block=1, **extra):
    path = Path(path)
    with path.open("wb") as fh:
        np.savez(fh, masks=np.asarray(masks, dtype=bool), block=np.int64(block), **extra)


def load_masks(path):
    with np.load(path) as data:
        return data["masks"], int(data["block"]) if "block" in data else 1


def load_masks_meta(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def apply_masks(hkl, masks, block=1, flip_x=False, flip_y=False, z_offset_deg=0.0,
                invert=False, deg_per_entry=None, filter_out=None, out=None):
    masked = masked_flags(hkl.records, masks, hkl.header, block=block, flip_x=flip_x, flip_y=flip_y,
                          z_offset_deg=z_offset_deg, invert=invert, deg_per_entry=deg_per_entry)
    if filter_out is not None:
        write_filter_hkl(filter_out, hkl.records, masked)
    if out is not None:
        hkl.write(out, records=hkl.records[~masked])
    return masked
