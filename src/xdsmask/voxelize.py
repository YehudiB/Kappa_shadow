from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelGrid:
    shape: tuple
    starting_frame: int
    count: np.ndarray
    mean: np.ndarray
    frac_unobserved: np.ndarray


def geometry(header):
    nx = int(float(header["nx"]))
    ny = int(float(header["ny"]))
    starting_frame = int(float(header.get("starting_frame", "1")))
    return nx, ny, starting_frame


def frame_count(header, records):
    _, _, sf = geometry(header)
    zmax = float(np.max(records["zcal"])) if records.size else float(sf)
    return max(int(round(zmax)) - sf + 1, 1), sf


def voxel_stats(records, header, value="maxc"):
    nx, ny, sf = geometry(header)
    nf, sf = frame_count(header, records)
    f = np.round(records["zcal"]).astype(np.int64) - sf
    ix = np.round(records["xcal"]).astype(np.int64)
    iy = np.round(records["ycal"]).astype(np.int64)
    valid = (f >= 0) & (f < nf) & (ix >= 1) & (ix <= nx) & (iy >= 1) & (iy <= ny)
    vid = ((f * ny) + (iy - 1)) * nx + (ix - 1)
    nvox = nf * ny * nx
    count = np.bincount(vid[valid], minlength=nvox).astype(np.int64).reshape(nf, ny, nx)
    vals = np.log1p(np.clip(records[value].astype(np.float64), 0.0, None))
    sums = np.bincount(vid[valid], weights=vals[valid], minlength=nvox).reshape(nf, ny, nx)
    mean = np.full((nf, ny, nx), np.nan)
    np.divide(sums, count, out=mean, where=count > 0)
    unobs = (records["xobs"] == 0).astype(np.float64)
    usum = np.bincount(vid[valid], weights=unobs[valid], minlength=nvox).reshape(nf, ny, nx)
    frac_unobserved = np.zeros((nf, ny, nx))
    np.divide(usum, count, out=frac_unobserved, where=count > 0)
    return VoxelGrid(shape=(nf, ny, nx), starting_frame=sf, count=count, mean=mean, frac_unobserved=frac_unobserved)
