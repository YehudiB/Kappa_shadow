from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_VOXEL_BUDGET = 2 ** 25


@dataclass
class VoxelGrid:
    shape: tuple
    starting_frame: int
    block: int
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


def choose_block(nf, ny, nx, budget=DEFAULT_VOXEL_BUDGET):
    block = 1
    while block < ny and block < nx and nf * int(np.ceil(ny / block)) * int(np.ceil(nx / block)) > budget:
        block *= 2
    return block


def voxel_stats(records, header, value="maxc", block=1, budget=DEFAULT_VOXEL_BUDGET):
    nx, ny, sf = geometry(header)
    nf, sf = frame_count(header, records)
    if block <= 0:
        block = choose_block(nf, ny, nx, budget)
    nxb = int(np.ceil(nx / block))
    nyb = int(np.ceil(ny / block))
    f = np.round(records["zcal"]).astype(np.int64) - sf
    bx = (np.round(records["xcal"]).astype(np.int64) - 1) // block
    by = (np.round(records["ycal"]).astype(np.int64) - 1) // block
    valid = (f >= 0) & (f < nf) & (bx >= 0) & (bx < nxb) & (by >= 0) & (by < nyb)
    vid = ((f * nyb) + by) * nxb + bx
    nvox = nf * nyb * nxb
    count = np.bincount(vid[valid], minlength=nvox).astype(np.int32).reshape(nf, nyb, nxb)
    vals = np.log1p(np.clip(records[value].astype(np.float32), 0.0, None))
    sums = np.bincount(vid[valid], weights=vals[valid], minlength=nvox).astype(np.float32).reshape(nf, nyb, nxb)
    mean = np.full((nf, nyb, nxb), np.nan, dtype=np.float32)
    np.divide(sums, count, out=mean, where=count > 0)
    unobs = (records["xobs"] == 0).astype(np.float32)
    usum = np.bincount(vid[valid], weights=unobs[valid], minlength=nvox).reshape(nf, nyb, nxb)
    frac_unobserved = np.zeros((nf, nyb, nxb), dtype=np.float32)
    np.divide(usum, count, out=frac_unobserved, where=count > 0)
    return VoxelGrid(shape=(nf, nyb, nxb), starting_frame=sf, block=block, count=count, mean=mean, frac_unobserved=frac_unobserved)
