from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .voxelize import DEFAULT_VOXEL_BUDGET, geometry, voxel_stats


def local_contrast(mean, kernel=31):
    kernel = int(kernel)
    if kernel % 2 == 0:
        kernel += 1
    fallback = float(np.nanmedian(mean)) if np.isfinite(mean).any() else 0.0
    filled = np.nan_to_num(mean, nan=fallback)
    trend = ndimage.median_filter(filled, size=(1, kernel, kernel), mode="nearest")
    score = mean - trend
    score[~np.isfinite(mean)] = np.nan
    return score


def low_voxels(score, count, k=5.0, min_count=1):
    low = np.zeros(score.shape, dtype=bool)
    for f in range(score.shape[0]):
        s = score[f]
        usable = (count[f] >= min_count) & np.isfinite(s)
        v = s[usable]
        if v.size == 0:
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        std = float(np.std(v))
        spread = mad if mad > 0 else (std if std > 0 else 1.0)
        low[f] = usable & (s < med - k * spread)
    return low


def label_components(low):
    structure = np.ones((3, 3, 3), dtype=bool)
    return ndimage.label(low, structure=structure)


def component_trajectories(labels, n, min_voxels=10, min_span=3):
    trajs = {}
    for lab in range(1, n + 1):
        vox = labels == lab
        if int(vox.sum()) < min_voxels:
            continue
        frames = np.unique(np.nonzero(vox)[0])
        if frames.size < min_span:
            continue
        traj = {}
        for f in frames:
            ys, xs = np.nonzero(vox[f])
            traj[int(f)] = (float(xs.mean()) + 1.0, float(ys.mean()) + 1.0)
        trajs[lab] = traj
    return trajs


def select_shadow(trajs, ny, nx, orgx=None, orgy=None, entry_frac=0.8, beamstop_radius=50.0, slope_tol=1e-9):
    best = None
    for lab, traj in trajs.items():
        zs = sorted(traj)
        xs = np.array([traj[z][0] for z in zs])
        ys = np.array([traj[z][1] for z in zs])
        slope_x = float(np.polyfit(zs, xs, 1)[0]) if len(zs) > 1 else 0.0
        entry_y = float(ys[0])
        cx, cy = float(xs.mean()), float(ys.mean())
        near_beamstop = (
            orgx is not None and orgy is not None
            and float(np.hypot(cx - orgx, cy - orgy)) < beamstop_radius
        )
        static = abs(slope_x) < slope_tol
        if entry_y > entry_frac * ny and slope_x < 0.0 and not (near_beamstop and static):
            if best is None or len(traj) > best["span"]:
                best = {"label": lab, "span": len(traj), "slope_x": slope_x, "entry_y": entry_y}
    return best


def _disk(radius):
    r = max(int(radius), 1)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= r * r


def tube_mask(labels, label, shape, radius=3, band=15.0):
    nf, ny, nx = shape
    vox = labels == label
    frames = np.unique(np.nonzero(vox)[0])
    if frames.size:
        zs = frames.astype(float)
        xs = np.array([np.nonzero(vox[f])[1].mean() + 1.0 for f in frames])
        ys = np.array([np.nonzero(vox[f])[0].mean() + 1.0 for f in frames])
        allz = np.arange(nf, dtype=float)
        if zs.size >= 2:
            xf = np.polyval(np.polyfit(zs, xs, 1), allz)
            yf = np.polyval(np.polyfit(zs, ys, 1), allz)
        else:
            xf = np.full(nf, xs[0])
            yf = np.full(nf, ys[0])
    else:
        xf = np.full(nf, (nx + 1) / 2.0)
        yf = np.full(nf, (ny + 1) / 2.0)
    yy, xx = np.indices((ny, nx), dtype=np.int32)
    masks = np.ones(shape, dtype=bool)
    struct = _disk(radius)
    for f in range(nf):
        sel = vox[f]
        dist = np.hypot((xx + 1.0 - xf[f]).astype(np.float32), (yy + 1.0 - yf[f]).astype(np.float32))
        if sel.any():
            sel = ndimage.binary_dilation(sel, structure=struct) & (dist <= band)
        else:
            sel = dist <= max(int(radius), 1)
        masks[f] = ~sel
    return masks, {"x_of_z": xf.tolist(), "y_of_z": yf.tolist()}


@dataclass
class DetectionParams:
    value: str = "maxc"
    kernel: int = 31
    k: float = 5.0
    min_count: int = 1
    min_voxels: int = 10
    min_span: int = 3
    entry_frac: float = 0.8
    beamstop_radius: float = 50.0
    radius: int = 3
    band: float = 15.0
    block: int = 0
    voxel_budget: int = DEFAULT_VOXEL_BUDGET


def detect_shadow(records, header, params=None):
    params = params or DetectionParams()
    nx, ny, sf = geometry(header)
    grid = voxel_stats(records, header, value=params.value, block=params.block, budget=params.voxel_budget)
    block = grid.block
    kernel_b = max(3, int(round(params.kernel / block)))
    if kernel_b % 2 == 0:
        kernel_b += 1
    score = local_contrast(grid.mean, kernel=kernel_b)
    low = low_voxels(score, grid.count, k=params.k, min_count=params.min_count)
    labels, n = label_components(low)
    trajs = component_trajectories(labels, n, min_voxels=params.min_voxels, min_span=params.min_span)
    orgx = float(header["orgx"]) if "orgx" in header else None
    orgy = float(header["orgy"]) if "orgy" in header else None
    best = select_shadow(
        trajs, grid.shape[1], grid.shape[2],
        orgx=orgx, orgy=orgy,
        entry_frac=params.entry_frac,
        beamstop_radius=params.beamstop_radius / block,
    )
    base_report = {
        "n_components": int(n),
        "shape": list(grid.shape),
        "pixel_shape": [ny, nx],
        "block": block,
        "starting_frame": sf,
    }
    if best is None:
        masks = np.ones(grid.shape, dtype=bool)
        return masks, {"selected": None, **base_report}
    radius_b = max(1, int(round(params.radius / block)))
    band_b = max(1.0, params.band / block)
    masks, traj = tube_mask(labels, best["label"], grid.shape, radius=radius_b, band=band_b)
    report = {
        "selected": best,
        **base_report,
        "trajectory_px": {
            "x_of_z": (np.asarray(traj["x_of_z"]) * block).tolist(),
            "y_of_z": (np.asarray(traj["y_of_z"]) * block).tolist(),
        },
        "masked_voxels_per_frame": (~masks).sum(axis=(1, 2)).tolist(),
    }
    return masks, report
