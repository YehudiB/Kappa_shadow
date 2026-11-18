import numpy as np

from xdsmask.apply import masked_flags
from xdsmask.detect import (DetectionParams, component_trajectories,
                            detect_shadow, label_components, local_contrast,
                            low_voxels, select_shadow, tube_mask)
from xdsmask.hkl import IntegrateHKL
from xdsmask.voxelize import choose_block

from conftest import grid_rows, rec, write_hkl


def _disk(cx, cy, r, ny, nx):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (xx + 1 - cx) ** 2 + (yy + 1 - cy) ** 2 <= r * r


def test_volume_pipeline_selects_moving_tube_over_static_beamstop():
    nf, ny, nx = 6, 40, 40
    mean = np.full((nf, ny, nx), 5.0)
    count = np.full((nf, ny, nx), 10)
    for f in range(nf):
        mean[f][_disk(20, 20, 3, ny, nx)] = 0.5
        mean[f][_disk(30 - 3 * f, 36, 2, ny, nx)] = 0.2
    score = local_contrast(mean, kernel=15)
    low = low_voxels(score, count, k=4.0)
    labels, n = label_components(low)
    trajs = component_trajectories(labels, n, min_voxels=5, min_span=3)
    best = select_shadow(trajs, ny, nx, orgx=20.0, orgy=20.0, entry_frac=0.8, beamstop_radius=10.0)
    assert best is not None
    assert best["slope_x"] < -2.0
    assert best["entry_y"] > 0.8 * ny
    masks, traj, span = tube_mask(labels, best["label"], (nf, ny, nx), radius=2, band=6.0)
    for f in range(nf):
        assert masks[f, 36 - 1, (30 - 3 * f) - 1] == False
        assert masks[f, 20 - 1, 20 - 1] == True
    assert len(traj["x_of_z"]) == nf
    assert span == (0, nf - 1)


def test_pre_dilation_connects_fast_moving_tube():
    from scipy import ndimage

    nf, ny, nx = 6, 40, 40
    mean = np.full((nf, ny, nx), 5.0)
    count = np.full((nf, ny, nx), 10)
    for f in range(nf):
        mean[f][_disk(30 - 3 * f, 36, 0, ny, nx)] = 0.2
    score = local_contrast(mean, kernel=15)
    low = low_voxels(score, count, k=4.0)
    labels_plain, n_plain = label_components(low)
    spans_plain = [len(t) for t in component_trajectories(labels_plain, n_plain, min_voxels=1, min_span=1).values()]
    low_d = ndimage.binary_dilation(low, structure=np.ones((3, 3, 3)))
    labels_d, n_d = label_components(low_d)
    spans_d = [len(t) for t in component_trajectories(labels_d, n_d, min_voxels=1, min_span=1).values()]
    assert max(spans_plain, default=0) <= 1
    assert max(spans_d, default=0) == nf


def test_trajectory_is_null_outside_detected_span(tmp_path):
    nx = ny = 8
    nf = 12
    rows = grid_rows(nx, ny, nf, shadow_fn=lambda ix, iy, f: 8 <= f <= 11 and iy == 8 and ix == 8 - (f - 8))
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, orgx=4.5, orgy=4.5)
    hkl = IntegrateHKL.read(p)
    params = DetectionParams(kernel=5, k=4.0, min_voxels=3, min_span=3, entry_frac=0.8,
                             beamstop_radius=5.0, radius=1, band=3.0)
    masks, report = detect_shadow(hkl.records, hkl.header, params)
    assert report["selected"] is not None
    assert report["span_frames"] == [8, 12]
    xs = report["trajectory_px"]["x_of_z"]
    assert xs[:7] == [None] * 7
    assert xs[7:] == [7.5, 7.0, 6.5, 6.0, 5.5]
    assert all(1 <= v <= nx for v in xs if v is not None)
    assert masks[0].all() and masks[5].all()
    masked = masked_flags(hkl.records, masks, hkl.header)
    for f in range(8, 12):
        idx = f * nx * ny + 7 * nx + (8 - (f - 8) - 1)
        assert masked[idx] == True


def test_detect_shadow_end_to_end_on_synthetic_hkl(tmp_path):
    nx = ny = 8
    nf = 6
    rows = grid_rows(nx, ny, nf, shadow_fn=lambda ix, iy, f: iy == 8 and ix == 8 - f)
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, orgx=4.5, orgy=4.5)
    hkl = IntegrateHKL.read(p)
    params = DetectionParams(kernel=5, k=4.0, min_voxels=3, min_span=3,
                             entry_frac=0.8, beamstop_radius=5.0, radius=1, band=3.0)
    masks, report = detect_shadow(hkl.records, hkl.header, params)
    assert report["selected"] is not None
    assert report["selected"]["slope_x"] < 0.0
    assert report["shape"] == [nf, ny, nx]
    masked = masked_flags(hkl.records, masks, hkl.header)
    assert masked.size == len(rows)
    for f in range(nf):
        idx = f * nx * ny + (8 - 1) * nx + (8 - f - 1)
        assert masked[idx] == True
    center = (4 - 1) * nx + (4 - 1)
    assert masked[center] == False


def test_detect_shadow_returns_all_keep_when_nothing_matches(tmp_path):
    nx = ny = 8
    rows = grid_rows(nx, ny, 3, shadow_fn=lambda ix, iy, f: False)
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny)
    hkl = IntegrateHKL.read(p)
    masks, report = detect_shadow(hkl.records, hkl.header, DetectionParams(kernel=5, k=4.0))
    assert report["selected"] is None
    assert masks.all()


def test_detect_shadow_inverse_geometry_needs_matching_prior(tmp_path):
    nx = ny = 8
    nf = 6
    rows = grid_rows(nx, ny, nf, shadow_fn=lambda ix, iy, f: iy == 1 and ix == 1 + f)
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, orgx=4.5, orgy=4.5)
    hkl = IntegrateHKL.read(p)
    base = dict(kernel=5, k=4.0, min_voxels=3, min_span=3, entry_frac=0.8,
                beamstop_radius=5.0, radius=1, band=3.0)
    _, report_default = detect_shadow(hkl.records, hkl.header, DetectionParams(**base))
    assert report_default["selected"] is None
    params = DetectionParams(**base, entry_edge="min_y", slope_sign=1)
    masks, report = detect_shadow(hkl.records, hkl.header, params)
    assert report["selected"] is not None
    assert report["selected"]["slope_x"] > 0.0
    assert report["selected"]["entry_y"] < (1 - 0.8) * ny
    masked = masked_flags(hkl.records, masks, hkl.header)
    for f in range(nf):
        idx = f * nx * ny + (1 - 1) * nx + (1 + f - 1)
        assert masked[idx] == True


def test_choose_block_caps_voxel_count():
    assert choose_block(6, 8, 8, budget=2 ** 30) == 1
    assert choose_block(3600, 4362, 4148, budget=2 ** 20) == 512


def test_auto_block_caps_memory_for_large_dataset(tmp_path):
    nx, ny, nf = 4148, 4362, 3600
    rows = [rec(1, 1, 1, 100.0, 100.0, 100.0, 1.0, 50.0),
            rec(2, 2, 2, 100.0, 2000.0, 2000.0, float(nf), 50.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny)
    hkl = IntegrateHKL.read(p)
    budget = 2 ** 20
    params = DetectionParams(kernel=31, k=5.0, min_voxels=1, min_span=1, voxel_budget=budget)
    masks, report = detect_shadow(hkl.records, hkl.header, params)
    assert report["block"] > 1
    assert report["shape"][0] * report["shape"][1] * report["shape"][2] <= budget
    assert report["pixel_shape"] == [ny, nx]
    assert masks.shape == tuple(report["shape"])
