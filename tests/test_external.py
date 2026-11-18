import json

import numpy as np
import pytest

from xdsmask.apply import load_masks_meta, masked_flags
from xdsmask.cli import main
from xdsmask.external import block_reduce_any, load_external_masks
from xdsmask.hkl import IntegrateHKL

from conftest import rec, write_hkl


def _ext_npz(tmp_path, arr, key="mask"):
    ext = tmp_path / "ext.npz"
    np.savez(ext, **{key: arr})
    return ext


def test_load_external_masks_transpose_and_degree_mapping(tmp_path):
    nx, ny, nz = 8, 6, 4
    arr = np.zeros((nz, nx, ny), dtype=bool)
    arr[2, 2, 3] = True
    ext = _ext_npz(tmp_path, arr)
    rows = [rec(1, 0, 0, 10.0, 3.0, 4.0, 5.0, 5.0),
            rec(0, 1, 0, 20.0, 3.0, 4.0, 1.0, 50.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, sf=1, start_angle=0.0, osc=0.5)
    hkl = IntegrateHKL.read(p)
    masks, dpe = load_external_masks(ext, hkl.header, deg_per_entry=1.0)
    assert dpe == 1.0
    assert masks.shape == (nz, ny, nx)
    masked = masked_flags(hkl.records, masks, hkl.header, invert=True, deg_per_entry=dpe)
    assert masked.tolist() == [True, False]


def test_load_external_masks_starting_angle_offset(tmp_path):
    nx = ny = 8
    nz = 4
    arr = np.zeros((nz, ny, nx), dtype=bool)
    arr[3] = True
    ext = _ext_npz(tmp_path, arr)
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, sf=1, start_angle=270.0, osc=0.5)
    hkl = IntegrateHKL.read(p)
    masks, dpe = load_external_masks(ext, hkl.header, deg_per_entry=90.0)
    masked = masked_flags(hkl.records, masks, hkl.header, invert=True, deg_per_entry=dpe)
    assert masked.tolist() == [True]


def test_load_external_masks_shape_mismatch(tmp_path):
    arr = np.zeros((4, 5, 5), dtype=bool)
    ext = _ext_npz(tmp_path, arr)
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=8, ny=8)
    hkl = IntegrateHKL.read(p)
    with pytest.raises(ValueError, match="does not match"):
        load_external_masks(ext, hkl.header)


def test_load_external_masks_missing_key(tmp_path):
    ext = _ext_npz(tmp_path, np.zeros((4, 8, 8), dtype=bool), key="other")
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=8, ny=8)
    hkl = IntegrateHKL.read(p)
    with pytest.raises(KeyError, match="mask"):
        load_external_masks(ext, hkl.header)


def test_block_reduce_any():
    arr = np.zeros((2, 5, 7), dtype=bool)
    arr[0, 1, 2] = True
    arr[1, 4, 6] = True
    out = block_reduce_any(arr, 2)
    assert out.shape == (2, 3, 4)
    assert out[0, 0, 1] == True
    assert out[1, 2, 3] == True
    assert out.sum() == 2


def test_from_npz_cli_writes_filter_masks_report_html_and_apply_roundtrip(tmp_path):
    nx, ny, nz = 8, 6, 4
    arr = np.zeros((nz, nx, ny), dtype=bool)
    arr[2, 2, 3] = True
    ext = _ext_npz(tmp_path, arr)
    rows = [rec(1, 0, 0, 10.0, 3.0, 4.0, 5.0, 5.0)]
    hkl_path = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, sf=1, start_angle=0.0, osc=0.5)
    filt = tmp_path / "FILTER.HKL"
    mp = tmp_path / "masks.npz"
    rp = tmp_path / "report.json"
    hp = tmp_path / "report.html"
    rc = main(["from-npz", str(hkl_path), "--masks-in", str(ext), "--deg-per-entry", "1.0",
               "--filter-out", str(filt), "--masks-out", str(mp),
               "--report", str(rp), "--plot", str(hp)])
    assert rc == 0
    assert len(filt.read_text().splitlines()) == 1
    meta = load_masks_meta(mp)
    assert meta["masks"].shape == (nz, ny, nx)
    assert float(meta["deg_per_entry"]) == 1.0
    assert bool(meta["invert"]) is True
    report = json.loads(rp.read_text())
    assert report["masked_reflections"] == 1
    assert "Plotly.newPlot" in hp.read_text()
    filt2 = tmp_path / "FILTER2.HKL"
    rc = main(["apply", str(hkl_path), "--masks", str(mp), "--filter-out", str(filt2)])
    assert rc == 0
    assert len(filt2.read_text().splitlines()) == 1
