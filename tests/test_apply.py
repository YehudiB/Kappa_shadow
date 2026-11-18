import numpy as np
import pytest

from xdsmask.apply import apply_masks, load_masks, masked_flags, save_masks
from xdsmask.hkl import IntegrateHKL

from conftest import rec, write_hkl


def test_masked_flags_inside_outside_and_offdetector(tmp_path):
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0),
            rec(0, 1, 0, 20.0, 5.0, 5.0, 1.0, 50.0),
            rec(0, 0, 1, 30.0, 99.0, 99.0, 1.0, 50.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows)
    hkl = IntegrateHKL.read(p)
    masks = np.ones((2, 8, 8), dtype=bool)
    masks[0, 4, 4] = False
    masked = masked_flags(hkl.records, masks, hkl.header)
    assert masked.tolist() == [False, True, True]


def test_apply_masks_writes_filter_hkl_and_filtered_copy(tmp_path):
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0),
            rec(0, 1, 0, 20.0, 5.0, 5.0, 1.0, 50.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows)
    hkl = IntegrateHKL.read(p)
    masks = np.ones((1, 8, 8), dtype=bool)
    masks[0, 4, 4] = False
    filt = tmp_path / "FILTER.HKL"
    out = tmp_path / "INTEGRATE.masked.HKL"
    masked = apply_masks(hkl, masks, filter_out=filt, out=out)
    assert masked.tolist() == [False, True]
    assert len(filt.read_text().splitlines()) == 1
    hkl2 = IntegrateHKL.read(out)
    assert hkl2.records.shape == (1,)
    assert int(hkl2.records["h"][0]) == 1
    assert hkl2.header_lines == hkl.header_lines


def test_masks_roundtrip_npz(tmp_path):
    masks = np.zeros((3, 8, 8), dtype=bool)
    masks[0, 0, 0] = True
    p = tmp_path / "masks.npz"
    save_masks(p, masks, block=4)
    loaded, block = load_masks(p)
    assert block == 4
    assert loaded.dtype == bool
    assert np.array_equal(loaded, masks)


def test_masked_flags_flip_xy(tmp_path):
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=8, ny=8)
    hkl = IntegrateHKL.read(p)
    masks = np.ones((1, 8, 8), dtype=bool)
    masks[0, 6, 6] = False
    assert masked_flags(hkl.records, masks, hkl.header).tolist() == [False]
    assert masked_flags(hkl.records, masks, hkl.header, flip_x=True, flip_y=True).tolist() == [True]
    assert masked_flags(hkl.records, masks, hkl.header, flip_y=True).tolist() == [False]


def test_masked_flags_z_offset_is_cyclical(tmp_path):
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0),
            rec(0, 1, 0, 20.0, 2.0, 2.0, 3.0, 50.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=8, ny=8, sf=1, osc=0.5)
    hkl = IntegrateHKL.read(p)
    masks = np.ones((4, 8, 8), dtype=bool)
    masks[2, 1, 1] = False
    plain = masked_flags(hkl.records, masks, hkl.header)
    assert plain.tolist() == [False, True]
    shifted = masked_flags(hkl.records, masks, hkl.header, z_offset_deg=1.0)
    assert shifted.tolist() == [True, False]


def test_z_offset_requires_oscillation_range(tmp_path):
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=8, ny=8, osc=0.0)
    hkl = IntegrateHKL.read(p)
    masks = np.ones((1, 8, 8), dtype=bool)
    with pytest.raises(ValueError, match="OSCILLATION_RANGE"):
        masked_flags(hkl.records, masks, hkl.header, z_offset_deg=326.1)


def test_masked_flags_with_block_binning(tmp_path):
    rows = [rec(1, 0, 0, 10.0, 5.0, 5.0, 1.0, 5.0),
            rec(0, 1, 0, 20.0, 1.0, 1.0, 1.0, 50.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=8, ny=8)
    hkl = IntegrateHKL.read(p)
    masks = np.ones((1, 2, 2), dtype=bool)
    masks[0, 1, 1] = False
    masked = masked_flags(hkl.records, masks, hkl.header, block=4)
    assert masked.tolist() == [True, False]
