import numpy as np

from xdsmask.hkl import IntegrateHKL, write_filter_hkl

from conftest import rec, write_hkl


def test_read_parses_header_items_and_records(tmp_path):
    rows = [rec(1, 2, 3, 100.0, 2.0, 3.0, 1.0, 500.0),
            rec(-1, 0, 2, 5.0, 4.5, 6.5, 2.0, 10.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows)
    hkl = IntegrateHKL.read(p)
    assert hkl.header["nx"] == "8"
    assert hkl.header["starting_frame"] == "1"
    assert hkl.header["name_template_of_data_frames"] == "frame_????.img"
    assert len(hkl.items) == 21
    assert hkl.records.shape == (2,)
    assert hkl.records["h"][0] == 1
    assert hkl.records["h"].dtype == np.int32
    assert hkl.records["xcal"][1] == 4.5
    assert hkl.records["iseg"][0] == 1


def test_roundtrip_preserves_header_and_records(tmp_path):
    rows = [rec(1, 2, 3, -19.37, 2454.8, 2519.9, 111.7, 46.0)]
    src = write_hkl(tmp_path / "a.HKL", rows, nx=3072, ny=3072, orgx=1220.98, orgy=1263.78)
    hkl = IntegrateHKL.read(src)
    out = tmp_path / "b.HKL"
    hkl.write(out)
    hkl2 = IntegrateHKL.read(out)
    assert hkl2.header_lines == hkl.header_lines
    for name in hkl.items:
        assert np.allclose(hkl2.records[name], hkl.records[name], equal_nan=True)


def test_write_filter_hkl_emits_masked_records_only(tmp_path):
    rows = [rec(1, 0, 0, 10.0, 2.0, 3.0, 1.0, 5.0),
            rec(0, 1, 0, 20.0, 4.0, 5.0, 1.0, 50.0)]
    p = write_hkl(tmp_path / "INTEGRATE.HKL", rows)
    hkl = IntegrateHKL.read(p)
    masked = np.array([True, False])
    out = tmp_path / "FILTER.HKL"
    write_filter_hkl(out, hkl.records, masked)
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    parts = lines[0].split()
    assert [int(x) for x in parts[:3]] == [1, 0, 0]
    assert abs(float(parts[3]) - 2.0) < 1e-3
    assert abs(float(parts[5]) - 1.0) < 1e-3
    assert parts[6:] == ["0.500", "0.500", "0.500"]
