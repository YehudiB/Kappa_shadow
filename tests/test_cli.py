import json

import numpy as np

from xdsmask.apply import save_masks
from xdsmask.cli import main

from conftest import grid_rows, rec, write_hkl


def test_detect_and_apply_cli(tmp_path):
    nx = ny = 8
    nf = 6
    rows = grid_rows(nx, ny, nf, shadow_fn=lambda ix, iy, f: iy == 8 and ix == 8 - f)
    hkl_path = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, orgx=4.5, orgy=4.5)
    masks_path = tmp_path / "masks.npz"
    report_path = tmp_path / "report.json"
    rc = main([
        "detect", str(hkl_path),
        "--masks-out", str(masks_path),
        "--report", str(report_path),
        "--kernel", "5", "--k", "4.0",
        "--min-voxels", "3", "--min-span", "3",
        "--beamstop-radius", "5.0",
        "--radius", "1", "--band", "3.0",
    ])
    assert rc == 0
    assert masks_path.exists()
    report = json.loads(report_path.read_text())
    assert report["selected"] is not None
    assert report["selected"]["slope_x"] < 0.0
    assert report["block"] == 1
    filt = tmp_path / "FILTER.HKL"
    out = tmp_path / "INTEGRATE.masked.HKL"
    rc = main(["apply", str(hkl_path), "--masks", str(masks_path),
               "--filter-out", str(filt), "--out", str(out)])
    assert rc == 0
    filter_lines = filt.read_text().splitlines()
    assert len(filter_lines) > 0
    assert all(len(line.split()) == 9 for line in filter_lines)
    assert out.exists()


def test_detect_cli_inverse_geometry_flags(tmp_path):
    nx = ny = 8
    nf = 6
    rows = grid_rows(nx, ny, nf, shadow_fn=lambda ix, iy, f: iy == 1 and ix == 1 + f)
    hkl_path = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, orgx=4.5, orgy=4.5)
    masks_path = tmp_path / "masks.npz"
    report_path = tmp_path / "report.json"
    rc = main([
        "detect", str(hkl_path),
        "--masks-out", str(masks_path),
        "--report", str(report_path),
        "--kernel", "5", "--k", "4.0",
        "--min-voxels", "3", "--min-span", "3",
        "--beamstop-radius", "5.0",
        "--radius", "1", "--band", "3.0",
        "--entry-edge", "top", "--motion", "right",
    ])
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["selected"] is not None
    assert report["selected"]["slope_x"] > 0.0


def test_apply_cli_flip_xy(tmp_path):
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0)]
    hkl_path = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=8, ny=8)
    masks = np.ones((1, 8, 8), dtype=bool)
    masks[0, 6, 6] = False
    masks_path = tmp_path / "masks.npz"
    save_masks(masks_path, masks, block=1)
    filt = tmp_path / "FILTER.HKL"
    rc = main(["apply", str(hkl_path), "--masks", str(masks_path),
               "--filter-out", str(filt), "--flip-x", "--flip-y"])
    assert rc == 0
    assert len(filt.read_text().splitlines()) == 1


def test_apply_cli_z_offset_deg(tmp_path):
    rows = [rec(1, 0, 0, 10.0, 2.0, 2.0, 1.0, 5.0),
            rec(0, 1, 0, 20.0, 2.0, 2.0, 3.0, 50.0)]
    hkl_path = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=8, ny=8, sf=1, osc=0.5)
    masks = np.ones((4, 8, 8), dtype=bool)
    masks[2, 1, 1] = False
    masks_path = tmp_path / "masks.npz"
    save_masks(masks_path, masks, block=1)
    filt = tmp_path / "FILTER.HKL"
    rc = main(["apply", str(hkl_path), "--masks", str(masks_path),
               "--filter-out", str(filt), "--z-offset-deg", "1.0"])
    assert rc == 0
    lines = filt.read_text().splitlines()
    assert len(lines) == 1
    assert [int(v) for v in lines[0].split()[:3]] == [1, 0, 0]
