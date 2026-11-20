import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xdsmask.refine import load_image, save_mask_png

from conftest import rec, write_hkl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "xds_shadow_refine.py"


def synth_image(ny, nx, x_edge, s=2.0, lo=5.0, hi=200.0):
    x = np.arange(nx)[None, :].astype(float)
    return np.repeat(lo + (hi - lo) / (1.0 + np.exp(-(x - x_edge) / s)), ny, axis=0)


def half_mask(ny, nx, x0):
    yy, xx = np.indices((ny, nx))
    return xx <= x0


def test_standalone_script_is_self_contained():
    text = SCRIPT.read_text()
    assert "from ." not in text and "from xdsmask" not in text and "import xdsmask" not in text
    assert "def refine_mask_frame" in text and "def grid_search" in text


def test_standalone_refine_debug(tmp_path):
    pytest.importorskip("PIL")
    ny = nx = 120
    mask_png = tmp_path / "mask.png"
    save_mask_png(mask_png, half_mask(ny, nx, 60))
    imgs = []
    for i in range(3):
        p = tmp_path / f"frame_{i + 1:04d}.npy"
        np.save(p, synth_image(ny, nx, x_edge=65.0))
        imgs.append(str(p))
    out = tmp_path / "refined_mask.png"
    rc = subprocess.run([sys.executable, str(SCRIPT), "refine-debug", str(mask_png), *imgs,
                         "-o", str(out), "--grid-search", "--gs-frames", "1,3", "--gs-widths", "9,27"],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert "median_shift" in rc.stdout and "timing:" in rc.stdout
    refined = load_image(out) > 127
    assert refined[60, 69] and not refined[60, 73]  # edge at ~x=71


def test_standalone_refine_with_npz(tmp_path):
    nz, ny, nx = 4, 60, 60
    arr = np.zeros((nz, ny, nx), dtype=bool)
    arr[:, :, :21] = True
    mask_npz = tmp_path / "mask.npz"
    np.savez(mask_npz, mask=arr)
    imgdir = tmp_path / "frames"
    imgdir.mkdir()
    for f in range(1, nz + 1):
        np.save(imgdir / f"frame_{f:04d}.npy", synth_image(ny, nx, x_edge=25.0))
    rows = [rec(1, 0, f, 10.0, 30.0, 30.0, float(f), 5.0) for f in range(1, nz + 1)]
    hkl = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, sf=1, start_angle=0.0, osc=90.0)
    out = tmp_path / "refined.npz"
    rep = tmp_path / "report.json"
    rc = subprocess.run([sys.executable, str(SCRIPT), "refine", str(hkl), "--images", str(imgdir),
                         "--mask", str(mask_npz), "--masks-out", str(out), "--report", str(rep)],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert "refined 4 of 4" in rc.stdout
    with np.load(out) as d:
        refined = d["mask"]
        assert refined.shape == (nz, ny, nx)
        assert refined[0, 30, 30] and not refined[0, 30, 34]
    import json
    report = json.loads(rep.read_text())
    assert report["n_images"] == 4 and "refine_seconds" in report
