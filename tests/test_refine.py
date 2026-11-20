import json

import numpy as np
import pytest

from xdsmask.cli import main
from xdsmask.external import load_external_masks
from xdsmask.hkl import IntegrateHKL
from xdsmask.refine import (
    RefineParams,
    boundary_chains,
    collect_images,
    fit_edge_profile,
    frame_number_from_name,
    image_omega,
    omega_to_entry,
    refine_mask_frame,
    resample_chain,
    sample_rectangle,
    sigmoid,
)

from conftest import rec, write_hkl


def synth_image(ny, nx, x_edge, s=2.0, lo=5.0, hi=200.0):
    """Vertical soft shadow edge: low counts for x << x_edge, high plateau for x >> x_edge."""
    x = np.arange(nx)[None, :].astype(float)
    return np.repeat(lo + (hi - lo) / (1.0 + np.exp(-(x - x_edge) / s)), ny, axis=0)


def half_mask(ny, nx, x0):
    yy, xx = np.indices((ny, nx))
    return xx <= x0


def test_omega_and_entry_mapping():
    assert image_omega(0, 326.01, 0.1, 1.0) == pytest.approx(326.01)
    assert image_omega(1000, 326.01, 0.1, 1.0) == pytest.approx(426.01)
    assert omega_to_entry(0.0, 360, 1.0) == 0
    assert omega_to_entry(359.6, 360, 1.0) == 0
    assert omega_to_entry(90.0, 4, 90.0) == 1
    assert omega_to_entry(-90.0, 4, 90.0) == 3
    assert frame_number_from_name("frame_00123.npy") == 123
    assert frame_number_from_name("run2_scan-7_0042.h5") == 42


def test_collect_images_dir_and_list(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in (2, 1, 10):
        np.save(d / f"frame_{i:04d}.npy", np.zeros((4, 4)))
    (d / "notes.txt").write_text("hello\n")
    files = collect_images(d)
    assert [f.name for f in files] == ["frame_0001.npy", "frame_0002.npy", "frame_0010.npy"]
    lst = tmp_path / "list.txt"
    lst.write_text("\n".join(str(f) for f in files) + "\n")
    assert collect_images(lst) == files
    with pytest.raises(ValueError, match="no images"):
        collect_images(tmp_path / "empty")


def test_boundary_chains_rectangle():
    mask = np.zeros((60, 60), dtype=bool)
    mask[:41, :31] = True
    chains = boundary_chains(mask)
    assert len(chains) == 1
    pts = resample_chain(chains[0], 2.0)
    assert len(pts) == pytest.approx(2 * (30 + 40) / 2, abs=3)


def test_fit_edge_profile_recovers_sigmoid_and_rejects_flat():
    v = np.arange(100) - 49.5
    p = sigmoid(v, 5.0, 195.0, 3.0, 1.5)
    fit = fit_edge_profile(p, v)
    assert fit is not None
    assert fit["v0"] == pytest.approx(3.0, abs=0.2)
    assert fit["s"] == pytest.approx(1.5, abs=0.2)
    assert fit["b"] == pytest.approx(195.0, rel=0.05)
    assert fit_edge_profile(np.full(100, 500.0), v) is None
    assert fit_edge_profile(np.full(100, 5.0), v) is None


def test_refine_mask_frame_expands_to_plateau():
    ny = nx = 120
    image = synth_image(ny, nx, x_edge=65.0, s=2.0)
    mask = half_mask(ny, nx, 60)
    res = refine_mask_frame(image, mask, RefineParams())
    # edge of the plateau: x = 65 + logit(0.95)*2 = 70.9 -> shift ~ +10.9 px
    med = float(np.median(res.shifts[np.isfinite(res.shifts)]))
    assert 8.0 < med < 13.5
    assert res.n_valid > 30
    assert not mask[60, 69] and res.mask[60, 69]
    assert res.mask[60, 73] == mask[60, 73] == False


def test_refine_mask_frame_shrinks_when_mask_overcovers():
    ny = nx = 120
    image = synth_image(ny, nx, x_edge=60.0, s=2.0)
    mask = half_mask(ny, nx, 70)
    res = refine_mask_frame(image, mask, RefineParams())
    med = float(np.median(res.shifts[np.isfinite(res.shifts)]))
    assert -6.5 < med < -2.0
    assert mask[60, 68] and not res.mask[60, 68]
    assert res.mask[60, 63]


def test_refine_mask_frame_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        refine_mask_frame(np.zeros((10, 10)), np.zeros((12, 12), dtype=bool))


def test_sample_rectangle_excludes_bad_pixels():
    image = np.full((40, 60), 100.0)
    image[:, 10] = -1.0            # narrow defective stripe
    image[:, 30:35] = -1.0         # wide dead block (5 px > min_valid_frac * 9)
    # rectangle centred at (20, 12), long axis along +x, 9 wide along y
    prof, v = sample_rectangle(image, 20.0, 12.0, 1.0, 0.0, 0.0, 1.0, 9, 40)
    x = 12.0 + v                   # v is the offset along the normal (+x)
    touched_bad = (np.abs(x - 10.0) < 1.0) | ((x > 29.0) & (x < 35.0))
    assert np.isnan(prof[touched_bad]).all()          # grid points touching a -1 pixel are dropped
    assert np.isfinite(prof[~touched_bad]).all()
    assert prof[~touched_bad] == pytest.approx(900.0)  # level renormalized to the full width


def test_refine_mask_frame_with_defective_pixels():
    ny = nx = 120
    image = synth_image(ny, nx, x_edge=65.0, s=2.0)
    image[:, 63:65] = -1.0         # defective stripe right at the soft edge
    image[80:100, 90:110] = -1.0   # dead block inside the plateau
    mask = half_mask(ny, nx, 60)
    res = refine_mask_frame(image, mask, RefineParams())
    exp = res.expansions[np.isfinite(res.expansions)]
    med = float(np.median(exp))
    assert 8.0 < med < 13.5        # same edge as without defects (~10.9 px)
    assert res.mask[60, 69] and not res.mask[60, 73]


def test_refine_mask_frame_curved_edge():
    ny = nx = 140
    yy, xx = np.indices((ny, nx))
    r = np.hypot(xx - 70, yy - 70)
    mask = r <= 30.0                                  # hard theoretical edge at r=30
    image = 5.0 + 195.0 / (1.0 + np.exp(-(r - 36.0) / 2.0))  # soft edge at r=36
    res = refine_mask_frame(image, mask, RefineParams())
    exp = res.expansions[np.isfinite(res.expansions)]
    assert res.n_valid > 50
    assert 9.0 < float(np.median(exp)) < 14.5          # (36-30) + logit(0.95)*2 = 11.9
    ang = np.deg2rad(30)
    for rr, want in ((38, True), (41, True), (44, False)):
        y, x = int(round(70 + rr * np.sin(ang))), int(round(70 + rr * np.cos(ang)))
        assert res.mask[y, x] == want


def _write_stack_and_images(tmp_path, nz=4, ny=60, nx=60, x_mask=20, x_edge=25.0):
    arr = np.zeros((nz, ny, nx), dtype=bool)
    arr[:, :, : x_mask + 1] = True
    mask_npz = tmp_path / "mask.npz"
    np.savez(mask_npz, mask=arr)
    imgdir = tmp_path / "frames"
    imgdir.mkdir()
    for f in range(1, nz + 1):
        np.save(imgdir / f"frame_{f:04d}.npy", synth_image(ny, nx, x_edge=x_edge))
    rows = [rec(1, 0, f, 10.0, 30.0, 30.0, float(f), 5.0) for f in range(1, nz + 1)]
    hkl = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, sf=1, start_angle=0.0, osc=90.0)
    return mask_npz, imgdir, hkl


def test_refine_cli_end_to_end(tmp_path):
    mask_npz, imgdir, hkl = _write_stack_and_images(tmp_path)
    out = tmp_path / "refined.npz"
    rep = tmp_path / "report.json"
    per_image = tmp_path / "per_image.npz"
    rc = main(["refine", str(hkl), "--images", str(imgdir), "--mask", str(mask_npz),
               "--masks-out", str(out), "--report", str(rep), "--per-image-out", str(per_image)])
    assert rc == 0
    with np.load(out) as d:
        refined = d["mask"]
        assert d["deg_per_entry"] == pytest.approx(90.0)
    assert refined.shape == (4, 60, 60)
    # plateau edge at x = 25 + logit(0.95)*2 = 30.9: refined mask covers up to ~x=31
    assert refined[0, 30, 30]
    assert not refined[0, 30, 34]
    assert refined[3, 10, 30] and not refined[3, 10, 34]
    report = json.loads(rep.read_text())
    assert report["n_images"] == 4
    assert report["n_entries_refined"] == 4
    assert report["entries"][0]["n_images"] == 1
    # the penumbra-edge samples report the true shift (~10.9 px); detector-border
    # samples see the same edge farther away and clamp at --max-shift 25
    assert 8.0 < report["entries"][0]["min_shift"] < 13.5
    assert report["entries"][0]["max_shift"] == pytest.approx(25.0)
    with np.load(per_image) as d:
        assert d["mask"].shape == (4, 60, 60)
        assert d["entry"].tolist() == [1, 2, 3, 0]  # omega = 90*frame mod 360
        assert d["frame"].tolist() == [1, 2, 3, 4]
    # the refined npz plugs straight back into from-npz
    hkl_obj = IntegrateHKL.read(hkl)
    arr, dpe = load_external_masks(out, hkl_obj.header)
    assert arr.shape == (4, 60, 60) and dpe == pytest.approx(90.0)


def test_refine_cli_without_hkl_uses_flags(tmp_path):
    mask_npz, imgdir, _ = _write_stack_and_images(tmp_path)
    out = tmp_path / "refined2.npz"
    rc = main(["refine", "--images", str(imgdir), "--mask", str(mask_npz),
               "--starting-angle", "0", "--oscillation-range", "90", "--masks-out", str(out)])
    assert rc == 0
    with np.load(out) as d:
        assert d["mask"].shape == (4, 60, 60)
        assert d["mask"][2, 30, 30] and not d["mask"][2, 30, 34]


def test_refine_cli_plot(tmp_path):
    pytest.importorskip("plotly")
    mask_npz, imgdir, hkl = _write_stack_and_images(tmp_path, nz=1)
    # nz=1 -> osc must map all frames to entry 0; use a fresh 1-entry hkl
    rows = [rec(1, 0, 1, 10.0, 30.0, 30.0, 1.0, 5.0)]
    hkl = write_hkl(tmp_path / "INTEGRATE1.HKL", rows, nx=60, ny=60, sf=1, start_angle=0.0, osc=360.0)
    html = tmp_path / "qc.html"
    rc = main(["refine", str(hkl), "--images", str(imgdir), "--mask", str(mask_npz),
               "--plot", str(html)])
    assert rc == 0
    assert "Plotly.newPlot" in html.read_text()
    assert "refined edge" in html.read_text()
