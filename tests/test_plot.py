import json

from xdsmask.apply import save_masks
from xdsmask.cli import main
from xdsmask.detect import DetectionParams, detect_shadow
from xdsmask.hkl import IntegrateHKL
from xdsmask.plot import build_figure

from conftest import grid_rows, write_hkl


def _dataset(tmp_path):
    nx = ny = 8
    nf = 6
    rows = grid_rows(nx, ny, nf, shadow_fn=lambda ix, iy, f: iy == 8 and ix == 8 - f)
    hkl_path = write_hkl(tmp_path / "INTEGRATE.HKL", rows, nx=nx, ny=ny, orgx=4.5, orgy=4.5)
    hkl = IntegrateHKL.read(hkl_path)
    params = DetectionParams(kernel=5, k=4.0, min_voxels=3, min_span=3,
                             entry_frac=0.8, beamstop_radius=5.0, radius=1, band=3.0)
    masks, report = detect_shadow(hkl.records, hkl.header, params)
    return hkl_path, hkl, masks, report, nf


def test_build_figure_has_one_frame_per_detector_frame(tmp_path):
    import plotly.graph_objects as go

    _, hkl, masks, report, nf = _dataset(tmp_path)
    fig = build_figure(masks, hkl.records, hkl.header, report=report, block=report["block"])
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == nf
    assert fig.layout.sliders, "expected a frame slider"
    assert any(t.name == "trajectory" for t in fig.data)


def test_plot_cli_writes_interactive_html(tmp_path):
    hkl_path, hkl, masks, report, _ = _dataset(tmp_path)
    masks_path = tmp_path / "masks.npz"
    save_masks(masks_path, masks, block=report["block"])
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    out = tmp_path / "report.html"
    rc = main(["plot", "--masks", str(masks_path), "--integrate-hkl", str(hkl_path),
               "--report", str(report_path), "--out", str(out)])
    assert rc == 0
    content = out.read_text()
    assert "Plotly.newPlot" in content
    assert len(content) > 10000
