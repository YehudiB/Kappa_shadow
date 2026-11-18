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
    return masks, report, nf


def test_build_figure_is_3d_with_masked_voxels_and_trajectory(tmp_path):
    import plotly.graph_objects as go

    masks, report, nf = _dataset(tmp_path)
    fig = build_figure(masks, report=report)
    assert isinstance(fig, go.Figure)
    assert {t.type for t in fig.data} == {"scatter3d"}
    names = {t.name for t in fig.data}
    assert names == {"masked voxels", "shadow trajectory", "detector volume"}
    masked_trace = next(t for t in fig.data if t.name == "masked voxels")
    assert len(masked_trace.x) == int((~masks).sum()) > 0
    traj_trace = next(t for t in fig.data if t.name == "shadow trajectory")
    assert len(traj_trace.x) == nf
    assert min(traj_trace.z) == report["starting_frame"]


def test_plot_cli_writes_3d_html(tmp_path):
    masks, report, _ = _dataset(tmp_path)
    masks_path = tmp_path / "masks.npz"
    save_masks(masks_path, masks, block=report["block"])
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    out = tmp_path / "report.html"
    rc = main(["plot", "--masks", str(masks_path), "--report", str(report_path), "--out", str(out)])
    assert rc == 0
    content = out.read_text()
    assert "Plotly.newPlot" in content
    assert "scatter3d" in content
    assert "masked voxels" in content
