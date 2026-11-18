from __future__ import annotations

import numpy as np

from .apply import masked_flags
from .voxelize import geometry


def build_figure(masks, records, header, report=None, block=1, max_points_per_frame=50000):
    import plotly.graph_objects as go

    nf, nyb, nxb = masks.shape
    _, _, sf = geometry(header)
    masked = masked_flags(records, masks, header, block=block)
    f = np.round(records["zcal"]).astype(np.int64) - sf
    x = np.round(records["xcal"]).astype(np.float64)
    y = np.round(records["ycal"]).astype(np.float64)
    valid = (f >= 0) & (f < nf)
    f, x, y, masked = f[valid], x[valid], y[valid], masked[valid]
    hkl = [f"{int(r['h'])},{int(r['k'])},{int(r['l'])}" for r in records[valid]]

    hx = (np.arange(nxb) + 0.5) * block
    hy = (np.arange(nyb) + 0.5) * block
    colorscale = [[0.0, "rgba(214,39,40,0.55)"], [1.0, "rgba(255,255,255,0)"]]
    rng = np.random.default_rng(0)

    traj_x = traj_y = None
    if report:
        tp = report.get("trajectory_px") or report.get("trajectory")
        if tp:
            traj_x = np.asarray(tp["x_of_z"], dtype=float)
            traj_y = np.asarray(tp["y_of_z"], dtype=float)

    def frame_traces(fr):
        heat = go.Heatmap(z=masks[fr].astype(float), x=hx, y=hy, zmin=0, zmax=1,
                          colorscale=colorscale, showscale=False, hoverinfo="skip", name="mask")
        sel = np.nonzero(f == fr)[0]
        if sel.size > max_points_per_frame:
            sel = rng.choice(sel, max_points_per_frame, replace=False)
        colors = ["red" if m else "royalblue" for m in masked[sel]]
        scat = go.Scatter(x=x[sel], y=y[sel], mode="markers", name="reflections",
                          marker=dict(size=4, color=colors, opacity=0.7),
                          text=[hkl[i] for i in sel],
                          hovertemplate="hkl=%{text}<br>x=%{x:.1f} y=%{y:.1f}<extra></extra>")
        mark = go.Scatter(
            x=[traj_x[fr]] if traj_x is not None else [],
            y=[traj_y[fr]] if traj_y is not None else [],
            mode="markers", name="shadow centroid",
            marker=dict(size=12, color="black", symbol="x"),
        )
        return [heat, scat, mark]

    traces = frame_traces(0)
    if traj_x is not None:
        traces.append(go.Scatter(x=traj_x, y=traj_y, mode="lines+markers", name="trajectory",
                                 line=dict(color="black", dash="dash"), marker=dict(size=5)))
    fig = go.Figure(data=traces)
    fig.frames = [go.Frame(data=frame_traces(fr), name=str(fr)) for fr in range(nf)]
    steps = [
        {
            "args": [[str(fr)], {"frame": {"duration": 0, "redraw": True},
                                 "mode": "immediate", "transition": {"duration": 0}}],
            "label": str(fr + sf),
            "method": "animate",
        }
        for fr in range(nf)
    ]
    fig.update_layout(
        title="xdsmask: detected shadow mask per frame (red = masked)",
        xaxis_title="X (pixels)",
        yaxis_title="Y (pixels)",
        yaxis=dict(autorange="reversed"),
        sliders=[dict(active=0, steps=steps, currentvalue={"prefix": "frame "})],
        updatemenus=[dict(type="buttons", buttons=[dict(
            label="Play", method="animate",
            args=[None, {"frame": {"duration": 200, "redraw": True},
                         "fromcurrent": True, "transition": {"duration": 0}}])])],
        width=950, height=950, template="plotly_white",
    )
    return fig
