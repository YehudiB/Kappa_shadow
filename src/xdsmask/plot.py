from __future__ import annotations

import numpy as np


def _box_edges(nx, ny, z0, z1):
    corners = [(0, 0, z0), (nx, 0, z0), (nx, ny, z0), (0, ny, z0),
               (0, 0, z1), (nx, 0, z1), (nx, ny, z1), (0, ny, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return xs, ys, zs


def build_figure(masks, report=None, max_voxels=200000):
    import plotly.graph_objects as go

    nf, nyb, nxb = masks.shape
    report = report or {}
    block = int(report.get("block", 1))
    sf = int(report.get("starting_frame", 1))
    pixel_shape = report.get("pixel_shape")
    ny = int(pixel_shape[0]) if pixel_shape else nyb * block
    nx = int(pixel_shape[1]) if pixel_shape else nxb * block

    zc, yc, xc = np.nonzero(~masks)
    if zc.size > max_voxels:
        rng = np.random.default_rng(0)
        sel = rng.choice(zc.size, max_voxels, replace=False)
        zc, yc, xc = zc[sel], yc[sel], xc[sel]
    masked_trace = go.Scatter3d(
        x=(xc + 0.5) * block,
        y=(yc + 0.5) * block,
        z=zc + sf,
        mode="markers",
        name="masked voxels",
        marker=dict(size=2, color="red", opacity=0.6),
        hovertemplate="frame=%{z}<br>x=%{x:.0f} y=%{y:.0f}<extra></extra>",
    )
    traces = [masked_trace]

    tp = report.get("trajectory_px") or report.get("trajectory")
    if tp:
        xs = np.asarray(tp["x_of_z"], dtype=float)
        ys = np.asarray(tp["y_of_z"], dtype=float)
        zz = np.arange(nf, dtype=float) + sf
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        traces.append(go.Scatter3d(
            x=xs[ok], y=ys[ok], z=zz[ok],
            mode="lines+markers", name="shadow trajectory",
            line=dict(color="black", width=4), marker=dict(size=4, color="black"),
            hovertemplate="frame=%{z}<br>x=%{x:.1f} y=%{y:.1f}<extra></extra>",
        ))

    bx, by, bz = _box_edges(nx, ny, sf, nf - 1 + sf)
    traces.append(go.Scatter3d(
        x=bx, y=by, z=bz, mode="lines", name="detector volume",
        line=dict(color="grey", width=1, dash="dot"), hoverinfo="skip",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"xdsmask: masked voxels in (frame, y, x) — {int((~masks).sum())} masked, block={block} px",
        scene=dict(
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)",
            zaxis_title="frame",
            yaxis=dict(autorange="reversed"),
        ),
        width=1000, height=900, template="plotly_white",
    )
    return fig
