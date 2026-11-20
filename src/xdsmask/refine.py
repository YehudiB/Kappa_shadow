from __future__ import annotations

import glob
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

IMAGE_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5", ".cbf", ".tif", ".tiff", ".png"}

_FRAME_RE = re.compile(r"(\d+)")


def load_image(path):
    """Read a 2-D diffraction image as a float64 array.

    .npy/.npz are native; .h5/.hdf5 need h5py, .cbf needs fabio, .tif/.tiff need
    imageio or PIL (first 2-D dataset / first page is used for multi-dataset files).
    """
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
    elif suf == ".npz":
        with np.load(path) as data:
            key = "data" if "data" in data.files else data.files[0]
            arr = data[key]
    elif suf in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("reading .h5 images requires h5py (pip install h5py)") from exc
        with h5py.File(path, "r") as fh:
            found = []

            def _visit(name, obj):
                if hasattr(obj, "shape") and obj.ndim == 2 and obj.shape[0] > 1 and obj.shape[1] > 1:
                    found.append(name)

            fh.visititems(_visit)
            if not found:
                raise ValueError(f"{path}: no 2-D dataset found")
            arr = fh[sorted(found)[0]][()]
    elif suf == ".cbf":
        try:
            import fabio
        except ImportError as exc:
            raise ImportError("reading .cbf images requires fabio (pip install fabio)") from exc
        arr = fabio.open(path).data
    elif suf in (".tif", ".tiff", ".png"):
        try:
            import imageio.v3 as iio
            arr = iio.imread(path)
        except ImportError:
            try:
                from PIL import Image
            except ImportError as exc:
                raise ImportError("reading .tif/.png images requires imageio or PIL") from exc
            arr = np.asarray(Image.open(path))
    else:
        raise ValueError(f"unsupported image format {suf!r} for {path} (supported: {sorted(IMAGE_SUFFIXES)})")
    arr = np.asarray(arr)
    if arr.ndim != 2:
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            arr = squeezed
        else:
            raise ValueError(f"{path}: expected a 2-D image, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def save_mask_png(path, mask):
    """Write a boolean mask (True = shadowed) as a black/white png (True = white)."""
    arr = (np.asarray(mask, dtype=bool) * 255).astype(np.uint8)
    try:
        import imageio.v3 as iio
        iio.imwrite(path, arr)
    except ImportError:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("writing png requires imageio or PIL") from exc
        Image.fromarray(arr, mode="L").save(path)
    return str(path)


def sum_images(paths, bad_value=-1.0):
    """Load several images and return their sum, excluding defective pixels.

    A pixel that is bad (== bad_value or NaN) in ANY of the images is set to
    bad_value in the sum, so it never contributes to the refinement.
    """
    total = None
    bad = None
    for p in paths:
        img = load_image(p)
        if total is None:
            total = np.zeros(img.shape, dtype=np.float64)
            bad = np.zeros(img.shape, dtype=bool)
        elif img.shape != total.shape:
            raise ValueError(f"{p}: image shape {img.shape} != {total.shape}")
        good = (img != bad_value) & np.isfinite(img)
        total += np.where(good, img, 0.0)
        bad |= ~good
    total[bad] = bad_value
    return total


def collect_images(src):
    """Resolve --images to a sorted list of image paths.

    src may be a directory (scanned for known image suffixes), a glob pattern,
    a single image file, or a .txt file with one path per line.
    """
    src = str(src)
    p = Path(src)
    if p.is_dir():
        files = sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES and q.is_file())
    elif p.suffix.lower() == ".txt" and p.is_file():
        files = sorted(Path(line.strip()) for line in p.read_text().splitlines() if line.strip())
    elif p.is_file():
        files = [p]
    else:
        files = sorted(Path(f) for f in glob.glob(src))
    if not files:
        raise ValueError(f"no images found for {src!r}")
    return files


def frame_number_from_name(path, pattern=None):
    """Frame number encoded in the file name (last digit group of the stem by default)."""
    rx = re.compile(pattern) if pattern else _FRAME_RE
    m = rx.findall(Path(path).stem)
    if not m:
        raise ValueError(f"no frame number found in file name {path}")
    return int(m[-1])


def image_omega(frame_no, starting_angle, oscillation_range, starting_frame=1.0):
    """Rotation angle of an image, same convention as the shadow filter:
    omega(z) = STARTING_ANGLE + OSCILLATION_RANGE * (z - (STARTING_FRAME - 1))."""
    return float(starting_angle) + float(oscillation_range) * (float(frame_no) - (float(starting_frame) - 1.0))


def omega_to_entry(omega_deg, nz, deg_per_entry):
    """Nearest mask entry (index into the nz axis) for a rotation angle."""
    return int(np.mod(np.round(np.mod(omega_deg, 360.0) / deg_per_entry), nz))


def boundary_pixels(mask):
    """One-pixel skin of the True (shadowed) region; for a 1-px thin region the region itself."""
    m = np.asarray(mask, dtype=bool)
    eroded = ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool), border_value=0)
    edge = m & ~eroded
    if not edge.any():
        edge = m
    return np.argwhere(edge)


def boundary_chains(mask, max_gap=2.5):
    """Order the boundary pixels into polylines (one per connected edge curve).

    Greedy nearest-neighbour walk (8-connected boundary), starting from endpoints
    when available. Returns a list of (n, 2) float arrays of (y, x) points.
    """
    pts = boundary_pixels(mask).astype(float)
    if pts.size == 0:
        return []
    tree = cKDTree(pts)
    pairs = tree.query_pairs(max_gap, output_type="ndarray")
    deg = np.zeros(len(pts), dtype=int)
    if pairs.size:
        np.add.at(deg, pairs[:, 0], 1)
        np.add.at(deg, pairs[:, 1], 1)
    unused = np.ones(len(pts), dtype=bool)
    chains = []
    while unused.any():
        cand = np.nonzero(unused)[0]
        start = int(cand[np.argmin(deg[cand])])
        chain = [start]
        unused[start] = False
        cur = start
        while True:
            nbrs = [j for j in tree.query_ball_point(pts[cur], max_gap) if unused[j]]
            if not nbrs:
                break
            nxt = min(nbrs, key=lambda j: float(np.hypot(*(pts[j] - pts[cur]))))
            chain.append(nxt)
            unused[nxt] = False
            cur = nxt
        chains.append(pts[np.asarray(chain)])
    return chains


def resample_chain(pts, step):
    """Resample a polyline at uniform arc-length spacing `step` (px)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    d = np.hypot(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total <= step:
        return pts
    snew = np.linspace(0.0, total, int(round(total / step)) + 1)
    return np.column_stack([np.interp(snew, s, pts[:, 0]), np.interp(snew, s, pts[:, 1])])


def tangents_normals(pts):
    """Unit tangents (central differences) and normals (left perpendicular), (y, x) order."""
    pts = np.asarray(pts, dtype=float)
    t = np.zeros_like(pts)
    t[1:-1] = pts[2:] - pts[:-2]
    t[0] = pts[1] - pts[0]
    t[-1] = pts[-1] - pts[-2]
    norm = np.hypot(t[:, 0], t[:, 1])
    norm[norm == 0] = 1.0
    t = t / norm[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return t, n


def sigmoid(v, a, b, v0, s):
    return a + b / (1.0 + np.exp(-np.clip((v - v0) / s, -500.0, 500.0)))


def fit_edge_profile(profile, v, min_amp=0.0):
    """Fit a + b / (1 + exp(-(v - v0)/s)) to an integrated edge profile.

    NaN entries (too few valid pixels) are dropped. Returns dict(a, b, v0, s)
    or None when the profile has no usable edge (too few points, flat, or
    contrast below 4x the Poisson noise floor / min_amp).
    """
    p = np.asarray(profile, dtype=float)
    v = np.asarray(v, dtype=float)
    fin = np.isfinite(p)
    if int(fin.sum()) < max(10, int(0.5 * p.size)):
        return None
    p, v = p[fin], v[fin]
    lo, hi = float(p.min()), float(p.max())
    noise = float(np.median(np.sqrt(np.clip(p, 1.0, None))))
    if hi - lo < max(float(min_amp), 4.0 * noise):
        return None
    mid = 0.5 * (lo + hi)
    i0 = int(np.argmin(np.abs(p - mid)))
    p0 = [lo, hi - lo, float(v[i0]), max(2.0 * (v[1] - v[0]), 0.5)]
    bounds = ([-np.inf, -np.inf, float(v[0]), 0.2],
              [np.inf, np.inf, float(v[-1]), float(v[-1] - v[0])])
    sigma = np.sqrt(np.clip(p, 1.0, None))
    try:
        popt, _ = curve_fit(sigmoid, v, p, p0=p0, sigma=sigma, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    a, b, v0, s = (float(x) for x in popt)
    if not np.isfinite([a, b, v0, s]).all() or abs(b) < min_amp:
        return None
    return dict(a=a, b=b, v0=v0, s=s)


def sample_rectangle(image, py, px, ty, tx, ny, nx, width, length, bad_value=-1.0, min_valid_frac=0.5):
    """Integrate the image over a width x length rectangle centred on (py, px),
    long axis along the normal (ny, nx), short axis along the tangent (ty, tx).

    Defective pixels (value == bad_value, or NaN) do not contribute: the profile
    is the sum over the valid width samples renormalized to the full width, and
    points with fewer than min_valid_frac * width valid samples are NaN.

    Returns (profile, v): profile[i] at normal offset v[i].
    """
    u = np.arange(width) - (width - 1) / 2.0
    v = np.arange(length) - (length - 1) / 2.0
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ys = py + uu * ty + vv * ny
    xs = px + uu * tx + vv * nx
    samples = ndimage.map_coordinates(image, [ys, xs], order=1, mode="nearest")
    good = ((image != bad_value) & np.isfinite(image)).astype(float)
    gsamples = ndimage.map_coordinates(good, [ys, xs], order=1, mode="nearest")
    valid = gsamples >= 0.999
    counts = valid.sum(axis=0)
    sums = np.where(valid, samples, 0.0).sum(axis=0)
    min_count = max(1, int(math.ceil(min_valid_frac * width)))
    profile = np.where(counts >= min_count, sums * width / np.maximum(counts, 1), np.nan)
    return profile, v


def _logit(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _paint(mask, py, px, ny, nx, shift, fill):
    """Set the pixels on the normal segment from the curve point to the shifted edge."""
    if shift == 0.0:
        return
    n = max(1, int(abs(shift) / 0.5) + 1)
    ts = np.linspace(0.0, shift, n + 1)[1:]
    ys = np.rint(py + ts * ny).astype(int)
    xs = np.rint(px + ts * nx).astype(int)
    ok = (ys >= 0) & (ys < mask.shape[0]) & (xs >= 0) & (xs < mask.shape[1])
    mask[ys[ok], xs[ok]] = fill


@dataclass
class RefineParams:
    rect_width: int = 9        # rectangle width along the mask tangent (px)
    rect_length: int = 100     # rectangle length along the normal, straddling the curve (px)
    step: float = 2.0          # spacing of samples along the mask curve (px)
    edge_frac: float = 0.95    # sigmoid level that defines "plateau reached"
    max_shift: float = 25.0    # clamp for the local shift (px)
    smooth: int = 5            # median window (samples) smoothing the shifts along the curve; <=1 off
    min_amp: float = 0.0       # minimum sigmoid amplitude in integrated counts to accept a sample
    max_gap: float = 2.5       # max neighbour distance when chaining boundary pixels (px)
    bad_value: float = -1.0    # defective pixel value excluded from the rectangle integration (NaN always excluded)
    min_valid_frac: float = 0.5  # min fraction of valid width samples for a profile point to be used


@dataclass
class RefineResult:
    mask: np.ndarray                       # refined mask, True = shadowed
    points: np.ndarray                     # (n, 2) sample points (y, x) along the curve
    normals: np.ndarray                    # (n, 2) unit normals at the sample points
    shifts: np.ndarray                     # (n,) local edge shift along the normal (px), NaN if not fitted
    amps: np.ndarray                       # (n,) sigmoid amplitude b per sample, NaN if not fitted

    @property
    def n_valid(self):
        return int(np.isfinite(self.shifts).sum())

    @property
    def expansions(self):
        """Shifts re-signed so that positive = the shadowed region grew (px)."""
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(self.shifts), self.shifts * np.sign(self.amps), np.nan)

    def shifted_points(self):
        s = np.nan_to_num(self.shifts)
        return self.points + s[:, None] * self.normals


def refine_mask_frame(image, mask, params=None):
    """Refine one boolean mask frame (True = shadowed) against its diffraction image.

    The mask boundary is walked as a curve; at each sample a rect_width x rect_length
    rectangle perpendicular to the local tangent is integrated, the resulting
    low->high profile is fitted with a sigmoid, and the mask edge is locally shifted
    to where the high-background plateau is reached (edge_frac of the step).
    """
    params = params or RefineParams()
    image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if image.shape != mask.shape:
        raise ValueError(f"image shape {image.shape} != mask frame shape {mask.shape}")
    refined = mask.copy()
    all_points, all_normals, all_shifts, all_amps = [], [], [], []
    v = np.arange(params.rect_length) - (params.rect_length - 1) / 2.0
    for chain in boundary_chains(mask, max_gap=params.max_gap):
        pts = resample_chain(chain, params.step)
        if len(pts) < 3:
            continue
        t, n = tangents_normals(pts)
        raw = np.zeros(len(pts))
        amps = np.full(len(pts), np.nan)
        for i in range(len(pts)):
            prof, vv = sample_rectangle(image, pts[i, 0], pts[i, 1], t[i, 0], t[i, 1],
                                        n[i, 0], n[i, 1], params.rect_width, params.rect_length,
                                        bad_value=params.bad_value, min_valid_frac=params.min_valid_frac)
            fit = fit_edge_profile(prof, vv, min_amp=params.min_amp)
            if fit is None:
                continue
            ve = fit["v0"] + math.copysign(_logit(params.edge_frac) * fit["s"], fit["b"])
            raw[i] = float(np.clip(ve, -params.max_shift, params.max_shift))
            amps[i] = fit["b"]
        shifts = raw if params.smooth <= 1 else ndimage.median_filter(raw, size=params.smooth, mode="nearest")
        for i in range(len(pts)):
            if not np.isfinite(amps[i]):
                continue
            fill = bool(amps[i] * shifts[i] > 0.0)
            _paint(refined, pts[i, 0], pts[i, 1], n[i, 0], n[i, 1], shifts[i], fill)
            if i + 1 < len(pts):
                seg = max(int(np.hypot(*(pts[i + 1] - pts[i])) / 0.5), 1)
                for a in np.linspace(0.0, 1.0, seg + 1)[1:]:
                    p = pts[i] + a * (pts[i + 1] - pts[i])
                    nn = n[i] + a * (n[i + 1] - n[i])
                    sh = shifts[i] + a * (shifts[i + 1] - shifts[i])
                    _paint(refined, p[0], p[1], nn[0], nn[1], sh, fill)
        all_points.append(pts)
        all_normals.append(n)
        all_shifts.append(np.where(np.isfinite(amps), raw, np.nan))
        all_amps.append(amps)
    if all_points:
        points = np.concatenate(all_points)
        normals = np.concatenate(all_normals)
        shifts = np.concatenate(all_shifts)
        amps = np.concatenate(all_amps)
    else:
        points = np.zeros((0, 2))
        normals = np.zeros((0, 2))
        shifts = np.zeros(0)
        amps = np.zeros(0)
    return RefineResult(mask=refined, points=points, normals=normals, shifts=shifts, amps=amps)


def refine_stack(image_paths, mask_stack, deg_per_entry, omegas, params=None):
    """Refine every image against its mask entry and aggregate per entry.

    image_paths: list of paths; omegas: rotation angle (deg) per image, same length.
    Returns (refined_stack, per_image, report):
      refined_stack  -- (nz, ny, nx) bool, True = shadowed; entries without images keep
                        the original frame, entries with several images use a majority vote
      per_image      -- list of dicts (path, frame, omega_deg, entry, result)
      report         -- JSON-able dict with per-entry statistics
    """
    params = params or RefineParams()
    mask_stack = np.asarray(mask_stack, dtype=bool)
    nz, ny, nx = mask_stack.shape
    omegas = np.asarray(omegas, dtype=float)
    if len(image_paths) != omegas.size:
        raise ValueError("image_paths and omegas must have the same length")
    entries = np.mod(np.round(np.mod(omegas, 360.0) / deg_per_entry), nz).astype(np.int64)
    refined = mask_stack.copy()
    per_entry = {}
    per_image = []
    for path, om, en in zip(image_paths, omegas, entries):
        img = load_image(path)
        if img.shape != (ny, nx):
            raise ValueError(f"{path}: image shape {img.shape} != mask frame shape {(ny, nx)}")
        res = refine_mask_frame(img, mask_stack[int(en)], params)
        per_entry.setdefault(int(en), []).append(res.mask)
        per_image.append(dict(path=str(path), frame=frame_number_from_name(path),
                              omega_deg=float(om), entry=int(en), result=res))
    entry_stats = []
    for en, ms in sorted(per_entry.items()):
        refined[en] = np.stack(ms).mean(axis=0) >= 0.5
        shifts = np.concatenate([p["result"].expansions for p in per_image if p["entry"] == en])
        shifts = shifts[np.isfinite(shifts)]
        entry_stats.append(dict(
            entry=en,
            n_images=len(ms),
            frames=[p["frame"] for p in per_image if p["entry"] == en],
            n_samples=int(shifts.size),
            median_shift=float(np.median(shifts)) if shifts.size else None,
            mean_shift=float(np.mean(shifts)) if shifts.size else None,
            min_shift=float(np.min(shifts)) if shifts.size else None,
            max_shift=float(np.max(shifts)) if shifts.size else None,
        ))
    report = dict(
        n_images=len(per_image),
        deg_per_entry=float(deg_per_entry),
        pixel_shape=[ny, nx],
        n_entries_refined=len(per_entry),
        entries=entry_stats,
    )
    return refined, per_image, report


def refine_figure(image, result, max_px=900):
    """QC plot: image with the original (red) and shifted (green) mask curve."""
    import plotly.graph_objects as go

    img = np.asarray(image, dtype=float)
    f = max(1, int(math.ceil(max(img.shape) / max_px)))
    small = img[::f, ::f]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=small[::-1], colorscale="Gray", showscale=False, name="image"))
    if len(result.points):
        fig.add_trace(go.Scatter(x=result.points[:, 1], y=result.points[:, 0], mode="markers",
                                 name="theoretical edge", marker=dict(size=3, color="red")))
        sp = result.shifted_points()
        fig.add_trace(go.Scatter(x=sp[:, 1], y=sp[:, 0], mode="markers",
                                 name="refined edge", marker=dict(size=3, color="lime")))
    fig.update_layout(title=f"mask refinement: {result.n_valid} fitted edge samples",
                      xaxis_title="X (pixels)", yaxis=dict(autorange="reversed"), yaxis_title="Y (pixels)",
                      width=900, height=800, template="plotly_white")
    return fig
