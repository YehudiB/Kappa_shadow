#!/usr/bin/env python3
"""
xds_shadow_filter.py  --  exclude shadowed reflections from XDS CORRECT

Reads an XDS INTEGRATE.HKL, looks every reflection up in a rotation-resolved
shadow mask (the 360-frame .npz written by the mini-kappa goniometer / pin-shadow
visualizer: mask[omega_deg, row, col], True = shadowed) and writes a FILTER.HKL
that makes XDS ignore the shadowed reflections in the CORRECT step.

QC figure (png): the diffraction images summed at the 25th / 50th / 75th
percentile of the shadow travel, with the shadowed mask entry outlined and a
red circle on every reflection that FILTER.HKL removes.  A sorted-intensity
(iobs) scatter of the removed reflections is drawn below the three panels.

Geometry used
-------------
  omega(z) = STARTING_ANGLE + OSCILLATION_RANGE * (z - (STARTING_FRAME - 1))
  (z is the continuous frame number, e.g. ZCAL; for STARTING_FRAME = 1 this is
   simply STARTING_ANGLE + OSCILLATION_RANGE * z, as in the task description)
  mask frame  = omega(z) mod 360      (the omega scan repeats every 360 deg)
  mask pixel  : column = floor(X), row = floor(Y)   (XDS pixel coordinates:
                lower-left corner of the first pixel is (0, 0))

Example: STARTING_ANGLE 326.01, OSCILLATION_RANGE 0.1
         z = 0    -> 326.01 deg -> mask frame 326
         z = 1000 -> 426.01 deg ->  66.01 deg -> mask frame 66

"Play" on the match (the mask has one frame per degree, the data one per 0.1 deg):
  omega_margin_deg : a reflection at omega is tested against every mask frame
                     within +-omega_margin_deg (default 1.0 -> the two mask
                     frames that bracket it; 0.5 -> nearest mask frame only).
  xy_margin_px     : a reflection is shadowed if any mask pixel within
                     +-xy_margin_px of (XCAL, YCAL) is shadowed.
  A reflection is excluded if it is shadowed in ANY of the tested mask frames.

Command line
------------
  python xds_shadow_filter.py INTEGRATE.HKL mask_omscan_Eiger2X16M_k200.0.npz \
         --images DIR [-o FILTER.HKL] [--flip-vertical] [--angle-offset 0.0] \
         [--qc-image filter_qc.png]

Python
------
  from xds_shadow_filter import make_filter_hkl
  res = make_filter_hkl("INTEGRATE.HKL", "mask.npz", "FILTER.HKL",
                        flip_vertical=False, flip_time=False, images="DIR",
                        qc_image="filter_qc.png")

Only numpy (and matplotlib for the plot) are required; fabio (cbf) / imageio or
PIL (tif/png) to read the diffraction images.
Assumes a single-segment detector (X, Y in whole-detector pixel coordinates).
"""
from __future__ import annotations

import argparse
import glob
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- INTEGRATE.HKL
N_COLS = 21
COLS = ("h k l iobs sigma xcal ycal zcal rlp peak corr maxc xobs yobs zobs "
        "alf0 bet0 alf1 bet1 psi iseg").split()
IC = {name: i for i, name in enumerate(COLS)}

_KV = re.compile(r"([A-Za-z0-9_.\-]+)=\s*([^\s=]+)")


def read_integrate_hkl(path):
    """Return (header: dict[str, str], data: float array (n_reflections, 21))."""
    header, chunks = {}, []
    with open(path, "r") as fh:
        in_header = True
        for line in fh:
            if in_header:
                if line.startswith("!"):
                    if line.startswith("!END_OF_HEADER"):
                        in_header = False
                    else:
                        for key, val in _KV.findall(line[1:]):
                            header.setdefault(key, val)
                    continue
                in_header = False          # tolerate a missing !END_OF_HEADER
            if line.startswith("!"):       # !END_OF_DATA
                break
            if line.strip():
                chunks.append(line)
    text = "".join(chunks)
    try:
        flat = np.fromstring(text, dtype=np.float64, sep=" ")
    except Exception:                      # pragma: no cover - very old/new numpy
        flat = np.array(text.split(), dtype=np.float64)
    if flat.size % N_COLS:
        raise ValueError(f"{path}: {flat.size} numbers is not a multiple of {N_COLS} columns")
    return header, flat.reshape(-1, N_COLS)


def _read_lp_value(lp_path, key):
    """Look up 'KEY= value' in INTEGRATE.LP (fallback for the oscillation range)."""
    rx = re.compile(rf"\b{key}=\s*([-+0-9.eE]+)")
    with open(lp_path, "r", errors="replace") as fh:
        for line in fh:
            m = rx.search(line)
            if m:
                return float(m.group(1))
    return None


# --------------------------------------------------------------------------- mask access
class MaskSource:
    """Frame-by-frame access to a mask (n_frames, rows, cols).

    Accepts a numpy array, a .npy file (memory-mapped) or the .npz written by the
    visualizer. For .npz the big array is read as a stream, one frame at a time,
    so a 6.5 GB mask never has to fit in memory.
    """

    def __init__(self, source, key="mask"):
        self._arr = None
        self._npz = None
        if isinstance(source, np.ndarray):
            self._arr = source
        else:
            path = Path(source)
            if path.suffix.lower() == ".npy":
                self._arr = np.load(path, mmap_mode="r")
            else:
                self._npz = path
                with zipfile.ZipFile(path) as zf, zf.open(key + ".npy") as fh:
                    self._read_header(fh)
                self._key = key
        if self._arr is not None:
            if self._arr.ndim != 3:
                raise ValueError(f"mask must be 3-D (frames, rows, cols), got {self._arr.shape}")
            self.shape = self._arr.shape

    def _read_header(self, fh):
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        if fortran or len(shape) != 3:
            raise ValueError(f"unsupported mask array (shape {shape}, fortran_order={fortran})")
        self.shape, self._dtype = tuple(shape), dtype

    def frames(self, needed):
        """Yield (frame_index, 2-D array) for every index in `needed`, in ascending order."""
        needed = sorted(set(int(k) for k in needed))
        if not needed:
            return
        if self._arr is not None:
            for k in needed:
                yield k, np.asarray(self._arr[k])
            return
        n, rows, cols = self.shape
        nbytes = rows * cols * self._dtype.itemsize
        wanted, last = set(needed), needed[-1]
        with zipfile.ZipFile(self._npz) as zf, zf.open(self._key + ".npy") as fh:
            self._read_header(fh)
            for k in range(last + 1):
                buf = fh.read(nbytes)
                if len(buf) != nbytes:
                    raise IOError("mask.npy ended early (truncated download?)")
                if k in wanted:
                    yield k, np.frombuffer(buf, dtype=self._dtype).reshape(rows, cols)


# --------------------------------------------------------------------------- main function
def make_filter_hkl(
    integrate_hkl,
    mask,
    filter_hkl="FILTER.HKL",
    *,
    flip_vertical=False,        # mask rows reversed (row 0 <-> last row)
    flip_horizontal=False,      # mask columns reversed (extra, in case X is mirrored)
    flip_time=False,            # mask k holds omega = -k deg: mask 350 <-> 10 deg, 360 <-> 0 deg
    invert_mask=False,          # use if your mask has True = valid instead of True = shadowed
    starting_angle=None,        # override !STARTING_ANGLE
    oscillation_range=None,     # override !OSCILLATION_RANGE
    starting_frame=None,        # override !STARTING_FRAME
    angle_offset=0.0,            # added to STARTING_ANGLE (before the time flip)
    integrate_lp=None,          # INTEGRATE.LP, only used if OSCILLATION_RANGE is not in the .HKL header
    omega_margin_deg=0.5,       # "play" in omega (see module docstring)
    xy_margin_px=2.0,           # "play" in detector position
    filter_box=(0.5, 0.5, 0.5), # DX DY DZ written to FILTER.HKL
    pixel_offset=0.0,           # 0 = XDS corner convention; use 0.5 if X,Y are pixel-centre coordinates
    mask_key="mask",            # array name inside the .npz
    allow_shape_mismatch=False,
    images=None,                 # directory / glob / .txt / single image: needed for the QC figure
    qc_image="filter_qc.png",   # QC png (None / "" to skip); 3 panels + sorted-iobs scatter
    qc_window_deg=1.0,           # rotation range summed per QC panel
    qc_max_px=1400,              # QC display: images binned to at most this many px on the long side
    verbose=True,
):
    """Write FILTER.HKL for the reflections of INTEGRATE.HKL that fall in the shadow mask.

    Returns a dict: n_reflections, n_excluded, excluded (bool array over the rows of
    INTEGRATE.HKL), data (the parsed INTEGRATE.HKL array), omega_deg, header values used.
    """
    header, data = read_integrate_hkl(integrate_hkl)
    n_refl = data.shape[0]

    # ---- rotation geometry
    def _num(name, override, default=None):
        if override is not None:
            return float(override)
        if name in header:
            return float(header[name])
        return default

    start = _num("STARTING_ANGLE", starting_angle) + float(angle_offset)
    osc = _num("OSCILLATION_RANGE", oscillation_range)
    if osc is None and integrate_lp:
        osc = _read_lp_value(integrate_lp, "OSCILLATION_RANGE")
    sframe = _num("STARTING_FRAME", starting_frame, 1.0)
    if start is None or osc is None:
        raise ValueError("STARTING_ANGLE / OSCILLATION_RANGE not found: pass starting_angle=, "
                         "oscillation_range= or integrate_lp=")

    # ---- mask
    src = MaskSource(mask, key=mask_key)
    n_mask, rows, cols = src.shape
    step = 360.0 / n_mask
    nx = int(float(header["NX"])) if "NX" in header else None
    ny = int(float(header["NY"])) if "NY" in header else None
    if nx and ny and (rows, cols) != (ny, nx) and not allow_shape_mismatch:
        hint = " (mask looks transposed)" if (rows, cols) == (nx, ny) else ""
        raise ValueError(f"mask frames are {rows} x {cols} (rows x cols) but INTEGRATE.HKL says "
                         f"NY x NX = {ny} x {nx}{hint}. Fix the mask or pass allow_shape_mismatch=True.")
    if "ISEG" in COLS and data.size and data[:, IC["iseg"]].max() > 1:
        print("WARNING: reflections on more than one detector segment (ISEG > 1); "
              "X/Y are assumed to be whole-detector pixel coordinates.", file=sys.stderr)

    h, k, l = data[:, 0], data[:, 1], data[:, 2]
    X, Y, Z = data[:, IC["xcal"]], data[:, IC["ycal"]], data[:, IC["zcal"]]

    # ---- frame -> omega -> candidate mask frames
    omega = start + osc * (Z - (sframe - 1.0))            # unwrapped rotation angle (deg)
    kc = np.mod(omega, 360.0) / step                      # position in mask-frame units
    half = omega_margin_deg / step
    kmin = np.floor(kc - half).astype(np.int64) + 1       # smallest integer strictly inside the window
    kmax = np.ceil(kc + half).astype(np.int64) - 1        # largest  integer strictly inside the window
    empty = kmax < kmin                                   # margin too small -> nearest mask frame
    nearest = np.rint(kc).astype(np.int64)
    kmin = np.where(empty, nearest, kmin)
    kmax = np.where(empty, nearest, kmax)
    span = kmax - kmin + 1

    # ---- (reflection, mask-array-index) pairs
    refl_i, mask_k = [], []
    for d in range(int(span.max()) if n_refl else 0):
        sel = np.nonzero(span > d)[0]
        phys = kmin[sel] + d                              # physical omega index (may be <0 or >= n_mask)
        idx = np.mod(-phys if flip_time else phys, n_mask)
        refl_i.append(sel)
        mask_k.append(idx)
    refl_i = np.concatenate(refl_i) if refl_i else np.zeros(0, np.int64)
    mask_k = np.concatenate(mask_k) if mask_k else np.zeros(0, np.int64)

    # ---- pixel windows (inclusive row/col ranges) in the (possibly flipped) mask
    c0 = np.floor(X - pixel_offset - xy_margin_px).astype(np.int64)
    c1 = np.floor(X - pixel_offset + xy_margin_px).astype(np.int64)
    r0 = np.floor(Y - pixel_offset - xy_margin_px).astype(np.int64)
    r1 = np.floor(Y - pixel_offset + xy_margin_px).astype(np.int64)
    on_det = (c1 >= 0) & (c0 < cols) & (r1 >= 0) & (r0 < rows)
    c0, c1 = np.clip(c0, 0, cols - 1), np.clip(c1, 0, cols - 1)
    r0, r1 = np.clip(r0, 0, rows - 1), np.clip(r1, 0, rows - 1)

    # ---- group pairs by mask frame, one streaming pass over the mask
    keep = on_det[refl_i]
    refl_i, mask_k = refl_i[keep], mask_k[keep]
    order = np.argsort(mask_k, kind="stable")
    refl_i, mask_k = refl_i[order], mask_k[order]
    uniq, first = np.unique(mask_k, return_index=True)
    bounds = dict(zip(uniq.tolist(), zip(first.tolist(), np.append(first[1:], mask_k.size).tolist())))

    excluded = np.zeros(n_refl, dtype=bool)
    active = []                                 # mask frames with shadow, encountered below
    for kf, frame in src.frames(uniq.tolist()):
        if flip_vertical:
            frame = frame[::-1, :]
        if flip_horizontal:
            frame = frame[:, ::-1]
        if invert_mask:
            frame = ~frame.astype(bool)
        if not frame.any():
            continue
        active.append(int(kf))
        a, b = bounds[kf]
        for j in refl_i[a:b]:
            if not excluded[j] and frame[r0[j]:r1[j] + 1, c0[j]:c1[j] + 1].any():
                excluded[j] = True

    # ---- FILTER.HKL
    dx, dy, dz = filter_box
    sel = np.nonzero(excluded)[0]
    out = np.column_stack([h[sel], k[sel], l[sel], X[sel], Y[sel], Z[sel],
                           np.full(sel.size, dx), np.full(sel.size, dy), np.full(sel.size, dz)])
    np.savetxt(filter_hkl, out, fmt="%6d %6d %6d %9.2f %9.2f %9.2f %6.2f %6.2f %6.2f")

    if verbose:
        print(f"INTEGRATE.HKL : {n_refl} reflections, omega(z) = {start:g} + {osc:g}*(z-{sframe - 1:g}) deg")
        print(f"mask          : {n_mask} frames of {step:g} deg, {rows} x {cols} px"
              f"{', time flipped' if flip_time else ''}{', vertically flipped' if flip_vertical else ''}"
              f"{', horizontally flipped' if flip_horizontal else ''}")
        print(f"excluded      : {sel.size} reflections ({100.0 * sel.size / max(n_refl, 1):.2f} %) -> {filter_hkl}")

    # ---- intensity (iobs) statistics of the excluded (filtered-out) reflections
    iobs_excluded = data[sel, IC["iobs"]] if sel.size else np.zeros(0)
    stats = dict(
        mean_iobs_excluded=float(np.mean(iobs_excluded)) if sel.size else None,
        median_iobs_excluded=float(np.median(iobs_excluded)) if sel.size else None,
        min_iobs_excluded=float(np.min(iobs_excluded)) if sel.size else None,
        max_iobs_excluded=float(np.max(iobs_excluded)) if sel.size else None,
        sum_iobs_excluded=float(np.sum(iobs_excluded)) if sel.size else 0.0,
        mean_iobs_kept=float(np.mean(data[~excluded, IC["iobs"]])) if (~excluded).any() else None,
    )
    if verbose and sel.size:
        print(f"iobs excluded : mean = {stats['mean_iobs_excluded']:.2f}, median = {stats['median_iobs_excluded']:.2f}, "
              f"min = {stats['min_iobs_excluded']:.2f}, max = {stats['max_iobs_excluded']:.2f}, "
              f"sum = {stats['sum_iobs_excluded']:.1f}")
        if stats["mean_iobs_kept"] is not None:
            print(f"iobs kept     : mean = {stats['mean_iobs_kept']:.2f} (for comparison)")

    if qc_image and images:
        picks = []
        if active:                                # percentiles of the shadow entries the data covers
            ks = np.sort(np.array(active, dtype=int))
            for pc in (25, 50, 75):
                picks.append((pc, int(ks[int(round(pc / 100.0 * (len(ks) - 1)))])))
        try:
            paths = collect_images(images)
            frames_no = [frame_number_from_name(p) for p in paths]
            plot_filter_qc(src, picks, X, Y, omega, excluded,
                           flip_vertical=flip_vertical, flip_horizontal=flip_horizontal,
                           flip_time=flip_time, invert_mask=invert_mask, step=step,
                           image_paths=paths, frame_numbers=frames_no,
                           start=start, osc=osc, sframe=sframe,
                           out_png=qc_image, window_deg=qc_window_deg, max_display_px=qc_max_px,
                           n_total=n_refl, iobs=iobs_excluded,
                           mean_excl=stats["mean_iobs_excluded"],
                           median_excl=stats["median_iobs_excluded"],
                           mean_kept=stats["mean_iobs_kept"],
                           title=f"FILTER.HKL QC - {Path(integrate_hkl).name} vs {Path(str(mask)).name}")
            if verbose:
                print(f"QC image      : {qc_image}")
        except Exception as exc:                  # the QC image is a convenience, never a reason to fail
            print(f"note: QC image skipped ({exc})", file=sys.stderr)
    elif qc_image:
        if verbose:
            print("note: no QC image (pass images=<directory of raw frames> for it)")

    return dict(n_reflections=n_refl, n_excluded=int(sel.size), excluded=excluded, data=data,
                omega_deg=omega, starting_angle=start, oscillation_range=osc, starting_frame=sframe,
                **stats)


# --------------------------------------------------------------------------- diffraction images
_IMAGE_SUFFIXES = {".npy", ".npz", ".cbf", ".tif", ".tiff", ".png"}


def _load_image(path):
    """Read one 2-D diffraction image (npy/npz native, cbf via fabio, tif/png via imageio/PIL)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
    elif suf == ".npz":
        with np.load(path) as z:
            arr = z["data" if "data" in z.files else z.files[0]]
    elif suf == ".cbf":
        try:
            import fabio
        except ImportError as exc:
            raise ImportError("reading .cbf images requires fabio (pip install fabio)") from exc
        arr = fabio.open(str(path)).data
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
        raise ValueError(f"unsupported image format {suf!r} for {path} "
                         f"(supported: {sorted(_IMAGE_SUFFIXES)})")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D image, got shape {arr.shape}")
    return arr


def collect_images(src):
    """Resolve --images to a sorted list of image paths (directory, glob, .txt list or single file)."""
    p = Path(str(src))
    if p.is_dir():
        files = sorted(q for q in p.iterdir() if q.suffix.lower() in _IMAGE_SUFFIXES and q.is_file())
    elif p.suffix.lower() == ".txt" and p.is_file():
        files = sorted(Path(l.strip()) for l in p.read_text().splitlines() if l.strip())
    elif p.is_file():
        files = [p]
    else:
        files = sorted(Path(f) for f in glob.glob(str(src)))
    if not files:
        raise ValueError(f"no images found for {str(src)!r}")
    return files


def frame_number_from_name(path):
    """Frame number encoded in the file name (last digit group of the stem)."""
    m = re.findall(r"(\d+)", Path(path).stem)
    if not m:
        raise ValueError(f"no frame number found in file name {path}")
    return int(m[-1])


def _bin_nanmean(a, s):
    """Block mean (s x s) ignoring NaN, for display; cropped to a whole number of blocks."""
    if s <= 1:
        return a
    r = (a.shape[0] // s) * s
    c = (a.shape[1] // s) * s
    b = a[:r, :c].reshape(r // s, s, c // s, s)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))


def _wrap180(a):
    """Angle difference wrapped to (-180, 180]."""
    a = np.asarray(a, dtype=float) % 360.0
    return np.where(a > 180.0, a - 360.0, a)


def _iobs_sorted_ax(ax, iobs, n_total, *, mean=None, median=None, mean_kept=None, max_points=500000):
    """Draw the sorted-iobs scatter of the filtered-out spots into an existing axis.

    X axis = rank (1..n, ascending intensity), Y axis = iobs. Negative intensities are marked
    in a different colour; horizontal lines show the mean / median of the excluded spots and,
    if given, the mean of the kept spots (mean_kept) for comparison.
    """
    iobs = np.asarray(iobs, dtype=float)
    n = iobs.size
    if n == 0:
        ax.text(0.5, 0.5, "no reflections excluded", ha="center", va="center", transform=ax.transAxes)
        return
    s = np.sort(iobs)
    if n > max_points:                                # keep the figure light
        pick = np.unique(np.linspace(0, n - 1, max_points).astype(int))
        ranks, s = np.arange(1, n + 1)[pick], s[pick]
    else:
        ranks = np.arange(1, n + 1)
    pos = s >= 0
    ax.scatter(ranks[pos], s[pos], s=4, color="tab:red", linewidths=0, label="iobs >= 0")
    if (~pos).any():
        ax.scatter(ranks[~pos], s[~pos], s=4, color="tab:blue", linewidths=0,
                   label="iobs < 0 (negative observations)")
    if mean is not None:
        ax.axhline(mean, color="tab:red", ls="--", lw=1, label=f"mean excluded = {mean:.2f}")
    if median is not None:
        ax.axhline(median, color="0.4", ls=":", lw=1, label=f"median excluded = {median:.2f}")
    if mean_kept is not None:
        ax.axhline(mean_kept, color="tab:green", ls="-.", lw=1, label=f"mean kept = {mean_kept:.2f}")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("rank (spots sorted by intensity)")
    ax.set_ylabel("iobs (counts)")
    ax.set_xlim(1, max(n, 2))
    tot = f" of {n_total}" if n_total else ""
    ax.set_title(f"Sorted intensities of the {n} reflections excluded by the shadow mask{tot}", fontsize=9)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)


# --------------------------------------------------------------------------- QC figure
def plot_filter_qc(mask_src, picks, x, y, omega, excluded, *, flip_vertical, flip_horizontal,
                   flip_time, invert_mask, step, image_paths, frame_numbers, start, osc, sframe,
                   out_png, window_deg=1.0, max_display_px=1400, n_total=None,
                   iobs=None, mean_excl=None, median_excl=None, mean_kept=None, dpi=130, title=""):
    """QC figure of the FILTER.HKL selection: three diffraction images with the removed spots.

    picks: [(percentile, mask_frame)] -- the mask frames to show (percentiles of the shadow
    travel).  Top row: one panel per pick, the mean of the images whose rotation angle is within
    window_deg / 2 of the pick's angle (repeating every 360 deg), the shadowed region of that
    mask entry outlined, and a red circle on every excluded reflection that falls on those
    images. Bottom row: the sorted intensity scatter of ALL excluded reflections.
    x, y: XCAL / YCAL of every reflection; omega: its rotation angle (deg, XDS axis);
    excluded: bool array over the reflections.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    n = len(picks)
    rows_img, cols_img = mask_src.shape[1], mask_src.shape[2]
    sb = max(1, int(math.ceil(max(rows_img, cols_img) / float(max_display_px))))
    omega_img = (start + osc * (np.asarray(frame_numbers, dtype=float) - sframe + 0.5)) % 360.0
    ex_x, ex_y, ex_om = x[excluded], y[excluded], omega[excluded] % 360.0
    c_mask = "#00d5ff"
    fig = plt.figure(figsize=(6.4 * n, 6.4 * rows_img / cols_img + 4.4))
    gs = gridspec.GridSpec(2, n, height_ratios=[rows_img / cols_img, 0.75], hspace=0.28, wspace=0.12)
    frames = dict(mask_src.frames(sorted(k for _, k in picks)))
    half = float(window_deg) / 2.0
    for col, (pc, kf) in enumerate(picks):
        ax = fig.add_subplot(gs[0, col])
        am = float(kf) * step % 360.0
        om_pick = (-am if flip_time else am) % 360.0
        sel_img = np.nonzero(np.abs(_wrap180(omega_img - om_pick)) <= half)[0]
        if not sel_img.size:                            # nothing in the window: the nearest image
            sel_img = np.array([int(np.argmin(np.abs(_wrap180(omega_img - om_pick))))])
        acc = None
        for j in sel_img:
            im = _load_image(image_paths[j])
            im[im < 0] = np.nan                       # -1 = bad pixel / masked: not averaged
            if im.shape != (rows_img, cols_img):
                raise ValueError(f"image {Path(image_paths[j]).name} is {im.shape}, "
                                 f"expected {(rows_img, cols_img)} (rows x cols)")
            acc = im if acc is None else acc + np.where(np.isfinite(im), im, 0.0)
        mx = np.nanmax(acc) if acc is not None and np.isfinite(acc).any() else 1.0
        img = (acc / len(sel_img)) if acc is not None else np.full((rows_img, cols_img), np.nan)
        disp = _bin_nanmean(img, sb)
        fin = disp[np.isfinite(disp)]
        vmax = max(float(np.percentile(fin, 99.5)) if fin.size else 1.0, 1e-6)
        dr, dc = disp.shape[0] * sb, disp.shape[1] * sb
        ax.imshow(disp, cmap="gray", vmin=0.0, vmax=vmax, origin="lower",
                  extent=(-0.5, dc - 0.5, -0.5, dr - 0.5), interpolation="nearest")
        frame = np.asarray(frames[kf], dtype=bool)
        if flip_vertical:
            frame = frame[::-1, :]
        if flip_horizontal:
            frame = frame[:, ::-1]
        if invert_mask:
            frame = ~frame
        bm = _bin_nanmean(frame.astype(float), sb)
        if bm.size and bm.min() < 0.5 <= bm.max():
            xs = (np.arange(bm.shape[1]) + 0.5) * sb - 0.5
            ys = (np.arange(bm.shape[0]) + 0.5) * sb - 0.5
            ax.contour(xs, ys, bm, levels=[0.5], colors=[c_mask], linewidths=1.4, alpha=0.8)
        sel_spots = np.abs(_wrap180(ex_om - om_pick)) <= half + abs(osc) / 2.0 if ex_om.size else np.zeros(0, bool)
        if sel_spots.any():
            ax.plot(ex_x[sel_spots], ex_y[sel_spots], "o", mfc="none", mec="red",
                    ms=max(5.0, 3.0 * sb), mew=1.2, ls="none")
        ax.set_xlim(-0.5, dc - 0.5)
        ax.set_ylim(-0.5, dr - 0.5)
        ax.set_title(f"{pc:g}th percentile of shadow travel - entry {kf} (omega {am:.1f} deg)\n"
                     f"{len(sel_img)} image{'s' if len(sel_img) != 1 else ''} summed"
                     f" (+-{window_deg:g} deg); {int(sel_spots.sum())} excluded spot"
                     f"{'s' if int(sel_spots.sum()) != 1 else ''} circled", fontsize=9)
        if col == 0:
            ax.set_ylabel("Y (pixel)")
        ax.set_xlabel("X (pixel)")
        ax.legend(handles=[Line2D([0], [0], color="red", lw=0, marker="o", mfc="none", ms=6,
                                 label="excluded (in FILTER.HKL)"),
                           Line2D([0], [0], color=c_mask, lw=1.4, label="shadowed mask entry")],
                  loc="upper left", fontsize=8, framealpha=0.85)
    ax_io = fig.add_subplot(gs[1, :])
    _iobs_sorted_ax(ax_io, iobs if iobs is not None else np.zeros(0), n_total,
                    mean=mean_excl, median=median_excl, mean_kept=mean_kept)
    fig.suptitle(title or "FILTER.HKL QC: excluded reflections on the diffraction images", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.07, wspace=0.10, hspace=0.55)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- command line
def _main(argv=None):
    p = argparse.ArgumentParser(description="Create an XDS FILTER.HKL from INTEGRATE.HKL and a shadow mask (.npz).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("integrate_hkl")
    p.add_argument("mask", help="mask .npz (array 'mask', frames x rows x cols, True = shadowed) or .npy")
    p.add_argument("-o", "--output", default="FILTER.HKL")
    p.add_argument("--flip-vertical", action="store_true", help="flip the mask upside down")
    p.add_argument("--flip-horizontal", action="store_true", help="mirror the mask left/right")
    p.add_argument("--flip-time", action="store_true", help="mask k is the shadow at -k deg (mask 350 = 10 deg)")
    p.add_argument("--invert-mask", action="store_true", help="mask has True = valid pixel")
    p.add_argument("--starting-angle", type=float)
    p.add_argument("--oscillation-range", type=float)
    p.add_argument("--starting-frame", type=float)
    p.add_argument("--angle-offset", type=float, default=0.0, help="offset (deg) added to STARTING_ANGLE (before the time flip)")
    p.add_argument("--integrate-lp", help="INTEGRATE.LP, used if OSCILLATION_RANGE is missing in the .HKL header")
    p.add_argument("--omega-margin", type=float, default=0.5, help="omega play in degrees")
    p.add_argument("--xy-margin", type=float, default=2.0, help="detector play in pixels")
    p.add_argument("--filter-box", type=float, nargs=3, default=(0.5, 0.5, 0.5), metavar=("DX", "DY", "DZ"))
    p.add_argument("--pixel-offset", type=float, default=0.0, help="0.5 if X,Y are pixel-centre coordinates")
    p.add_argument("--images", default=None,
                   help="directory, glob, .txt list or single image of the raw frames (for the QC figure)")
    p.add_argument("--qc-image", default="filter_qc.png",
                   help="QC png: images at the 25/50/75th percentile of the shadow travel with the excluded "
                        "spots circled and the sorted-iobs scatter of all excluded spots")
    p.add_argument("--no-qc-image", action="store_true", help="do not write the QC png")
    p.add_argument("--qc-window-deg", type=float, default=1.0, help="rotation range summed per QC panel (deg)")
    p.add_argument("--qc-max-px", type=int, default=1400, help="QC display: images binned to at most this many px")
    a = p.parse_args(argv)
    make_filter_hkl(a.integrate_hkl, a.mask, a.output,
                    flip_vertical=a.flip_vertical, flip_horizontal=a.flip_horizontal,
                    flip_time=a.flip_time, invert_mask=a.invert_mask,
                    starting_angle=a.starting_angle, oscillation_range=a.oscillation_range,
                    starting_frame=a.starting_frame, angle_offset=a.angle_offset,
                    integrate_lp=a.integrate_lp,
                    omega_margin_deg=a.omega_margin, xy_margin_px=a.xy_margin,
                    filter_box=tuple(a.filter_box), pixel_offset=a.pixel_offset,
                    images=a.images,
                    qc_image=(None if a.no_qc_image else a.qc_image),
                    qc_window_deg=a.qc_window_deg, qc_max_px=a.qc_max_px)


if __name__ == "__main__":
    _main()
