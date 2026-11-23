#!/usr/bin/env python3
"""
xds_shadow_refine.py -- per-image refinement of a theoretical shadow mask (npz -> refined npz)

Input  : the per-degree mask npz written by the goniometer / pin-shadow visualizer
         (array 'mask', (n_entries, rows, cols) bool, True = shadowed; 6.5 GB for an Eiger2X 16M,
         so it is streamed, never loaded whole) and the raw diffraction images.
Output : a refined npz with the same layout (array 'mask' + the metadata arrays of the input +
         per-entry statistics), written as a compressed zip64 stream.

How it works
------------
For every mask entry the shadow boundary is walked as a curve.  Along it a rect_width x rect_length
rectangle (long axis = curve normal, straddling the curve) is integrated from the raw images; the
low -> high background profile is fitted with a sigmoid and the mask edge is moved to where the high
plateau is reached (edge_frac).  Compared with a plain per-image fit, this version

  * puts the rectangle of every image at the edge position expected for the EXACT omega of that image
    (interpolated from the neighbouring mask entries), so the 1-degree sampling of the mask no longer
    limits the fit even where the shadow moves ~100 px/deg, and it adds the aligned profiles of all
    images of an entry (shift-and-add), which is what makes low backgrounds usable;
  * never lets rectangles hang over the detector edge (invalid pixels are NaN, not replicated) and holds
    the shift of the last clean sample (the "edge anchor", about half a rectangle from the border) out to
    the detector edge instead of extrapolating;
  * rejects fits with the wrong polarity (the shadow must be the dark side), outliers along the curve,
    and edges that hit the search limit, and handles closed (pin-cone) boundaries.

Orientation of the starting mask
--------------------------------
  --flip-vertical   (alias --flip-y)  reverse the rows of every mask frame
  --flip-horizontal (alias --flip-x)  reverse the columns
  --transpose                         swap rows/cols (also done automatically if INTEGRATE.HKL says the
                                      mask is NX x NY instead of NY x NX)
  --flip-time                         mask entry k holds omega = -k deg, i.e. the XDS angle is negated
                                      before it is mapped to an entry (mask 350 <-> 10 deg)
  --angle-offset DEG                  added to STARTING_ANGLE, i.e. omega_xds = START + OFFSET + osc*...
                                      (applied BEFORE the time flip)
  Order: transpose, vertical, horizontal.  The same flags are what xds_shadow_filter.py needs
  later (there --time-offset acts AFTER the flip, so its sign is opposite to --angle-offset when
  --flip-time is used).  By default the refined npz stays in the orientation and time axis of the input,
  so those flags remain valid for it; --output-orientation image writes the flipped frames instead.

Image angle: omega of an image is taken at the centre of the exposure,
  omega = START + OFFSET + osc * (frame - STARTING_FRAME + 0.5)      (--no-half-frame: end of exposure)

Usage
-----
  python xds_shadow_refine.py refine INTEGRATE.HKL --images DIR --mask mask_omscan.npz \\
      --flip-vertical --flip-time --angle-offset 0.0 \\
      --masks-out refined.npz --report refine.json --jobs 4
  python xds_shadow_refine.py tune-offset INTEGRATE.HKL --images DIR --mask mask_omscan.npz \\
      --flip-vertical --flip-time --angle-offset 0.0 --out tune.png       # judge / tune the angle offset by eye
  python xds_shadow_refine.py refine-debug mask.npz frame_0001.cbf [more frames] --entry 180 \\
      -o refined.npz                                                     # refined entry written in place + refined_qc.png
  python xds_shadow_refine.py mask-images INTEGRATE.HKL --images DIR --mask refined.npz \\
      --flip-vertical --flip-time --angle-offset 0.0                     # images_masked/ = symlinks + masked copies


Only numpy + scipy are required; h5py (h5), fabio (cbf), imageio/PIL (tif/png) and matplotlib
(png figures) are optional.  Images must be one 2-D frame per file (names contain the frame number); .gz / .bz2 compressed files
(e.g. frame_0001.cbf.gz) are read transparently.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import shutil
import struct
import time
import zipfile
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

IMAGE_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5", ".cbf", ".tif", ".tiff", ".png"}
COMPRESSION_SUFFIXES = {".gz", ".bz2"}          # image.cbf.gz, image.cbf.bz2, ... are decompressed transparently


def _image_suffix(path):
    """Lower-case image suffix with a trailing compression suffix removed ('x.cbf.gz' -> '.cbf')."""
    p = Path(path)
    if p.suffix.lower() in COMPRESSION_SUFFIXES:
        p = Path(p.stem)
    return p.suffix.lower()


def _image_stem(path):
    """File name without compression and image suffix ('frame_0001.cbf.gz' -> 'frame_0001')."""
    p = Path(path)
    if p.suffix.lower() in COMPRESSION_SUFFIXES:
        p = Path(p.stem)
    return p.stem


def _load_compressed(path):
    """Decompress a .gz/.bz2 image into a temporary file that keeps its inner name, read it with the normal
    reader for that format (so .cbf.gz goes through fabio exactly like .cbf), and delete the copy."""
    import bz2
    import gzip
    import shutil
    import tempfile
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else bz2.open
    with tempfile.TemporaryDirectory(prefix="xdsrefine_") as tmp:
        inner = Path(tmp) / path.stem                   # keeps the inner suffix, e.g. frame_0001.cbf
        with opener(path, "rb") as src, open(inner, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
        return load_image(inner)

_FRAME_RE = re.compile(r"(\d+)")


# --------------------------------------------------------------------------- image I/O
def load_image(path):
    """Read a 2-D diffraction image as a float64 array.

    .npy/.npz are native; .h5/.hdf5 need h5py, .cbf needs fabio, .tif/.tiff need
    imageio or PIL (first 2-D dataset / first page is used for multi-dataset files).
    Any of these may be gzip/bzip2 compressed (.cbf.gz, .tif.bz2, ...); they are decompressed on the fly.
    """
    path = Path(path)
    if path.suffix.lower() in COMPRESSION_SUFFIXES:
        return _load_compressed(path)
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
    if arr.dtype.kind == "u" and arr.dtype.itemsize > 1:
        # a -1 "bad pixel" sentinel in detector data read through an unsigned
        # integer dtype wraps to the dtype maximum (e.g. 4294967295 for uint32):
        # map it back to -1. uint8 is left alone (255 is a legitimate white in
        # a black/white mask png).
        arr = np.where(arr == np.iinfo(arr.dtype).max, -1.0, arr.astype(np.float64))
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


def save_image(path, arr):
    """Write a 2-D image in the format given by its suffix (npy/npz/cbf/tif/tiff/png; .gz/.bz2 wrapped).

    The inverse of load_image: .npy/.npz are native, .cbf needs fabio, .tif/.tiff/.png need imageio
    or PIL; compressed names keep the inner format (frame_0001.cbf.gz is written gzip-compressed)."""
    path = Path(path)
    arr = np.asarray(arr)
    suf = path.suffix.lower()
    if suf in COMPRESSION_SUFFIXES:
        import bz2
        import gzip
        import tempfile
        with tempfile.TemporaryDirectory(prefix="xdsrefine_") as tmp:
            inner = Path(tmp) / path.stem                   # keeps the inner suffix, e.g. frame_0001.cbf
            save_image(inner, arr)
            payload = inner.read_bytes()
        opener = gzip.open if suf == ".gz" else bz2.open
        with opener(path, "wb") as dst:
            dst.write(payload)
        return str(path)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D image, got shape {arr.shape}")
    if suf == ".npy":
        np.save(path, arr)
    elif suf == ".npz":
        np.savez(path, arr)
    elif suf == ".cbf":
        try:
            import fabio
        except ImportError as exc:
            raise ImportError("writing .cbf images requires fabio (pip install fabio)") from exc
        fabio.cbfimage.CbfImage(data=arr).write(str(path))
    elif suf in (".tif", ".tiff", ".png"):
        try:
            import imageio.v3 as iio
            iio.imwrite(path, arr)
        except ImportError:
            try:
                from PIL import Image
            except ImportError as exc:
                raise ImportError("writing .tif/.png images requires imageio or PIL") from exc
            Image.fromarray(arr).save(path)
    else:
        raise ValueError(f"unsupported image format {suf!r} for {path} (supported: .npy/.npz/.cbf/.tif/.tiff/.png)")
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


def prepare_image(img, bad_value=-1.0, edge_margin=0, spot_clip="auto"):
    """float32 copy of an image in which every unusable pixel is NaN.

    NaN = defective (== bad_value / non-finite), the outermost `edge_margin` rows/columns, and
    (spot_clip) pixels far above the background, which are Bragg peaks / ice and would wreck the
    low-count profile fits.  spot_clip: 'auto' (median + 12 sqrt(median + 1) + 20), a number of counts,
    or None/0/'off' to disable.
    """
    img = np.asarray(img)
    bad = (img == bad_value) | ~np.isfinite(img)
    out = img.astype(np.float32)
    if spot_clip not in (None, 0, 0.0, False, "off", "none"):
        if spot_clip == "auto":
            sample = img[::7, ::7][~bad[::7, ::7]]
            med = float(np.median(sample)) if sample.size else 0.0
            thr = med + 12.0 * math.sqrt(max(med, 0.0) + 1.0) + 20.0
        else:
            thr = float(spot_clip)
        with np.errstate(invalid="ignore"):
            bad |= img > thr
    m = int(edge_margin)
    if m > 0:
        bad[:m, :] = True
        bad[-m:, :] = True
        bad[:, :m] = True
        bad[:, -m:] = True
    out[bad] = np.nan
    return out


def collect_images(src):
    """Resolve --images to a sorted list of image paths.

    src may be a directory (scanned for known image suffixes), a glob pattern,
    a single image file, or a .txt file with one path per line.
    """
    src = str(src)
    p = Path(src)
    if p.is_dir():
        files = sorted(q for q in p.iterdir() if _image_suffix(q) in IMAGE_SUFFIXES and q.is_file())
    elif p.suffix.lower() == ".txt" and p.is_file():
        files = sorted(Path(line.strip()) for line in p.read_text().splitlines() if line.strip())
    elif p.is_file():
        files = [p]
    else:
        files = sorted(Path(f) for f in glob.glob(src))
    if not files:
        raise ValueError(f"no images found for {src!r}")
    return files


def exclude_files(paths, *others):
    """Drop our own outputs / the mask from an image list (a directory scan would otherwise pick up .npz/.png files)."""
    skip = set()
    for o in others:
        if o:
            try:
                skip.add(Path(o).resolve())
            except OSError:
                pass
    return [p for p in paths if Path(p).resolve() not in skip]


def frame_number_from_name(path, pattern=None):
    """Frame number encoded in the file name (last digit group of the stem by default)."""
    rx = re.compile(pattern) if pattern else _FRAME_RE
    m = rx.findall(_image_stem(path))
    if not m:
        raise ValueError(f"no frame number found in file name {path}")
    return int(m[-1])


# --------------------------------------------------------------------------- angles
def image_omega(frame_no, starting_angle, oscillation_range, starting_frame=1.0, half_frame=True):
    """XDS rotation angle of an image (deg), at the CENTRE of its exposure by default:
    omega = STARTING_ANGLE + OSCILLATION_RANGE * (frame - STARTING_FRAME + 0.5).
    half_frame=False gives the end of the exposure (frame - STARTING_FRAME + 1)."""
    z = float(frame_no) - float(starting_frame) + (0.5 if half_frame else 1.0)
    return float(starting_angle) + float(oscillation_range) * z


def mask_angle(omega_xds, flip_time=False):
    """Angle on the mask's own time axis: -omega when the mask time axis is reversed."""
    return -float(omega_xds) if flip_time else float(omega_xds)


def assign_entries(omega_mask, nz, deg_per_entry):
    """(entry index, deviation in deg) of each mask-axis angle: nearest entry, |dev| <= deg_per_entry/2."""
    w = np.asarray(omega_mask, dtype=float) / deg_per_entry
    k = np.rint(w)
    return np.mod(k, nz).astype(np.int64), (w - k) * deg_per_entry


def omega_to_entry(omega_deg, nz, deg_per_entry):
    """Nearest mask entry (index into the nz axis) for a rotation angle."""
    return int(np.mod(np.round(np.mod(omega_deg, 360.0) / deg_per_entry), nz))


# --------------------------------------------------------------------------- mask orientation
@dataclass(frozen=True)
class Orientation:
    """Maps a frame of the starting mask into the image orientation: transpose, then vertical, then horizontal flip."""
    transpose: bool = False
    flip_vertical: bool = False
    flip_horizontal: bool = False

    def apply(self, a):
        if self.transpose:
            a = a.T
        if self.flip_vertical:
            a = a[::-1, :]
        if self.flip_horizontal:
            a = a[:, ::-1]
        return np.ascontiguousarray(a)

    def invert(self, a):
        if self.flip_horizontal:
            a = a[:, ::-1]
        if self.flip_vertical:
            a = a[::-1, :]
        if self.transpose:
            a = a.T
        return np.ascontiguousarray(a)

    def shape(self, rows, cols):
        return (cols, rows) if self.transpose else (rows, cols)


# --------------------------------------------------------------------------- npz mask I/O (streamed)
class MaskStream:
    """Sequential frame access to a mask (n_entries, rows, cols).

    Accepts a numpy array, a .npy file (memory-mapped) or the .npz written by the visualizer.  For .npz the
    big array is read as a stream, one frame at a time (compressed zip64 entries cannot be memory mapped).
    """

    def __init__(self, source, key="mask"):
        self._arr = None
        self._path = None
        self.key = key
        if isinstance(source, np.ndarray):
            self._arr = source
        else:
            path = Path(source)
            if path.suffix.lower() == ".npy":
                self._arr = np.load(path, mmap_mode="r")
            else:
                self._path = path
                with zipfile.ZipFile(path) as zf:
                    if key + ".npy" not in zf.namelist():
                        raise KeyError(f"{path}: no array named {key!r} (available: "
                                       f"{[n[:-4] for n in zf.namelist() if n.endswith('.npy')]})")
                    with zf.open(key + ".npy") as fh:
                        self._read_header(fh)
        if self._arr is not None:
            if self._arr.ndim != 3:
                raise ValueError(f"mask must be 3-D (entries, rows, cols), got {self._arr.shape}")
            self.shape = tuple(self._arr.shape)
            self._dtype = self._arr.dtype

    def _read_header(self, fh):
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        if fortran or len(shape) != 3:
            raise ValueError(f"unsupported mask array (shape {shape}, fortran_order={fortran})")
        self.shape, self._dtype = tuple(shape), dtype

    def iter_frames(self):
        """Yield (index, 2-D bool array) for every entry, in file order."""
        n, rows, cols = self.shape
        if self._arr is not None:
            for k in range(n):
                yield k, np.asarray(self._arr[k]).astype(bool)
            return
        nbytes = rows * cols * self._dtype.itemsize
        with zipfile.ZipFile(self._path) as zf, zf.open(self.key + ".npy") as fh:
            self._read_header(fh)
            for k in range(n):
                buf = fh.read(nbytes)
                if len(buf) != nbytes:
                    raise IOError("mask array ended early (truncated file?)")
                yield k, np.frombuffer(buf, dtype=self._dtype).reshape(rows, cols).astype(bool)

    def get_frames(self, indices):
        """{index: 2-D bool array} for the requested entries (one streaming pass, stops after the last one)."""
        want = sorted({int(i) % self.shape[0] for i in indices})
        out = {}
        if not want:
            return out
        for k, fr in self.iter_frames():
            if k in want:
                out[k] = fr
            if k >= want[-1]:
                break
        return out

    def small_arrays(self):
        """Every other array stored next to the mask (omega_deg, kappa_deg, pixel_size_mm, ...)."""
        if self._path is None:
            return {}
        out = {}
        with np.load(self._path) as z:
            for name in z.files:
                if name != self.key:
                    out[name] = z[name]
        return out


def _npy_header(descr, shape):
    shp = f"({shape[0]},)" if len(shape) == 1 else "(" + ", ".join(str(s) for s in shape) + ")"
    hdr = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shp}, }}"
    hdr += " " * ((64 - ((10 + len(hdr) + 1) % 64)) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode("latin1")


class NpzStreamWriter:
    """Write an npz whose 'mask' array (n, rows, cols) bool is produced frame by frame (zip64, deflate)."""

    def __init__(self, path, n, rows, cols, compresslevel=4):
        self.n, self.rows, self.cols, self.count = int(n), int(rows), int(cols), 0
        self.zf = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel,
                                  allowZip64=True)
        self.fh = self.zf.open("mask.npy", "w", force_zip64=True)
        self.fh.write(_npy_header("|b1", (self.n, self.rows, self.cols)))

    def write(self, frame):
        frame = np.asarray(frame)
        if frame.shape != (self.rows, self.cols):
            raise ValueError(f"frame shape {frame.shape} != {(self.rows, self.cols)}")
        self.fh.write(np.ascontiguousarray(frame, dtype=np.bool_).tobytes())
        self.count += 1

    def close(self, extra=None):
        if self.count != self.n:
            raise RuntimeError(f"wrote {self.count} of {self.n} frames")
        self.fh.close()
        for name, arr in (extra or {}).items():
            buf = io.BytesIO()
            np.save(buf, np.asarray(arr))
            self.zf.writestr(name + ".npy", buf.getvalue())
        self.zf.close()


# --------------------------------------------------------------------------- curve geometry
def boundary_pixels(mask, outside_is_shadow=True):
    """One-pixel skin of the True (shadowed) region; for a 1-px thin region the region itself.

    With outside_is_shadow the area beyond the detector counts as shadow, so a shadow that runs
    into the detector edge does NOT get a spurious boundary along the edge: its boundary is just the
    curve, ending at the border."""
    m = np.asarray(mask, dtype=bool)
    if not m.any() or (outside_is_shadow and m.all()):
        return np.zeros((0, 2), dtype=np.int64)
    eroded = ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool),
                                    border_value=1 if outside_is_shadow else 0)
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


def _is_closed(chain, max_gap):
    return len(chain) >= 30 and float(np.hypot(*(chain[0] - chain[-1]))) <= 2.0 * max_gap


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


def smooth_curve(pts, sigma, closed=False):
    """Gaussian smoothing of a polyline (sigma in samples); used for the tangent only, never for positions."""
    if sigma <= 0 or len(pts) < 5:
        return pts
    mode = "wrap" if closed else "nearest"
    return np.column_stack([ndimage.gaussian_filter1d(pts[:, 0], sigma, mode=mode),
                            ndimage.gaussian_filter1d(pts[:, 1], sigma, mode=mode)])


def tangents_normals(pts, closed=False, smooth_sigma=0.0):
    """Unit tangents (central differences of the smoothed curve) and normals (left perpendicular), (y, x) order."""
    p = smooth_curve(np.asarray(pts, dtype=float), smooth_sigma, closed)
    t = np.zeros_like(p)
    if closed:
        t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    else:
        t[1:-1] = p[2:] - p[:-2]
        t[0] = p[1] - p[0]
        t[-1] = p[-1] - p[-2]
    norm = np.hypot(t[:, 0], t[:, 1])
    norm[norm == 0] = 1.0
    t = t / norm[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return t, n


def bright_side_sign(mask, pts, n, probe=3.0, smooth=9):
    """+1 where the bright (non-shadowed) side is along +normal, -1 where along -normal, from the mask itself.

    The detector edge is 'unknown' (not shadow), so samples next to the border take the answer
    from the in-detector side or from their neighbours along the curve."""
    ny, nx = mask.shape

    def look(s):
        yy = np.rint(pts[:, 0] + s * probe * n[:, 0]).astype(np.int64)
        xx = np.rint(pts[:, 1] + s * probe * n[:, 1]).astype(np.int64)
        inside = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        val = np.zeros(len(pts), dtype=bool)
        val[inside] = mask[yy[inside], xx[inside]]
        return val & inside, (~val) & inside      # (dark, bright)

    dp, bp = look(+1.0)
    dm, bm = look(-1.0)
    pos = (bp & ~bm) | (dm & ~dp)
    neg = (bm & ~bp) | (dp & ~dm)
    g = np.where(pos & ~neg, 1.0, np.where(neg & ~pos, -1.0, 0.0))
    nzi = np.nonzero(g)[0]
    if nzi.size == 0:
        return g
    if nzi.size < len(g):
        idx = np.arange(len(g))
        r = np.clip(np.searchsorted(nzi, idx), 0, nzi.size - 1)
        l = np.clip(r - 1, 0, nzi.size - 1)
        near = np.where(np.abs(nzi[l] - idx) <= np.abs(nzi[r] - idx), nzi[l], nzi[r])
        g = g[near]
    if smooth > 1:
        g = np.sign(ndimage.median_filter(g, size=smooth, mode="nearest"))
    return g


# --------------------------------------------------------------------------- edge profile fit
def sigmoid(v, a, b, v0, s):
    return a + b / (1.0 + np.exp(-np.clip((v - v0) / s, -500.0, 500.0)))


def fit_edge_profile(profile, v, min_amp=0.0, reweight=True):
    """Fit a + b / (1 + exp(-(v - v0)/s)) to an integrated edge profile.

    NaN entries (too few valid pixels) are dropped. Returns dict(a, b, v0, s)
    or None when the profile has no usable edge (too few points, flat, or
    contrast below 4x the Poisson noise floor / min_amp).  With reweight the
    least-squares weights of a second pass come from the fitted model instead of the
    noisy data (Pearson chi-square, much closer to the Poisson maximum likelihood at low counts).
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
        popt, _ = curve_fit(sigmoid, v, p, p0=p0, sigma=sigma, bounds=bounds, maxfev=5000)
        if reweight:
            model = sigmoid(v, *popt)
            sigma2 = np.sqrt(np.clip(model, 0.5, None))
            popt, _ = curve_fit(sigmoid, v, p, p0=popt, sigma=sigma2, bounds=bounds, maxfev=5000)
    except Exception:
        return None
    a, b, v0, s = (float(x) for x in popt)
    if not np.isfinite([a, b, v0, s]).all() or abs(b) < min_amp:
        return None
    return dict(a=a, b=b, v0=v0, s=s)


# --------------------------------------------------------------------------- sampling helpers
@lru_cache(maxsize=16)
def _rect_grid(width, length):
    u = np.arange(width) - (width - 1) / 2.0
    v = np.arange(length) - (length - 1) / 2.0
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return uu, vv, v


def _logit(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _interp_gaps(idx, xp, fp, order=2, window=6):
    """Interpolate values at `idx` between known (xp, fp) samples along the curve.

    Inside the fitted range: order >= 2 uses a local quadratic least-squares fit over the `window` nearest
    known points (the edge of a spherical object is curved), clamped to the range of those points so it
    cannot overshoot; order 1 is plain linear.  Outside the fitted range the end value is HELD (np.interp
    semantics): the old quadratic extrapolation at the chain ends is gone.
    """
    idx = np.atleast_1d(np.asarray(idx, dtype=float))
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    if xp.size == 0:
        return np.zeros(idx.shape)
    if xp.size == 1:
        return np.full(idx.shape, fp[0])
    out = np.interp(idx, xp, fp)
    if order >= 2 and xp.size >= 3:
        for k in np.nonzero((idx >= xp[0]) & (idx <= xp[-1]))[0]:
            x = idx[k]
            sel = np.argsort(np.abs(xp - x))[: min(window, xp.size)]
            xs, ys = xp[sel], fp[sel]
            val = np.polyval(np.polyfit(xs, ys, 2 if xs.size >= 3 else 1), x)
            out[k] = np.clip(val, ys.min(), ys.max())
    return out


def _paint(mask, py, px, ny, nx, shift, fill):
    """Set the pixels on the normal segment from the curve point (t=0, included)
    to the shifted edge (t=shift)."""
    if shift == 0.0:
        return
    n = max(1, int(abs(shift) / 0.5) + 1)
    ts = np.linspace(0.0, shift, n + 1)
    ys = np.rint(py + ts * ny).astype(int)
    xs = np.rint(px + ts * nx).astype(int)
    ok = (ys >= 0) & (ys < mask.shape[0]) & (xs >= 0) & (xs < mask.shape[1])
    mask[ys[ok], xs[ok]] = fill


def _neighbor_mask(nb_mask):
    """The neighbouring entry's mask, or None if it is missing or has no shadow."""
    if nb_mask is None or not np.asarray(nb_mask).any():
        return None
    return np.asarray(nb_mask, dtype=bool)


def _expansion(nb_mask, pts, n, bright, rmax, step=0.5):
    """How far (px) the shadow of the neighbouring entry reaches beyond the boundary points `pts` of this
    entry, measured ALONG THE NORMAL (a nearest-boundary distance would pick up other edges at corners):
    positive = grew toward the bright side, negative = retreated, NaN = no transition within rmax."""
    if nb_mask is None:
        return None
    ny, nx = nb_mask.shape
    ts = np.arange(0.0, rmax + step, step)
    d = bright[:, None] * n

    def ray(sign):
        yy = np.clip(np.rint(pts[:, 0, None] + sign * ts[None, :] * d[:, 0, None]).astype(np.int64), 0, ny - 1)
        xx = np.clip(np.rint(pts[:, 1, None] + sign * ts[None, :] * d[:, 1, None]).astype(np.int64), 0, nx - 1)
        return nb_mask[yy, xx]

    toward_bright, toward_dark = ray(+1.0), ray(-1.0)
    inside0 = toward_bright[:, 0]
    out = np.full(len(pts), np.nan)
    first_clear = np.argmax(~toward_bright, axis=1)
    first_shadow = np.argmax(toward_dark, axis=1)
    sel = inside0 & (~toward_bright).any(axis=1)
    out[sel] = ts[first_clear[sel]]
    sel = (~inside0) & toward_dark.any(axis=1)
    out[sel] = -ts[first_shadow[sel]]
    return out


def edge_isolated(mask, pts, t, n, width, length, iso, tol):
    """True where the mask edge under the rectangle is ONE straight edge: no second edge within `iso` px
    along the normal, and the edge positions on the two long sides of the rectangle agree with the centre line
    within `tol` px (rejects rectangles at corners and at thin features)."""
    ny, nx = mask.shape
    vs = np.arange(-length / 2.0, length / 2.0 + 1e-9, 1.0)
    mid = vs[:-1] + 0.5
    inz = np.abs(mid) <= iso
    lines = []
    for u in (0.0, -width / 2.0, +width / 2.0):
        yy = np.clip(np.rint(pts[:, 0, None] + u * t[:, 0, None] + vs[None, :] * n[:, 0, None]).astype(np.int64), 0, ny - 1)
        xx = np.clip(np.rint(pts[:, 1, None] + u * t[:, 1, None] + vs[None, :] * n[:, 1, None]).astype(np.int64), 0, nx - 1)
        lines.append(mask[yy, xx].astype(np.int8))
    ok = np.ones(len(pts), dtype=bool)
    main0 = None
    for val in lines:
        tr = (np.diff(val, axis=1) != 0) & inz[None, :]
        has = tr.any(axis=1)
        dist = np.where(tr, np.abs(mid)[None, :], np.inf)
        main = mid[np.argmin(dist, axis=1)]
        other = (tr & (np.abs(mid[None, :] - main[:, None]) > 3.0)).any(axis=1)
        ok &= has & ~other
        if main0 is None:
            main0 = main
        else:
            ok &= np.abs(main - main0) <= tol
    return ok


def _prior_shift(a_prev, a_next, d, dpe, bright, max_prior):
    """Expected edge displacement along +normal (px) at an angle `d` deg away from this entry,
    linearly interpolated toward the neighbouring entry on that side (the other neighbour mirrored if missing)."""
    if d >= 0:
        a = a_next if a_next is not None else (-a_prev if a_prev is not None else None)
    else:
        a = a_prev if a_prev is not None else (-a_next if a_next is not None else None)
    if a is None or abs(d) < 1e-9:
        return np.zeros_like(bright)
    return np.clip(bright * np.nan_to_num(a, nan=0.0) * (abs(d) / dpe), -max_prior, max_prior)


def _finalize_shifts(raw, fitted, closed, params):
    """Fitted shifts -> a shift for every sample of the curve.

    1. robust outlier rejection against the running median of the neighbouring fits,
    2. gaps between fits are interpolated, the two ENDS of an open curve (where it runs into the detector
       edge) are held at the median of the nearest `anchor_n` fits (the edge anchor),
    3. a median filter smooths the result.
    Returns (shifts, accepted-mask, n_outliers, n_held_samples)."""
    n = len(raw)
    idx = np.arange(n)
    ok = fitted.copy()
    n_out = 0
    w = int(params.outlier_window)
    if params.outlier_sigma > 0 and int(ok.sum()) >= 5:
        vals = np.where(ok, raw, np.nan)
        rej = np.zeros(n, dtype=bool)
        for i in np.nonzero(ok)[0]:
            ids = np.arange(i - w, i + w + 1)
            ids = ids[ids != i]
            ids = ids % n if closed else ids[(ids >= 0) & (ids < n)]
            nb = vals[ids]
            nb = nb[np.isfinite(nb)]
            if nb.size < 3:
                continue
            med = float(np.median(nb))
            mad = max(1.4826 * float(np.median(np.abs(nb - med))), params.outlier_floor)
            if abs(raw[i] - med) > params.outlier_sigma * mad:
                rej[i] = True
        n_out = int(rej.sum())
        ok &= ~rej
    fi = idx[ok]
    if fi.size == 0:
        return np.zeros(n), ok, n_out, 0
    n_held = 0
    if closed:
        x3 = np.concatenate([fi - n, fi, fi + n])
        f3 = np.tile(raw[fi], 3)
        out = _interp_gaps(idx, x3, f3, order=params.interp_order)
    else:
        out = _interp_gaps(idx, fi, raw[fi], order=params.interp_order)
        k = max(1, int(params.anchor_n))
        lead, trail = idx < fi[0], idx > fi[-1]
        out[lead] = float(np.median(raw[fi[:k]]))
        out[trail] = float(np.median(raw[fi[-k:]]))
        n_held = int(lead.sum() + trail.sum())
    if params.smooth > 1:
        out = ndimage.median_filter(out, size=int(params.smooth), mode="wrap" if closed else "nearest")
    return out, ok, n_out, n_held


# --------------------------------------------------------------------------- refinement of one mask frame
@dataclass
class RefineParams:
    rect_width: int = 9        # rectangle width along the mask tangent (px)
    rect_length: int = 120     # rectangle length along the normal, straddling the curve (px)
    step: float = 2.0          # spacing of samples along the mask curve (px)
    edge_frac: float = 0.95    # sigmoid level that defines "plateau reached"
    max_shift: float = 35.0    # largest accepted shift from the expected edge (px); larger = rejected, not clamped
    smooth: int = 5            # median window (samples) smoothing the shifts along the curve; <=1 off
    min_amp: float = 0.0       # minimum sigmoid amplitude in integrated counts to accept a sample
    max_gap: float = 2.5       # max neighbour distance when chaining boundary pixels (px)
    bad_value: float = -1.0    # defective pixel value excluded from the integration (NaN always excluded)
    min_valid_frac: float = 0.5  # min fraction of valid width samples (and of images) for a profile point
    refine_border: bool = False  # DEPRECATED, ignored: the detector edge is handled by the edge anchor
    keep_profiles: int = 12      # number of background profiles + fits kept for the QC plot
    skip_bad_radius: float = 2.0  # skip samples whose centre is within this many px of a bad pixel
    interp_order: int = 2          # polynomial order for interpolating the shift between fitted samples (1 = linear)
    # --- new
    edge_margin: int = 2         # outermost detector rows/columns that never contribute (px)
    min_side_bins: int = 20      # valid profile bins required on EACH side of the edge (cuts off at the detector edge)
    anchor_n: int = 5            # edge anchor = median of the nearest n accepted fits at each open end
    outlier_sigma: float = 4.0   # reject fits further than this many robust sigma from their neighbours (0 = off)
    outlier_window: int = 8      # neighbours on each side used for that test (samples)
    outlier_floor: float = 0.75  # minimum robust sigma (px) for that test
    min_fit_frac: float = 0.05   # a curve needs this fraction of its samples fitted, otherwise it is left alone
    tangent_smooth: float = 8.0  # Gaussian smoothing length of the curve for the tangent/normal (px)
    use_prior: bool = True       # centre each image's rectangle on the edge expected at its exact omega
    max_prior: float = 60.0      # clamp of that expected displacement (px)
    spot_clip: object = "auto"   # Bragg-peak clipping: 'auto', counts per pixel, or None
    isolation_px: float = 35.0   # no second mask edge within this distance along the normal (0 = no check)
    corner_tol: float = 3.0      # edge positions on the rectangle's two long sides must agree within this (px)


@dataclass
class RefineResult:
    mask: np.ndarray                       # refined mask, True = shadowed
    points: np.ndarray                     # (n, 2) sample points (y, x) along the curve
    normals: np.ndarray                    # (n, 2) unit normals at the sample points
    shifts: np.ndarray                     # (n,) local edge shift along the normal (px), NaN if not fitted
    amps: np.ndarray                       # (n,) sigmoid amplitude b per sample, NaN if not fitted
    bright: np.ndarray = None              # (n,) +1/-1: bright side along +/- normal (from the mask)
    profile_pt: object = None              # (n_keep, 2) sample points of the kept background profiles
    profile_v: object = None               # (length,) normal offsets of the kept profiles
    profile_p: object = None               # (n_keep, length) stacked profiles (NaN where too few valid pixels)
    profile_fit: object = None             # (n_keep, 4) sigmoid params (a, b, v0, s) of the kept profiles
    profile_edge: object = None            # (n_keep,) high-side edge position ve of the kept profiles
    timings: dict = field(default_factory=dict)
    resid_rms: float = float("nan")        # mean normalized sigmoid fit residual over the fitted samples
    stats: dict = field(default_factory=dict)  # counts of fitted / rejected samples, anchored samples, ...

    @property
    def n_valid(self):
        return int(np.isfinite(self.shifts).sum())

    @property
    def expansions(self):
        """Shifts re-signed so that positive = the shadowed region grew (px)."""
        b = self.bright if self.bright is not None else np.sign(self.amps)
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(self.shifts), self.shifts * b, np.nan)

    def shifted_points(self):
        s = np.nan_to_num(self.shifts)
        return self.points + s[:, None] * self.normals


def refine_mask_frame(image, mask, params=None, stack=None, neighbors=None, dpe=1.0):
    """Refine one boolean mask frame (True = shadowed) against its diffraction image(s).

    image     : 2-D image (bad pixels == params.bad_value / NaN), or None when `stack` is given.
    stack     : list of (prepared float32 image with NaN, deviation_deg) -- every image of this entry and
                how many degrees its angle is away from the entry's angle.  Their profiles are stacked
                after centring each on its own expected edge position.
    neighbors : dict(prev=..., next=...) adjacent mask frames (bool) used to predict the edge position
                at the images' exact angles; dpe = degrees per entry.
    The mask boundary is walked as a curve (resampled every `step` px); at each sample the stacked
    profile along the normal is fitted with a sigmoid and the edge is moved to where the high plateau is
    reached.  Samples without a clean fit (detector edge, gaps, wrong polarity, outliers) get their shift
    by interpolation; at the ends of a curve the shift of the last clean samples is held ("edge anchor").
    """
    params = params or RefineParams()
    t_start = time.perf_counter()
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    if stack is None:
        stack = [(prepare_image(image, params.bad_value, params.edge_margin, params.spot_clip), 0.0)]
    for im, _ in stack:
        if im.shape != mask.shape:
            raise ValueError(f"image shape {im.shape} != mask frame shape {mask.shape}")
    n_img = len(stack)
    refined = mask.copy()
    W, L = int(params.rect_width), int(params.rect_length)
    uu, vv, v = _rect_grid(W, L)
    min_count = max(1, int(math.ceil(params.min_valid_frac * W)))
    need_img = max(1, int(math.ceil(params.min_valid_frac * n_img)))
    bad0 = ~np.isfinite(stack[0][0]) if params.skip_bad_radius > 0 else None
    nbs = {}
    if params.use_prior and neighbors is not None and any(abs(d) > 1e-9 for _, d in stack):
        for key in ("prev", "next"):
            nbs[key] = _neighbor_mask(neighbors.get(key))
    st = dict(n_chains=0, n_chains_refined=0, n_samples=0, n_fitted=0, n_cov_rej=0, n_fit_fail=0,
              n_sign_rej=0, n_range_rej=0, n_outlier=0, n_held=0, n_iso_rej=0)
    all_points, all_normals, all_shifts, all_amps, all_bright = [], [], [], [], []
    ref_q, ref_b = [], []
    kept, resids = [], []
    t_sample = t_fit = 0.0
    t_chain0 = time.perf_counter()
    chains = boundary_chains(mask, max_gap=params.max_gap)
    t_chains = time.perf_counter() - t_chain0
    for chain in chains:
        closed = _is_closed(chain, params.max_gap)
        pts = resample_chain(np.vstack([chain, chain[:1]]) if closed else chain, params.step)
        if closed:
            pts = pts[:-1]
        if len(pts) < 5:
            continue
        st["n_chains"] += 1
        m = len(pts)
        t, n = tangents_normals(pts, closed, smooth_sigma=params.tangent_smooth / max(params.step, 1e-6))
        G = bright_side_sign(mask, pts, n)
        skip = G == 0.0
        if params.isolation_px > 0:
            iso_ok = edge_isolated(mask, pts, t, n, W, L, params.isolation_px, params.corner_tol)
            st["n_iso_rej"] += int((~iso_ok & ~skip).sum())
            skip |= ~iso_ok
        if bad0 is not None:
            r = int(math.ceil(params.skip_bad_radius))
            y0 = np.clip(np.rint(pts[:, 0]).astype(int), 0, ny - 1)
            x0 = np.clip(np.rint(pts[:, 1]).astype(int), 0, nx - 1)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    skip |= bad0[np.clip(y0 + dy, 0, ny - 1), np.clip(x0 + dx, 0, nx - 1)]
        rmax = params.max_prior + 5.0
        a_prev = _expansion(nbs.get("prev"), pts, n, G, rmax)
        a_next = _expansion(nbs.get("next"), pts, n, G, rmax)
        priors = [_prior_shift(a_prev, a_next, d, dpe, G, params.max_prior) for _, d in stack]
        raw = np.zeros(m)
        amps = np.full(m, np.nan)
        for i in range(m):
            if skip[i]:
                continue
            ts = time.perf_counter()
            acc = np.zeros(L)
            nval = np.zeros(L, dtype=int)
            for j, (im, _) in enumerate(stack):
                cy = pts[i, 0] + priors[j][i] * n[i, 0]
                cx = pts[i, 1] + priors[j][i] * n[i, 1]
                ys = cy + uu * t[i, 0] + vv * n[i, 0]
                xs = cx + uu * t[i, 1] + vv * n[i, 1]
                smp = ndimage.map_coordinates(im, [ys, xs], order=1, mode="constant", cval=np.nan)
                valid = np.isfinite(smp)
                cnt = valid.sum(axis=0)
                okb = cnt >= min_count
                ssum = np.where(valid, smp, 0.0).sum(axis=0, dtype=np.float64)
                acc += np.where(okb, ssum * W / np.maximum(cnt, 1), 0.0)
                nval += okb
            prof = np.where(nval >= need_img, acc / np.maximum(nval, 1) * n_img, np.nan)
            t_sample += time.perf_counter() - ts
            fin = np.isfinite(prof)
            if int((fin & (v < 0)).sum()) < params.min_side_bins or int((fin & (v > 0)).sum()) < params.min_side_bins:
                st["n_cov_rej"] += 1
                continue
            tf = time.perf_counter()
            fit = fit_edge_profile(prof, v, min_amp=params.min_amp)
            t_fit += time.perf_counter() - tf
            if fit is None:
                st["n_fit_fail"] += 1
                continue
            if fit["b"] * G[i] <= 0.0:          # the shadow must be the dark side
                st["n_sign_rej"] += 1
                continue
            ve = fit["v0"] + math.copysign(_logit(params.edge_frac) * fit["s"], fit["b"])
            if abs(ve) >= params.max_shift:
                st["n_range_rej"] += 1
                continue
            raw[i] = ve
            amps[i] = fit["b"]
            fm = np.isfinite(prof)
            resid = prof[fm] - sigmoid(v[fm], fit["a"], fit["b"], fit["v0"], fit["s"])
            resids.append(float(np.sqrt(np.mean(resid ** 2)) / max(abs(fit["b"]), 1e-9)))
            kept.append((pts[i].copy(), prof, v, fit, ve))
        fitted = np.isfinite(amps)
        st["n_samples"] += m
        st["n_fitted"] += int(fitted.sum())
        all_points.append(pts)
        all_normals.append(n)
        all_bright.append(G)
        if int(fitted.sum()) < max(5, int(params.min_fit_frac * m)):
            all_shifts.append(np.full(m, np.nan))
            all_amps.append(amps)
            continue
        shifts, accepted, n_out, n_held = _finalize_shifts(raw, fitted, closed, params)
        st["n_outlier"] += n_out
        st["n_held"] += n_held
        st["n_chains_refined"] += 1
        shifts = np.where(G == 0.0, 0.0, shifts)
        for i in range(m):
            if G[i] == 0.0:
                continue
            _paint(refined, pts[i, 0], pts[i, 1], n[i, 0], n[i, 1], shifts[i], bool(G[i] * shifts[i] > 0.0))
            j = (i + 1) % m if closed else i + 1
            if j < m and G[j] == G[i]:
                seg = max(int(np.hypot(*(pts[j] - pts[i])) / 0.5), 1)
                for a in np.linspace(0.0, 1.0, seg + 1)[1:]:
                    p = pts[i] + a * (pts[j] - pts[i])
                    nn = n[i] + a * (n[j] - n[i])
                    nn = nn / max(float(np.hypot(*nn)), 1e-9)
                    sh = shifts[i] + a * (shifts[j] - shifts[i])
                    _paint(refined, p[0], p[1], nn[0], nn[1], sh, bool(G[i] * sh > 0.0))
        if not closed:
            for end, tdir in ((0, -1.0), (m - 1, +1.0)):
                p0 = pts[end]
                if G[end] == 0.0 or not (p0[0] <= 2.5 or p0[0] >= ny - 3.5 or p0[1] <= 2.5 or p0[1] >= nx - 3.5):
                    continue                      # only ends that run into the detector edge
                ext = 1.5 * abs(shifts[end]) + 6.0
                for sdist in np.arange(0.5, ext, 0.5):
                    p = p0 + tdir * sdist * t[end]
                    _paint(refined, p[0], p[1], n[end, 0], n[end, 1], shifts[end], bool(G[end] * shifts[end] > 0.0))
        sel = G != 0.0
        ref_q.append(pts[sel] + shifts[sel, None] * n[sel])
        ref_b.append(G[sel, None] * n[sel])
        all_shifts.append(np.where(accepted, raw, np.nan))
        all_amps.append(np.where(accepted, amps, np.nan))
    # remove theoretical boundary PIXELS that stick out beyond the refined edge
    # (after a shrink, or detached spikes, the old edge must not stay masked)
    if ref_q:
        q = np.concatenate(ref_q)
        bright = np.concatenate(ref_b)
        tree = cKDTree(q)
        bp = boundary_pixels(mask).astype(float)
        if len(bp):
            dist, jj = tree.query(bp)
            d = np.einsum("ij,ij->i", bp - q[jj], bright[jj])
            for p in bp[(dist <= params.max_shift + 3.0) & (d > 0.5)]:
                refined[int(round(p[0])), int(round(p[1]))] = False
    if all_points:
        points = np.concatenate(all_points)
        normals = np.concatenate(all_normals)
        shifts_all = np.concatenate(all_shifts)
        amps_all = np.concatenate(all_amps)
        bright_all = np.concatenate(all_bright)
    else:
        points = np.zeros((0, 2)); normals = np.zeros((0, 2))
        shifts_all = np.zeros(0); amps_all = np.zeros(0); bright_all = np.zeros(0)
    if len(kept) > params.keep_profiles > 0:
        kept = [kept[j] for j in np.linspace(0, len(kept) - 1, params.keep_profiles).astype(int)]
    if kept:
        profile_pt = np.array([k[0] for k in kept])
        profile_v = kept[0][2]
        profile_p = np.array([k[1] for k in kept])
        profile_fit = np.array([[k[3]["a"], k[3]["b"], k[3]["v0"], k[3]["s"]] for k in kept])
        profile_edge = np.array([k[4] for k in kept])
    else:
        profile_pt = profile_v = profile_p = profile_fit = profile_edge = None
    t_total = time.perf_counter() - t_start
    timings = dict(t_chains=t_chains, t_sample=t_sample, t_fit=t_fit, t_total=t_total,
                   t_other=t_total - t_chains - t_sample - t_fit)
    return RefineResult(mask=refined, points=points, normals=normals, shifts=shifts_all, amps=amps_all,
                        bright=bright_all, profile_pt=profile_pt, profile_v=profile_v, profile_p=profile_p,
                        profile_fit=profile_fit, profile_edge=profile_edge, timings=timings,
                        resid_rms=float(np.mean(resids)) if resids else float("nan"), stats=st)


def grid_search(image_paths, mask, params, frame_counts=(1, 3, 5), rect_widths=(13, 27, 35)):
    """Evaluate refinement reliability over a small parameter grid.

    For each (n_frames, rect_width) combo the first n_frames images are summed
    and one mask frame is refined; per combo a row with the number of fitted
    samples, the median edge shift and its MAD, the mean normalized fit
    residual and the wall time is returned. Use it to pick the summation depth
    and rectangle width before running the whole dataset.
    """
    rows = []
    for k in frame_counts:
        k = max(1, min(int(k), len(image_paths)))
        img = sum_images(image_paths[:k], bad_value=params.bad_value)
        for w in rect_widths:
            p = replace(params, rect_width=int(w))
            res = refine_mask_frame(img, np.asarray(mask, dtype=bool), p)
            exp = res.expansions[np.isfinite(res.expansions)]
            med = float(np.median(exp)) if exp.size else None
            rows.append(dict(
                frames=k, rect_width=int(w),
                n_samples=int(len(res.shifts)), n_valid=res.n_valid,
                valid_frac=res.n_valid / max(1, len(res.shifts)),
                median_shift=med,
                shift_mad=float(np.median(np.abs(exp - med))) if exp.size else None,
                fit_rms=float(res.resid_rms),
                seconds=float(res.timings.get("t_total", 0.0)),
            ))
    return rows


def grid_figure(rows, out_png):
    """Bar chart of the grid search: median edge shift (+/- MAD) and fit quality per combo (png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['frames']}f x {r['rect_width']}px" for r in rows]
    x = np.arange(len(rows))
    med = [r["median_shift"] if r["median_shift"] is not None else np.nan for r in rows]
    mad = [r["shift_mad"] if r["shift_mad"] is not None else np.nan for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.subplots_adjust(hspace=0.15)
    ax1.bar(x, med, yerr=mad, color="steelblue", capsize=3)
    ax1.set_title("refine grid search", fontsize=11)
    ax1.set_ylabel("median edge shift +/- MAD (px)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.axhline(0.0, color="0.5", lw=0.8)
    ax2.bar(x, [r["valid_frac"] for r in rows], color="seagreen", label="valid fraction")
    ax2.plot(x, [r["fit_rms"] for r in rows], "o-", color="darkred", label="fit rms (norm.)")
    ax2.set_ylabel("valid fraction / mean normalized fit residual")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


# --------------------------------------------------------------------------- QC figure of a refinement
def _sum_images_nan(files, params):
    """NaN-aware sum of several images (bad pixels / Bragg peaks excluded, normalised to len(files) images)."""
    acc = cnt = None
    for fp in files:
        im = prepare_image(load_image(fp), params.bad_value, 0, params.spot_clip)
        if acc is None:
            acc = np.zeros(im.shape, np.float32)
            cnt = np.zeros(im.shape, np.float32)
        fin = np.isfinite(im)
        acc += np.where(fin, im, 0.0)
        cnt += fin
    return np.where(cnt > 0, acc * (len(files) / np.maximum(cnt, 1)), np.nan)


def refine_qc_figure(items, params, out_png, max_display_px=1400, zoom_px=300, n_traces=12, title=""):
    """Overview + zoom + fitted background profiles of the refinement QC.

    items: list of dict(label, entry, mask_angle, files, orig, refined, median_expansion, n_fitted, n_samples,
    result)  (orig / refined are boolean frames in image orientation; result = RefineResult of the entry, whose
    kept background profiles are drawn in the bottom row).  Top row: whole detector, binned; middle row:
    a zoom_px square at full resolution around the middle of the visible refined edge, marked in the top row;
    bottom row: the integrated background profiles kept during the refinement (kept spread over the curve,
    '--keep-profiles' of them) with their sigmoid fits and the refined edge position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    n = len(items)
    H, W = items[0]["orig"].shape
    fig, axes = plt.subplots(3, n, figsize=(6.4 * n, 6.4 * H / W + 8.6), squeeze=False,
                             gridspec_kw=dict(height_ratios=[H / W, 1.0, 0.9]))
    c_orig, c_ref = "#00d5ff", "#ff2d2d"
    a_ovw, a_zoom = 0.55, 0.7                  # opacity of the mask outlines over the images
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for col, it in enumerate(items):
        img = _sum_images_nan(it["files"], params)
        sb = max(1, int(math.ceil(max(H, W) / float(max_display_px))))
        disp = _bin_nanmean(img, sb)
        fin = disp[np.isfinite(disp)]
        vmax = max(float(np.percentile(fin, 99.5)) if fin.size else 1.0, 1e-6)
        ax = axes[0, col]
        ax.imshow(disp, cmap="gray_r", vmin=0.0, vmax=25, origin="upper",
                  extent=(-0.5, disp.shape[1] * sb - 0.5, disp.shape[0] * sb - 0.5, -0.5), interpolation="nearest")
        for fr, colr, lw in ((it["orig"], c_orig, 1.2), (it["refined"], c_ref, 1.4)):
            bm = _bin_mask(fr, sb)
            if bm.min() < 0.5 <= bm.max():
                xs = (np.arange(bm.shape[1]) + 0.5) * sb - 0.5
                ys = (np.arange(bm.shape[0]) + 0.5) * sb - 0.5
                ax.contour(xs, ys, bm, levels=[0.5], colors=[colr], linewidths=lw, alpha=a_ovw)
        # zoom window: middle of the visible refined edge
        bp = boundary_pixels(it["refined"], True)
        if len(bp):
            med = np.median(bp, axis=0)
            cy, cx = bp[int(np.argmin(((bp - med) ** 2).sum(axis=1)))]
        else:
            cy, cx = H // 2, W // 2
        z = int(min(zoom_px, H, W))
        y0 = int(np.clip(cy - z // 2, 0, H - z))
        x0 = int(np.clip(cx - z // 2, 0, W - z))
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), z, z, fill=False, ec="yellow", lw=1.2, ls="--"))
        ang = f"mask {float(it['mask_angle']):.1f} deg" if np.isfinite(float(it["mask_angle"])) else "single frame"
        ax.set_title(f"{it['label']} - entry {it['entry']} ({ang})\n"
                     f"{len(it['files'])} image{'s' if len(it['files']) != 1 else ''} summed; edge moved by "
                     f"{it['median_expansion']:+.1f} px (median), {it['n_fitted']}/{it['n_samples']} samples fitted", fontsize=9)
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        # zoom row at full resolution (2x2 binned for the noise)
        az = axes[1, col]
        crop = img[y0:y0 + z, x0:x0 + z]
        zb = 2 if z >= 200 else 1
        zd = _bin_nanmean(crop, zb)
        zf = zd[np.isfinite(zd)]
        zmax = max(float(np.percentile(zf, 99.5)) if zf.size else 1.0, 1e-6)
        az.imshow(zd, cmap="gray", vmin=0.0, vmax=zmax, origin="upper",
                  extent=(x0 - 0.5, x0 + zd.shape[1] * zb - 0.5, y0 + zd.shape[0] * zb - 0.5, y0 - 0.5),
                  interpolation="nearest")
        xs = np.arange(x0, x0 + z)
        ys = np.arange(y0, y0 + z)
        for fr, colr, lw in ((it["orig"], c_orig, 1.6), (it["refined"], c_ref, 2.0)):
            cm = fr[y0:y0 + z, x0:x0 + z].astype(float)
            if cm.min() < 0.5 <= cm.max():
                az.contour(xs, ys, cm, levels=[0.5], colors=[colr], linewidths=lw, alpha=a_zoom)
        az.set_xlim(x0 - 0.5, x0 + z - 0.5)
        az.set_ylim(y0 + z - 0.5, y0 - 0.5)
        az.set_title(f"zoom {z} x {z} px at x {x0}-{x0 + z}, y {y0}-{y0 + z}", fontsize=9)
        az.set_xlabel("X (pixel)")
        az.set_ylabel("Y (pixel)")
        # third row: kept background profiles + sigmoid fits + refined edge, as in refine-debug
        ap = axes[2, col]
        res = it.get("result")
        shown = 0
        if res is not None and getattr(res, "profile_p", None) is not None and len(res.profile_p):
            v = np.asarray(res.profile_v, dtype=float)
            nv = max(1, min(int(n_traces), len(res.profile_p)))
            sel = np.linspace(0, len(res.profile_p) - 1, nv).astype(int)
            vv = np.linspace(v[0], v[-1], 200)
            shown = len(sel)
            for j, k in enumerate(sel):
                clr = cyc[int(j) % len(cyc)]
                p = np.asarray(res.profile_p[k], dtype=float)
                fm = np.isfinite(p)
                ap.plot(v[fm], p[fm], "-", color=clr, lw=0.9, alpha=0.75)
                a, b, v0, s = res.profile_fit[k]
                ap.plot(vv, sigmoid(vv, a, b, v0, s), "--", color=clr, lw=1.2, alpha=0.85)
                ve = float(res.profile_edge[k])
                ap.plot([ve], [sigmoid(ve, a, b, v0, s)], "x", color="red", ms=8, mew=1.5)
            ap.axvline(0.0, color="0.6", lw=0.8, ls=":")
        if not shown:
            ap.text(0.5, 0.5, "no profiles kept\n(run with --keep-profiles > 0)", ha="center", va="center",
                    fontsize=10, transform=ap.transAxes)
        ap.set_title(f"background profiles + sigmoid fits (x = refined edge{f'; {shown} kept' if shown else ''})",
                     fontsize=9)
        ap.set_xlabel("offset along mask normal (px)")
        ap.set_ylabel("intensity (counts)")
    handles = [Line2D([0], [0], color=c_orig, lw=1.6, label="original (theoretical) mask"),
               Line2D([0], [0], color=c_ref, lw=2.0, label="refined mask"),
               Line2D([0], [0], color="0.35", lw=1.0, label="background profile"),
               Line2D([0], [0], color="0.35", lw=1.2, ls="--", label="sigmoid fit"),
               Line2D([0], [0], color="red", lw=0, marker="x", ms=8, label="refined edge")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, framealpha=0.9)
    fig.suptitle(title or "refinement QC: summed diffraction images with original and refined mask", fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


# --------------------------------------------------------------------------- dataset driver (streamed npz -> npz)
def _load_debug_mask(path, key="mask", entry=None):
    """One mask frame for refine-debug: entry `entry` of a per-degree (entries, rows, cols) npz,
    or the single (rows, cols) frame itself if the array is 2-D.

    Returns (frame, meta, entry, mask_angle, stream) with meta = all other arrays of the npz,
    entry/mask_angle = entry index and its mask-axis angle (None / nan for a single frame) and
    stream = the MaskStream of a per-degree input (None for a single frame)."""
    path = Path(path)
    try:
        stream = MaskStream(path, key)
    except ValueError:
        with np.load(path) as z:
            names = z.files
            use = key if key in names else names[0]
            arr = np.asarray(z[use])
            if arr.ndim != 2:
                raise SystemExit(f"{path}: expected a (rows, cols) or (entries, rows, cols) mask, got {use} {arr.shape}")
            meta = {nm: z[nm] for nm in names if nm != use}
            return np.asarray(arr, dtype=bool), meta, None, float("nan"), None
    nz, rows, cols = stream.shape
    meta = stream.small_arrays()
    dpe = 360.0 / nz
    if "omega_deg" in meta and np.asarray(meta["omega_deg"]).size > 1 and abs(float(np.diff(meta["omega_deg"])[0])) > 0:
        dpe = abs(float(np.diff(meta["omega_deg"])[0]))
    if entry is None:
        act = mask_shadow_activity(path, key, cache=True)
        entry = procession_entries(shadow_procession(act, dpe), (50,))[0]
    entry = int(entry) % nz
    frame = stream.get_frames([entry])[entry]
    angle = float(np.asarray(meta["omega_deg"])[entry]) if "omega_deg" in meta else entry * dpe
    return frame, meta, entry, angle, stream


def _refine_entry_task(task):
    """Worker: load the images of one mask entry, refine the entry, return (entry, RefineResult)."""
    params = task["params"]
    stack = []
    for path, d in task["items"]:
        stack.append((prepare_image(load_image(path), params.bad_value, params.edge_margin, params.spot_clip), float(d)))
    if stack[0][0].shape != task["m_cur"].shape:
        raise ValueError(f"image shape {stack[0][0].shape} != oriented mask frame shape {task['m_cur'].shape}: "
                         "check --transpose / the mask orientation")
    res = refine_mask_frame(None, task["m_cur"], params, stack=stack,
                            neighbors=dict(prev=task["m_prev"], next=task["m_next"]), dpe=task["dpe"])
    return task["k"], res


def _orient_beam_position(pos, orient, rows, cols):
    """Transform a (row, col) continuous pixel position through the orientation (transpose, flip v, flip h)."""
    r, c = float(pos[0]), float(pos[1])
    if orient.transpose:
        r, c = c, r
        rows, cols = cols, rows
    if orient.flip_vertical:
        r = rows - r
    if orient.flip_horizontal:
        c = cols - c
    return np.array([r, c])


def refine_dataset(mask_source, out_path, image_paths, frame_numbers, *, starting_angle, oscillation_range,
                   starting_frame=1.0, angle_offset=0.0, flip_time=False, half_frame=True,
                   orient=Orientation(), output_orientation="input", params=None, key="mask",
                   deg_per_entry=None, max_dev_deg=None, min_images=1, entries=None, jobs=1,
                   report_path=None, qc_image=None, qc_percentiles=(25, 50, 75), qc_max_px=1400,
                   qc_zoom_px=300, verbose=True):
    """Refine a per-degree mask npz against raw images and write a refined npz.

    Entries are processed in file order from the streamed input; entries without images (or without
    shadow) are copied unchanged.  qc_image: path of a png with the summed images at the given percentiles
    of the refined entries, original and refined outlines overlaid.  Returns the report dict.  See the
    module docstring for the flags.
    """
    params = params or RefineParams()
    stream = MaskStream(mask_source, key)
    nz, rows, cols = stream.shape
    meta = stream.small_arrays()
    if deg_per_entry:
        dpe = float(deg_per_entry)
    elif "omega_deg" in meta and np.asarray(meta["omega_deg"]).size > 1 and abs(float(np.diff(meta["omega_deg"])[0])) > 0:
        dpe = abs(float(np.diff(meta["omega_deg"])[0]))
    else:
        dpe = 360.0 / nz
    oshape = orient.shape(rows, cols)
    out_shape = oshape if output_orientation == "image" else (rows, cols)

    # --- images -> mask entries (with their deviation from the entry angle)
    omega_xds = np.array([image_omega(f, starting_angle + angle_offset, oscillation_range, starting_frame, half_frame)
                          for f in frame_numbers])
    omega_mask = np.array([mask_angle(o, flip_time) for o in omega_xds])
    ent, dev = assign_entries(omega_mask, nz, dpe)
    max_dev = dpe / 2.0 + 1e-9 if max_dev_deg is None else float(max_dev_deg)
    items_by_entry = {}
    for p, k, d in zip(image_paths, ent, dev):
        if abs(d) <= max_dev:
            items_by_entry.setdefault(int(k), []).append((str(p), float(d)))
    for k in items_by_entry:
        items_by_entry[k].sort(key=lambda it: abs(it[1]))
    if verbose:
        print(f"mask: {nz} entries of {dpe:g} deg, {rows} x {cols} px; {len(image_paths)} images -> "
              f"{len(items_by_entry)} entries; omega_xds {omega_xds.min():.2f} .. {omega_xds.max():.2f}, "
              f"mask-axis {omega_mask.min():.2f} .. {omega_mask.max():.2f}")

    qc_labels, qc_data = {}, {}
    if qc_image:
        try:
            act = mask_shadow_activity(mask_source, key, cache=True)
            will = [k for k in range(nz) if act[k] and len(items_by_entry.get(k, [])) >= min_images and
                    (entries is None or entries[0] <= k < entries[1])]
            if will:
                proc = shadow_procession(act, dpe)
                order = sorted(will, key=lambda k: (k - proc["start"]) % nz)
                for pc in qc_percentiles:
                    qc_labels.setdefault(order[int(round(pc / 100.0 * (len(order) - 1)))], float(pc))
        except Exception as exc:                      # the QC image is a convenience, never a reason to fail
            print(f"note: QC image skipped ({exc})")
    arrays = {name: np.zeros(nz, dtype=dt) for name, dt in
              (("refined", bool), ("n_images", np.int32), ("n_samples", np.int32), ("n_fitted", np.int32),
               ("n_held", np.int32), ("median_expansion_px", np.float64), ("mad_expansion_px", np.float64))}
    arrays["median_expansion_px"][:] = np.nan
    arrays["mad_expansion_px"][:] = np.nan
    entry_stats = []
    writer = NpzStreamWriter(out_path, nz, *out_shape)
    pool = ProcessPoolExecutor(max_workers=int(jobs)) if jobs and jobs > 1 else None
    max_pending = (2 * int(jobs) + 2) if pool else 2
    pending = deque()
    t0 = time.perf_counter()
    done_entries = 0

    def write_one(k, res, raw):
        nonlocal done_entries
        if res is None:
            frame = orient.apply(raw) if output_orientation == "image" else raw
        else:
            frame = res.mask if output_orientation == "image" else orient.invert(res.mask)
            exp = res.expansions[np.isfinite(res.expansions)]
            med = float(np.median(exp)) if exp.size else float("nan")
            arrays["refined"][k] = True
            arrays["n_images"][k] = len(items_by_entry[k])
            arrays["n_samples"][k] = res.stats.get("n_samples", 0)
            arrays["n_fitted"][k] = res.stats.get("n_fitted", 0)
            arrays["n_held"][k] = res.stats.get("n_held", 0)
            arrays["median_expansion_px"][k] = med
            arrays["mad_expansion_px"][k] = float(np.median(np.abs(exp - med))) if exp.size else float("nan")
            entry_stats.append(dict(entry=k, n_images=len(items_by_entry[k]), omega_mask_deg=k * dpe,
                                    frames=[Path(p).name for p, _ in items_by_entry[k]],
                                    median_expansion_px=med if exp.size else None,
                                    mad_expansion_px=float(arrays["mad_expansion_px"][k]) if exp.size else None,
                                    seconds=res.timings.get("t_total"), **{kk: int(vv) for kk, vv in res.stats.items()}))
            if k in qc_labels:
                qc_data[k] = dict(orig=orient.apply(raw), refined=res.mask.copy(), median_expansion=med,
                                  n_fitted=int(res.stats.get("n_fitted", 0)), n_samples=int(res.stats.get("n_samples", 0)),
                                  result=res)
            done_entries += 1
            if verbose:
                print(f"  entry {k:3d}: {len(items_by_entry[k]):2d} images, {res.stats['n_fitted']:5d}/"
                      f"{res.stats['n_samples']:5d} fitted, {res.stats['n_held']:4d} held at edge, "
                      f"median expansion {med:6.2f} px, {res.timings.get('t_total', 0):.1f}s", flush=True)
        writer.write(frame)

    def drain(force=False):
        while pending:
            kind, k, payload, raw = pending[0]
            if kind == "fut" and not (force or len(pending) > max_pending or payload.done()):
                break
            res = payload.result()[1] if kind == "fut" else payload
            pending.popleft()
            write_one(k, res, raw)

    try:
        it = stream.iter_frames()
        _, first_raw = next(it)
        cur_raw, prev_raw = first_raw, None
        for k in range(nz):
            nxt_raw = next(it)[1] if k + 1 < nz else first_raw
            items = items_by_entry.get(k, [])
            wanted = (entries is None or entries[0] <= k < entries[1])
            if items and len(items) >= min_images and wanted and cur_raw.any() and not cur_raw.all():
                task = dict(k=k, items=items, params=params, dpe=dpe, m_cur=orient.apply(cur_raw),
                            m_prev=orient.apply(prev_raw) if prev_raw is not None else None,
                            m_next=orient.apply(nxt_raw))
                if pool:
                    pending.append(("fut", k, pool.submit(_refine_entry_task, task), cur_raw))
                else:
                    pending.append(("res", k, _refine_entry_task(task)[1], cur_raw))
            else:
                pending.append(("raw", k, None, cur_raw))
            drain()
            prev_raw, cur_raw = cur_raw, nxt_raw
        drain(force=True)
    finally:
        if pool:
            pool.shutdown(wait=True)

    extra = {k_: v_ for k_, v_ in meta.items()}
    if output_orientation == "image" and "beam_position_row_col_px" in extra:
        extra["beam_position_row_col_px"] = _orient_beam_position(extra["beam_position_row_col_px"], orient, rows, cols)
    extra.update(arrays)
    extra["refine_params_json"] = np.array(json.dumps({k_: (v_ if isinstance(v_, (int, float, str, bool, type(None))) else str(v_))
                                                       for k_, v_ in vars(params).items()}))
    extra["orientation"] = np.array(dict(transpose=orient.transpose, flip_vertical=orient.flip_vertical,
                                         flip_horizontal=orient.flip_horizontal, flip_time=bool(flip_time),
                                         angle_offset_deg=float(angle_offset),
                                         output_orientation=output_orientation).__repr__())
    writer.close(extra)
    report = dict(n_images=len(image_paths), n_entries=nz, deg_per_entry=dpe, pixel_shape=[rows, cols],
                  n_entries_refined=done_entries, starting_angle=float(starting_angle),
                  oscillation_range=float(oscillation_range), starting_frame=float(starting_frame),
                  angle_offset_deg=float(angle_offset), flip_time=bool(flip_time), half_frame=bool(half_frame),
                  orientation=dict(transpose=orient.transpose, flip_vertical=orient.flip_vertical,
                                   flip_horizontal=orient.flip_horizontal),
                  output_orientation=output_orientation, seconds=time.perf_counter() - t0,
                  entries=sorted(entry_stats, key=lambda e: e["entry"]), output=str(out_path))
    if report_path:
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
    if qc_image and qc_data:
        items = []
        for k in sorted(qc_data, key=lambda k: qc_labels[k]):
            items.append(dict(qc_data[k], label=f"{qc_labels[k]:g}th percentile of shadow travel",
                              entry=k, mask_angle=k * dpe, files=[p_ for p_, _ in items_by_entry[k]]))
        refine_qc_figure(items, params, qc_image, qc_max_px, qc_zoom_px,
                         title=f"refinement QC - {Path(out_path).name}")
        report["qc_image"] = str(qc_image)
        if report_path:
            with open(report_path, "w") as fh:
                json.dump(report, fh, indent=2)
    if verbose:
        print(f"refined {done_entries} of {nz} mask entries -> {out_path}  ({report['seconds']:.1f} s)"
              + (f"; QC image -> {qc_image}" if report.get("qc_image") else ""))
    return report


# --------------------------------------------------------------------------- angle-offset tuning helper
def mask_shadow_activity(mask_source, key="mask", cache=True):
    """bool array (n_entries,): which mask entries contain any shadow.

    One streaming pass over the mask (10-40 s for a 6.5 GB npz); the result is cached in
    '<mask>.active.json' next to the file (re-used while size and modification time are unchanged)."""
    cache_path = None
    if cache and not isinstance(mask_source, np.ndarray):
        mp = Path(mask_source)
        cache_path = mp.with_suffix(".active.json")
        try:
            st = mp.stat()
            if cache_path.is_file():
                c = json.loads(cache_path.read_text())
                if c.get("size") == st.st_size and c.get("mtime") == st.st_mtime and c.get("key") == key:
                    act = np.zeros(int(c["n"]), dtype=bool)
                    act[c["active"]] = True
                    return act
        except OSError:
            cache_path = None
    stream = MaskStream(mask_source, key)
    act = np.zeros(stream.shape[0], dtype=bool)
    for k, fr in stream.iter_frames():
        act[k] = bool(fr.any())
    if cache_path is not None:
        try:
            st = Path(mask_source).stat()
            cache_path.write_text(json.dumps(dict(size=st.st_size, mtime=st.st_mtime, key=key, n=int(act.size),
                                                  active=np.nonzero(act)[0].tolist())))
        except OSError:
            pass
    return act


def shadow_procession(active, deg_per_entry):
    """The contiguous (circular) run of mask entries that carry shadow -- the 'procession' of the shadow.

    Returns dict(start, length, mid, arcs): start = first entry, length = number of entries,
    duration_deg, start_deg / end_deg / mid_deg (mask-axis angles) and the list of all runs
    (the longest one is used when the shadow appears in several separate runs)."""
    active = np.asarray(active, dtype=bool)
    n = active.size
    if not active.any():
        raise ValueError("the mask contains no shadow at all")
    if active.all():
        runs = [(0, n)]
    else:
        first_off = int(np.argmin(active))              # first False
        rolled = np.roll(active, -first_off)             # starts on an empty entry, so runs never wrap
        runs, i = [], 0
        while i < n:
            if rolled[i]:
                j = i
                while j < n and rolled[j]:
                    j += 1
                runs.append(((i + first_off) % n, j - i))
                i = j
            else:
                i += 1
    start, length = max(runs, key=lambda r: r[1])
    dpe = float(deg_per_entry)
    return dict(start=int(start), length=int(length), n=int(n), duration_deg=length * dpe,
                start_deg=start * dpe, end_deg=((start + length - 1) % n) * dpe,
                mid_deg=((start + (length - 1) / 2.0) % n) * dpe, runs=[(int(a), int(b)) for a, b in runs])


def procession_entries(proc, percentiles=(25, 50, 75)):
    """Mask entries at the given percentiles (0-100) of the shadow's travel along its procession."""
    return [int((proc["start"] + int(round(pc / 100.0 * (proc["length"] - 1)))) % proc["n"]) for pc in percentiles]


def frame_for_mask_angle(omega_mask, starting_angle, oscillation_range, starting_frame, flip_time,
                         frames, half_frame=True):
    """Continuous frame number whose exposure centre is at the mask-axis angle `omega_mask`
    (starting_angle already includes the angle offset).  The rotation repeats every 360 deg, so
    the branch that falls inside the available frame range is returned; None if the images do not cover it."""
    s = -1.0 if flip_time else 1.0
    target = s * float(omega_mask)                       # XDS angle (mod 360)
    off = 0.5 if half_frame else 1.0
    fmin, fmax = float(min(frames)), float(max(frames))
    span = abs(oscillation_range) * (fmax - fmin + 2.0)
    cands = []
    for m in range(-int(span // 360) - 3, int(span // 360) + 4):
        f = float(starting_frame) - off + (target - starting_angle + 360.0 * m) / oscillation_range
        if fmin - 0.5 <= f <= fmax + 0.5:
            cands.append(f)
    return min(cands) if cands else None


def _bin_nanmean(a, s):
    """Block mean (s x s) ignoring NaN; NaN where a block has no valid pixel."""
    H, W = a.shape
    h, w = H // s * s, W // s * s
    v = a[:h, :w].reshape(h // s, s, w // s, s)
    fin = np.isfinite(v)
    tot = np.where(fin, v, 0.0).sum(axis=(1, 3))
    cnt = fin.sum(axis=(1, 3))
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _bin_mask(m, s):
    H, W = m.shape
    h, w = H // s * s, W // s * s
    return m[:h, :w].reshape(h // s, s, w // s, s).mean(axis=(1, 3))


def tune_angle_offset(mask_source, image_paths, frame_numbers, *, starting_angle, oscillation_range,
                      starting_frame=1.0, angle_offset=0.0, flip_time=False, half_frame=True,
                      orient=Orientation(), key="mask", deg_per_entry=None, percentiles=(25, 50, 75),
                      step_deg=5.0, sum_window_deg=1.0, sum_images=None, max_display_px=1400,
                      out_png="tune_angle_offset.png", show=False, cache=True, params=None, verbose=True):
    """Figure to judge the angle offset by eye.

    The shadow of the mask moves ("processes") over a range of mask angles.  Three mask angles are taken at
    the 25th / 50th / 75th percentile of that travel.  For the given angle_offset the matching diffraction
    images are located, the images around each one are summed (sum_window_deg or sum_images) and
    shown with three mask outlines on top: the mask at that angle (thick, the current offset) and the masks
    step_deg before and after it.  If the shadow edge in an image follows the +step (or -step) outline instead of
    the middle one, the legend of that outline says which angle offset to try (offset + s*step, s = -1 with --flip-time).

    Returns a dict: picks (entries, mask angles, XDS angles, frames, files) and the figure path."""
    params = params or RefineParams()
    stream = MaskStream(mask_source, key)
    nz, rows, cols = stream.shape
    meta = stream.small_arrays()
    if deg_per_entry:
        dpe = float(deg_per_entry)
    elif "omega_deg" in meta and np.asarray(meta["omega_deg"]).size > 1 and abs(float(np.diff(meta["omega_deg"])[0])) > 0:
        dpe = abs(float(np.diff(meta["omega_deg"])[0]))
    else:
        dpe = 360.0 / nz
    active = mask_shadow_activity(mask_source, key, cache)
    proc = shadow_procession(active, dpe)
    entries = procession_entries(proc, percentiles)
    step_e = max(1, int(round(step_deg / dpe)))
    s_sign = -1.0 if flip_time else 1.0
    if verbose:
        print(f"shadow procession: mask angles {proc['start_deg']:.1f} -> {proc['end_deg']:.1f} deg "
              f"({proc['length']} entries = {proc['duration_deg']:.1f} deg, middle {proc['mid_deg']:.1f} deg)"
              + (f"; {len(proc['runs']) - 1} other shadow run(s) ignored" if len(proc["runs"]) > 1 else ""))
    frame_map = {int(f): Path(p) for f, p in zip(frame_numbers, image_paths)}
    avail = sorted(frame_map)
    n_sum = int(sum_images) if sum_images else max(1, int(round(sum_window_deg / abs(oscillation_range))))
    n_sum += 1 - (n_sum % 2)                              # odd, so the window is centred on one image
    needed = {(e + j * step_e) % nz for e in entries for j in (-1, 0, 1)}
    mframes = {k: orient.apply(f) for k, f in stream.get_frames(needed).items()}
    start_eff = starting_angle + angle_offset
    picks = []
    for pc, e in zip(percentiles, entries):
        w_mask = e * dpe
        f = frame_for_mask_angle(w_mask, start_eff, oscillation_range, starting_frame, flip_time, avail, half_frame)
        pick = dict(percentile=pc, entry=e, mask_angle=w_mask, omega_xds=(s_sign * w_mask) % 360.0, frame=None, files=[])
        if f is not None:
            i0 = int(np.argmin(np.abs(np.asarray(avail) - f)))
            lo = max(0, i0 - n_sum // 2)
            sel = avail[lo: lo + n_sum]
            pick.update(frame=float(f), frames=sel, files=[str(frame_map[x]) for x in sel])
        picks.append(pick)
    # ---- sum the images of each pick (NaN-aware: bad pixels / Bragg peaks excluded)
    for pick in picks:
        if not pick["files"]:
            continue
        acc = cnt = None
        for fp in pick["files"]:
            im = prepare_image(load_image(fp), params.bad_value, 0, params.spot_clip)
            if acc is None:
                acc = np.zeros(im.shape, np.float32)
                cnt = np.zeros(im.shape, np.float32)
            if im.shape != mframes[pick["entry"] % nz].shape:
                raise ValueError(f"image {fp} is {im.shape} but the oriented mask frame is "
                                 f"{mframes[pick['entry'] % nz].shape}: check --transpose / --flip-* options")
            fin = np.isfinite(im)
            acc += np.where(fin, im, 0.0)
            cnt += fin
        total = np.where(cnt > 0, acc * (len(pick["files"]) / np.maximum(cnt, 1)), np.nan)
        pick["sum_image"] = total
        if verbose:
            print(f"  {pick['percentile']:>3}th: mask {pick['mask_angle']:7.2f} deg (entry {pick['entry']}) -> XDS "
                  f"{pick['omega_xds']:7.2f} deg, frames {pick['frames'][0]}..{pick['frames'][-1]} ({len(pick['files'])} summed)")
        pick.pop("frames", None)
    for pick in picks:
        if not pick["files"] and verbose:
            print(f"  {pick['percentile']:>3}th: mask {pick['mask_angle']:7.2f} deg -> XDS {pick['omega_xds']:7.2f} deg: "
                  f"NOT covered by the images at angle-offset {angle_offset:g}")
    if not any(pk["files"] for pk in picks):
        raise ValueError("none of the three shadow angles is covered by the images "
                         "(check --flip-time / --angle-offset / the frame numbers)")
    # ---- figure
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {-1: "#00d5ff", 0: "#ff2d2d", +1: "#7cfc00"}
    Hm, Wm = next(iter(mframes.values())).shape
    fig, axes = plt.subplots(1, len(picks), figsize=(6.4 * len(picks), 6.4 * Hm / Wm + 2.4), squeeze=False)
    axes = axes[0]
    for ax, pick in zip(axes, picks):
        e = pick["entry"]
        if "sum_image" not in pick:
            ax.text(0.5, 0.5, f"no images cover XDS angle {pick['omega_xds']:.1f} deg", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{pick['percentile']}th percentile - mask {pick['mask_angle']:.1f} deg")
            ax.axis("off")
            continue
        img = pick["sum_image"]
        H, W = img.shape
        sb = max(1, int(math.ceil(max(H, W) / float(max_display_px))))
        disp = _bin_nanmean(img, sb)
        fin = disp[np.isfinite(disp)]
        vmax = float(np.percentile(fin, 99.5)) if fin.size else 1.0
        ax.imshow(disp, cmap="gray", vmin=0.0, vmax=max(vmax, 1e-6), origin="upper",
                  extent=(-0.5, disp.shape[1] * sb - 0.5, disp.shape[0] * sb - 0.5, -0.5), interpolation="nearest")
        handles = []
        for j in (-1, 0, 1):
            fr = mframes.get((e + j * step_e) % nz)
            ang = ((e + j * step_e) % nz) * dpe
            new_off = angle_offset + s_sign * j * step_e * dpe
            if j == 0:
                lab = f"mask {ang:.1f} deg  (current, offset {angle_offset:g})"
            else:
                lab = f"mask {ang:.1f} deg  ({'+' if j > 0 else '-'}{step_e * dpe:g}) -> offset {new_off:g}"
            if fr is not None and fr.any():
                bm = _bin_mask(fr, sb)
                if bm.min() < 0.5 <= bm.max():
                    xs = (np.arange(bm.shape[1]) + 0.5) * sb - 0.5
                    ys = (np.arange(bm.shape[0]) + 0.5) * sb - 0.5
                    ax.contour(xs, ys, bm, levels=[0.5], colors=[colors[j]], linewidths=2.2 if j == 0 else 1.4)
            else:
                lab += "  [no shadow]"
            handles.append(Line2D([0], [0], color=colors[j], lw=2.2 if j == 0 else 1.4, label=lab))
        ax.legend(handles=handles, loc="upper center", fontsize=8, framealpha=0.85, bbox_to_anchor=(0.5, -0.12))
        ax.set_title(f"{pick['percentile']}th percentile of shadow travel\nmask {pick['mask_angle']:.1f} deg = XDS "
                     f"{pick['omega_xds']:.1f} deg, {len(pick['files'])} image{'s' if len(pick['files']) != 1 else ''} summed", fontsize=10)
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
    fig.suptitle(f"angle-offset = {angle_offset:g} deg"
                 f"   (flip_time={flip_time}, flip_v={orient.flip_vertical}, flip_h={orient.flip_horizontal}, "
                 f"transpose={orient.transpose}).   The outline that follows the shadow edge tells the offset.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    if out_png:
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    for pick in picks:
        pick.pop("sum_image", None)
    return dict(procession=proc, picks=picks, angle_offset=float(angle_offset), figure=str(out_png) if out_png else None,
                step_deg=step_e * dpe, n_summed=n_sum)


def _resolve_geometry(args, stream):
    """(starting_angle, oscillation_range, starting_frame, Orientation, header) from the CLI arguments."""
    header = read_hkl_header(args.integrate_hkl) if args.integrate_hkl else {}
    sa = args.starting_angle if args.starting_angle is not None else (float(header["starting_angle"]) if "starting_angle" in header else None)
    osc = args.oscillation_range if args.oscillation_range is not None else (float(header["oscillation_range"]) if "oscillation_range" in header else None)
    sf = args.starting_frame if args.starting_frame is not None else float(header.get("starting_frame", 1.0))
    if sa is None or osc is None:
        raise SystemExit("need INTEGRATE.HKL or both --starting-angle and --oscillation-range")
    if not osc:
        raise SystemExit("OSCILLATION_RANGE is zero/missing; cannot map images to mask entries")
    _, rows, cols = stream.shape
    transpose = bool(args.transpose)
    if "nx" in header and "ny" in header:
        nx, ny = int(float(header["nx"])), int(float(header["ny"]))
        if (rows, cols) == (nx, ny) and (rows, cols) != (ny, nx):
            transpose = True                      # the mask is NX x NY: swap, as the old loader did
            print(f"note: mask frames are {rows} x {cols} = NX x NY in the header -> transposing")
        if Orientation(transpose).shape(rows, cols) != (ny, nx):
            raise SystemExit(f"mask frame {rows} x {cols} does not match the detector NY x NX = {ny} x {nx}")
    return sa, osc, sf, Orientation(transpose, args.flip_vertical, args.flip_horizontal), header


# --------------------------------------------------------------------------- shadow masking of the diffraction images
def mask_images(mask_source, image_paths, frame_numbers, out_dir, *, starting_angle, oscillation_range,
                starting_frame=1.0, angle_offset=0.0, flip_time=False, half_frame=True,
                orient=Orientation(), key="mask", deg_per_entry=None, masked_value=-1.0, verbose=True):
    """Populate <out_dir> with the diffraction images, the shadowed pixels set to `masked_value`.

    For every image the per-degree mask entry that matches its omega is looked up (the same
    mapping as in refine: omega = STARTING_ANGLE + angle_offset + osc * (frame - STARTING_FRAME + 0.5),
    --flip-time negates omega first, --transpose/--flip-vertical/--flip-horizontal bring the mask
    into the image orientation).  An image whose entry carries no shadow gets a SYMLINK to the
    original file; an image with any shadowed pixel gets a copy in its original format with those
    pixels set to masked_value (default -1, the XDS 'bad pixel' sentinel).  The mask is streamed
    once, so any number of images can be masked without holding more than one mask entry in memory.

    Returns a report dict(n_images, n_linked, n_copied, out_dir, ...).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = MaskStream(mask_source, key)
    nz, rows, cols = stream.shape
    meta = stream.small_arrays()
    if deg_per_entry:
        dpe = float(deg_per_entry)
    elif "omega_deg" in meta and np.asarray(meta["omega_deg"]).size > 1 and abs(float(np.diff(meta["omega_deg"])[0])) > 0:
        dpe = abs(float(np.diff(meta["omega_deg"])[0]))
    else:
        dpe = 360.0 / nz
    omega_xds = np.array([image_omega(f, starting_angle + angle_offset, oscillation_range, starting_frame, half_frame)
                          for f in frame_numbers])
    omega_mask = np.array([mask_angle(o, flip_time) for o in omega_xds])
    ent, dev = assign_entries(omega_mask, nz, dpe)
    by_entry = defaultdict(list)
    for p, k, d in zip(image_paths, ent, dev):
        by_entry.setdefault(int(k), []).append((Path(p), float(d)))

    def emit_link(src, dst):
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(src.resolve(), dst)
            return True
        except OSError:                              # filesystem without symlink support: fall back to a copy
            shutil.copy2(src, dst)
            return False

    n_linked = n_copied = 0
    linked = []
    copied = []
    if verbose:
        print(f"mask: {nz} entries of {dpe:g} deg, {rows} x {cols} px; {len(image_paths)} images -> {out_dir}")
    for k, fr in stream.iter_frames():
        if k not in by_entry:
            continue
        fr = orient.apply(fr)
        if not fr.any():                             # no shadow at this omega: keep the original file
            for p, d in by_entry[k]:
                n_linked += emit_link(p, out_dir / p.name)
                linked.append(str(p))
            continue
        for p, d in by_entry[k]:
            arr = load_image(p)
            if arr.shape != fr.shape:
                raise SystemExit(f"image {p.name} is {arr.shape} but the oriented mask frame is {fr.shape}: "
                             "check --transpose / the mask orientation")
            arr = np.asarray(arr, dtype=float)
            arr[fr] = masked_value
            save_image(out_dir / p.name, arr)
            n_copied += 1
            copied.append(str(p))
    if verbose:
        print(f"masked {n_copied} image(s), symlinked {n_linked} (no shadow) -> {out_dir}")
    return dict(n_images=len(image_paths), n_copied=n_copied, n_linked=n_linked, copied=copied, linked=linked,
                out_dir=str(out_dir), deg_per_entry=dpe, starting_angle=float(starting_angle),
                oscillation_range=float(oscillation_range), starting_frame=float(starting_frame),
                angle_offset_deg=float(angle_offset), flip_time=bool(flip_time),
                orientation=dict(transpose=orient.transpose, flip_vertical=orient.flip_vertical,
                                 flip_horizontal=orient.flip_horizontal))


# --------------------------------------------------------------------------- command line
_KV = re.compile(r"([A-Za-z0-9_.\-]+)=\s*([^\s=]+)")


def read_hkl_header(path):
    """Parse the '!KEY= value' lines of an XDS INTEGRATE.HKL header (keys lowercased)."""
    header = {}
    with open(path, "r") as fh:
        for line in fh:
            if not line.startswith("!"):
                break
            body = line[1:].strip()
            if body.upper().startswith("END_OF_HEADER"):
                break
            for key, val in _KV.findall(body):
                header.setdefault(key.lower(), val)
    return header


def _add_refine_params(p):
    d = RefineParams()
    p.add_argument("--rect-width", type=int, default=d.rect_width, help="integration rectangle width along the mask tangent (px)")
    p.add_argument("--rect-length", type=int, default=d.rect_length, help="integration rectangle length along the normal (px)")
    p.add_argument("--step", type=float, default=d.step, help="spacing of samples along the mask curve (px)")
    p.add_argument("--edge-frac", type=float, default=d.edge_frac, help="sigmoid level taken as 'plateau reached'")
    p.add_argument("--max-shift", type=float, default=d.max_shift,
                   help="largest accepted edge shift from the expected position (px); larger fits are rejected")
    p.add_argument("--smooth", type=int, default=d.smooth, help="median window along the curve smoothing the shifts (samples); <=1 off")
    p.add_argument("--min-amp", type=float, default=d.min_amp, help="minimum sigmoid amplitude (integrated counts) to accept a sample")
    p.add_argument("--bad-value", type=float, default=d.bad_value, help="defective pixel value (NaN always excluded)")
    p.add_argument("--min-valid-frac", type=float, default=d.min_valid_frac, help="min fraction of valid width samples / images for a profile point")
    p.add_argument("--refine-border", action="store_true", help=argparse.SUPPRESS)  # deprecated, ignored
    p.add_argument("--keep-profiles", type=int, default=d.keep_profiles, help="number of profiles + fits shown in the QC plot")
    p.add_argument("--skip-bad-radius", type=float, default=d.skip_bad_radius, help="skip samples whose centre is within this many px of a bad pixel")
    p.add_argument("--interp-order", type=int, default=d.interp_order, help="polynomial order for interpolating the shift between fitted samples (1 = linear)")
    p.add_argument("--edge-margin", type=int, default=d.edge_margin, help="outermost detector rows/columns excluded from the integration (px)")
    p.add_argument("--min-side-bins", type=int, default=d.min_side_bins,
                   help="valid profile bins required on each side of the edge; shorter profiles (at the detector edge) are not fitted")
    p.add_argument("--anchor-n", type=int, default=d.anchor_n, help="edge anchor = median of the nearest N accepted fits at each open curve end")
    p.add_argument("--outlier-sigma", type=float, default=d.outlier_sigma, help="reject fits this many robust sigma from their neighbours along the curve (0 = off)")
    p.add_argument("--min-fit-frac", type=float, default=d.min_fit_frac, help="a curve needs this fraction of its samples fitted, otherwise it is left unchanged")
    p.add_argument("--tangent-smooth", type=float, default=d.tangent_smooth, help="smoothing length of the curve for the tangent/normal (px)")
    p.add_argument("--no-omega-prior", action="store_true", help="do not centre the rectangles on the edge expected at each image's exact omega")
    p.add_argument("--max-prior", type=float, default=d.max_prior, help="clamp of the expected edge displacement (px)")
    p.add_argument("--isolation", type=float, default=d.isolation_px,
                   help="skip samples with a second mask edge within this distance along the normal (px; 0 = off)")
    p.add_argument("--corner-tol", type=float, default=d.corner_tol,
                   help="skip samples whose edge is not straight across the rectangle (tolerance px)")
    p.add_argument("--spot-clip", default="auto", help="Bragg-peak clipping: 'auto' (default), counts per pixel, or 'off'")


def _params_from_args(args):
    clip = args.spot_clip
    if str(clip).lower() in ("off", "none", "0"):
        clip = None
    elif str(clip).lower() != "auto":
        clip = float(clip)
    return RefineParams(rect_width=args.rect_width, rect_length=args.rect_length, step=args.step,
                        edge_frac=args.edge_frac, max_shift=args.max_shift, smooth=args.smooth,
                        min_amp=args.min_amp, bad_value=args.bad_value, min_valid_frac=args.min_valid_frac,
                        keep_profiles=args.keep_profiles, skip_bad_radius=args.skip_bad_radius,
                        interp_order=args.interp_order, edge_margin=args.edge_margin,
                        min_side_bins=args.min_side_bins, anchor_n=args.anchor_n,
                        outlier_sigma=args.outlier_sigma, min_fit_frac=args.min_fit_frac,
                        tangent_smooth=args.tangent_smooth, use_prior=not args.no_omega_prior,
                        max_prior=args.max_prior, spot_clip=clip,
                        isolation_px=args.isolation, corner_tol=args.corner_tol)


def build_parser():
    p = argparse.ArgumentParser(
        prog="xds_shadow_refine.py",
        description="Refine a theoretical per-degree shadow mask (npz) against the raw diffraction images and "
                    "write a refined npz. Standalone: needs only numpy + scipy; h5py/fabio/imageio/PIL/matplotlib "
                    "are optional extras.")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("refine", help="refine a per-degree mask npz against a set of images")
    r.add_argument("integrate_hkl", nargs="?", default=None,
                   help="INTEGRATE.HKL supplying STARTING_ANGLE / OSCILLATION_RANGE / STARTING_FRAME / NX / NY (optional)")
    r.add_argument("--images", required=True, help="directory, glob, list .txt or single image of raw diffraction frames (one 2-D frame per file)")
    r.add_argument("--mask", required=True, help="mask npz with array 'mask' (entries, rows, cols), True = shadowed")
    r.add_argument("--key", default="mask")
    r.add_argument("--masks-out", default="refined_mask.npz", help="refined npz to write (same layout as the input)")
    r.add_argument("--report", default=None, help="json report with per-entry statistics")
    r.add_argument("--qc-image", default=None,
                   help="QC png: summed images at the given percentiles of the refined entries, original + refined mask "
                        "outlines and the fitted background profiles (default: <masks-out>_qc.png)")
    r.add_argument("--no-qc-image", action="store_true", help="do not write the QC png")
    r.add_argument("--qc-percentiles", default="25,50,75", help="percentiles of the refined entries shown in the QC png")
    r.add_argument("--qc-max-px", type=int, default=1400, help="QC overview: images binned to at most this many px on the long side")
    r.add_argument("--qc-zoom-px", type=int, default=300, help="QC zoom window size in full-resolution pixels")
    r.add_argument("--deg-per-entry", type=float, default=None, help="degrees per mask entry (default: from omega_deg in the npz, else 360/n)")
    r.add_argument("--starting-angle", type=float, default=None)
    r.add_argument("--oscillation-range", type=float, default=None)
    r.add_argument("--starting-frame", type=float, default=None)
    r.add_argument("--angle-offset", type=float, default=0.0, help="offset (deg) added to STARTING_ANGLE (before the time flip)")
    r.add_argument("--flip-vertical", "--flip-y", dest="flip_vertical", action="store_true", help="flip the starting mask upside down")
    r.add_argument("--flip-horizontal", "--flip-x", dest="flip_horizontal", action="store_true", help="mirror the starting mask left/right")
    r.add_argument("--transpose", action="store_true", help="swap rows/cols of the starting mask (automatic if the header shows NX x NY)")
    r.add_argument("--flip-time", action="store_true", help="mask entry k holds omega = -k deg (mask 350 = 10 deg)")
    r.add_argument("--no-half-frame", action="store_true", help="use the end of the exposure as the image angle instead of its centre")
    r.add_argument("--output-orientation", choices=["input", "image"], default="input",
                   help="'input': refined npz keeps the orientation of the input (default); 'image': spatial flips applied")
    r.add_argument("--max-dev-deg", type=float, default=None, help="only use images within this many degrees of the entry angle (default: half an entry)")
    r.add_argument("--min-images", type=int, default=1, help="entries with fewer images are copied unchanged")
    r.add_argument("--entries", default=None, help="only refine entries START:STOP (others are copied), e.g. 200:260")
    r.add_argument("--jobs", type=int, default=1, help="worker processes (entries are refined in parallel)")
    r.add_argument("--frame-regex", default=None, help="regex for the frame number in image file names (default: last digit group)")
    _add_refine_params(r)
    t = sub.add_parser("tune-offset", help="figure for tuning --angle-offset: 3 summed images with the mask outlines on top")
    t.add_argument("integrate_hkl", nargs="?", default=None, help="INTEGRATE.HKL (optional, see refine)")
    t.add_argument("--images", required=True)
    t.add_argument("--mask", required=True)
    t.add_argument("--key", default="mask")
    t.add_argument("--out", default="tune_angle_offset.png", help="figure to write (png)")
    t.add_argument("--show", action="store_true", help="open the figure interactively (zoomable) as well")
    t.add_argument("--starting-angle", type=float, default=None)
    t.add_argument("--oscillation-range", type=float, default=None)
    t.add_argument("--starting-frame", type=float, default=None)
    t.add_argument("--angle-offset", type=float, default=0.0, help="the offset to judge (deg, added to STARTING_ANGLE before the time flip)")
    t.add_argument("--flip-vertical", "--flip-y", dest="flip_vertical", action="store_true")
    t.add_argument("--flip-horizontal", "--flip-x", dest="flip_horizontal", action="store_true")
    t.add_argument("--transpose", action="store_true")
    t.add_argument("--flip-time", action="store_true")
    t.add_argument("--no-half-frame", action="store_true")
    t.add_argument("--frame-regex", default=None)
    t.add_argument("--deg-per-entry", type=float, default=None)
    t.add_argument("--percentiles", default="25,50,75", help="percentiles of the shadow travel to show (default 25,50,75)")
    t.add_argument("--step-deg", type=float, default=5.0, help="angle step of the neighbouring outlines (default 5 deg)")
    t.add_argument("--sum-window-deg", type=float, default=1.0, help="rotation range of the images summed per panel (default 1 deg)")
    t.add_argument("--sum-images", type=int, default=None, help="number of images summed per panel (overrides --sum-window-deg)")
    t.add_argument("--max-display-px", type=int, default=1400, help="images are binned to at most this many pixels on the long side")
    t.add_argument("--no-cache", action="store_true", help="do not cache which mask entries contain shadow")
    t.add_argument("--spot-clip", default="auto", help="Bragg-peak clipping of the summed images: 'auto', counts, or 'off'")
    d = sub.add_parser("refine-debug", help="debug: refine one entry of a per-degree mask npz (or a single-frame mask npz) against 1/3/5 sequential cbf images")
    d.add_argument("mask", help="mask npz: one per-degree (entries, rows, cols) array or a single (rows, cols) frame, True = shadowed")
    d.add_argument("images", nargs="+", help="1, 3 or 5 sequential images (.npy/.npz/.h5/.cbf/.tif/.png); refined as one entry")
    d.add_argument("--key", default="mask", help="array name of the mask in the npz (default 'mask')")
    d.add_argument("--entry", type=int, default=None,
                   help="entry index to refine for a per-degree mask (default: the middle of the shadow procession)")
    d.add_argument("-o", "--out", default="refined_mask.npz", help="output mask npz, same layout / entry order as the input")
    d.add_argument("--qc-max-px", type=int, default=1400, help="QC overview: images binned to at most this many px on the long side")
    d.add_argument("--qc-zoom-px", type=int, default=300, help="QC zoom window size in full-resolution pixels")
    d.add_argument("--qc-n-traces", type=int, default=12, help="QC profiles row: number of kept profiles drawn")
    d.add_argument("--plot", default=None,
                   help="QC png in the style of the refine QC figure: overview + zoom + fitted background profiles "
                        "(default: <out>_qc.png; with --grid-search: the grid-search figure instead)")
    d.add_argument("--no-qc-image", action="store_true", help="do not write the QC png")
    d.add_argument("--grid-search", action="store_true",
                   help="first evaluate the refinement over a grid of summed-frame counts x rectangle widths and print a table")
    d.add_argument("--gs-frames", default="1,3,5", help="comma-separated frame counts to sum for the grid search (default: 1,3,5)")
    d.add_argument("--gs-widths", default="13,27,35", help="comma-separated rectangle widths for the grid search (default: 13,27,35)")
    d.add_argument("--grid-report", default=None, help="write the grid-search table as json here")
    _add_refine_params(d)
    msk = sub.add_parser("mask-images",
                        help="populate a folder with the diffraction images, shadowed pixels set to -1 "
                             "(symlinks to the originals where no masking is needed)")
    msk.add_argument("integrate_hkl", nargs="?", default=None,
                     help="INTEGRATE.HKL supplying STARTING_ANGLE / OSCILLATION_RANGE / STARTING_FRAME / NX / NY (optional)")
    msk.add_argument("--images", required=True,
                     help="directory, glob, list .txt or single image of raw diffraction frames (one 2-D frame per file)")
    msk.add_argument("--mask", required=True, help="refined mask npz with array 'mask' (entries, rows, cols), True = shadowed")
    msk.add_argument("--key", default="mask")
    msk.add_argument("--out-dir", default="images_masked", help="folder to populate (default: ./images_masked)")
    msk.add_argument("--masked-value", type=float, default=-1.0,
                    help="value written to the shadowed pixels (default: -1, the XDS 'bad pixel' sentinel)")
    msk.add_argument("--starting-angle", type=float, default=None)
    msk.add_argument("--oscillation-range", type=float, default=None)
    msk.add_argument("--starting-frame", type=float, default=None)
    msk.add_argument("--angle-offset", type=float, default=0.0, help="offset (deg) added to STARTING_ANGLE (before the time flip)")
    msk.add_argument("--flip-vertical", "--flip-y", dest="flip_vertical", action="store_true", help="flip the mask upside down")
    msk.add_argument("--flip-horizontal", "--flip-x", dest="flip_horizontal", action="store_true", help="mirror the mask left/right")
    msk.add_argument("--transpose", action="store_true", help="swap rows/cols of the mask (automatic if the header shows NX x NY)")
    msk.add_argument("--flip-time", action="store_true", help="mask entry k holds omega = -k deg (mask 350 = 10 deg)")
    msk.add_argument("--no-half-frame", action="store_true", help="use the end of the exposure as the image angle instead of its centre")
    msk.add_argument("--deg-per-entry", type=float, default=None, help="degrees per mask entry (default: from omega_deg in the npz, else 360/n)")
    msk.add_argument("--frame-regex", default=None, help="regex for the frame number in image file names (default: last digit group)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "refine":
        stream = MaskStream(args.mask, args.key)
        sa, osc, sf, orient, header = _resolve_geometry(args, stream)
        qc_default = args.qc_image or str(Path(args.masks_out).with_suffix("")) + "_qc.png"
        qc_path = None if args.no_qc_image else qc_default
        paths = exclude_files(collect_images(args.images), args.masks_out, qc_default, args.report, args.mask)
        frames = [frame_number_from_name(p, args.frame_regex) for p in paths]
        entries = None
        if args.entries:
            a, b = args.entries.split(":")
            entries = (int(a), int(b))
        refine_dataset(args.mask, args.masks_out, paths, frames, starting_angle=sa, oscillation_range=osc,
                       starting_frame=sf, angle_offset=args.angle_offset, flip_time=args.flip_time,
                       half_frame=not args.no_half_frame, orient=orient, output_orientation=args.output_orientation,
                       params=_params_from_args(args), key=args.key, deg_per_entry=args.deg_per_entry,
                       max_dev_deg=args.max_dev_deg, min_images=args.min_images, entries=entries, jobs=args.jobs,
                       report_path=args.report, qc_image=qc_path,
                       qc_percentiles=tuple(float(x) for x in args.qc_percentiles.split(",") if x.strip()),
                       qc_max_px=args.qc_max_px, qc_zoom_px=args.qc_zoom_px)
        return 0
    if args.command == "tune-offset":
        stream = MaskStream(args.mask, args.key)
        sa, osc, sf, orient, header = _resolve_geometry(args, stream)
        paths = exclude_files(collect_images(args.images), args.out, args.mask)
        frames = [frame_number_from_name(p, args.frame_regex) for p in paths]
        clip = None if str(args.spot_clip).lower() in ("off", "none", "0") else (
            "auto" if str(args.spot_clip).lower() == "auto" else float(args.spot_clip))
        tune_angle_offset(args.mask, paths, frames, starting_angle=sa, oscillation_range=osc, starting_frame=sf,
                          angle_offset=args.angle_offset, flip_time=args.flip_time, half_frame=not args.no_half_frame,
                          orient=orient, key=args.key, deg_per_entry=args.deg_per_entry,
                          percentiles=tuple(float(x) for x in args.percentiles.split(",") if x.strip()),
                          step_deg=args.step_deg, sum_window_deg=args.sum_window_deg, sum_images=args.sum_images,
                          max_display_px=args.max_display_px, out_png=args.out, show=args.show,
                          cache=not args.no_cache, params=RefineParams(spot_clip=clip))
        print(f"figure -> {args.out}")
        return 0
    if args.command == "mask-images":
        stream = MaskStream(args.mask, args.key)
        sa, osc, sf, orient, header = _resolve_geometry(args, stream)
        paths = exclude_files(collect_images(args.images), args.out_dir, args.mask)
        frames = [frame_number_from_name(p, args.frame_regex) for p in paths]
        mask_images(args.mask, paths, frames, args.out_dir, starting_angle=sa, oscillation_range=osc,
                    starting_frame=sf, angle_offset=args.angle_offset, flip_time=args.flip_time,
                    half_frame=not args.no_half_frame, orient=orient, key=args.key,
                    deg_per_entry=args.deg_per_entry, masked_value=args.masked_value)
        return 0
    if args.command == "refine-debug":
        params = _params_from_args(args)
        qc_png = args.plot if args.plot else str(Path(args.out).with_suffix("")) + "_qc.png"
        if args.no_qc_image:
            qc_png = None
        frame, meta, entry, angle, stream = _load_debug_mask(args.mask, args.key, args.entry)
        imgs = [prepare_image(load_image(p), params.bad_value, params.edge_margin, params.spot_clip)
                for p in args.images]
        if imgs[0].shape != frame.shape:
            raise SystemExit(f"mask {args.mask} is {frame.shape} but image {args.images[0]} is {imgs[0].shape}: "
                             "check --transpose / the mask orientation")
        stack = [(im, 0.0) for im in imgs]
        if args.grid_search:
            frames = [int(x) for x in args.gs_frames.split(",") if x.strip()]
            widths = [int(x) for x in args.gs_widths.split(",") if x.strip()]
            rows_ = grid_search(args.images, frame, params, frame_counts=frames, rect_widths=widths)
            print(f"{'frames':>6} {'width':>5} {'valid':>9} {'median_shift':>12} {'MAD':>7} {'fit_rms':>8} {'sec':>6}")
            for r in rows_:
                med = f"{r['median_shift']:.2f}" if r['median_shift'] is not None else "nan"
                mad = f"{r['shift_mad']:.2f}" if r['shift_mad'] is not None else "nan"
                print(f"{r['frames']:>6} {r['rect_width']:>5} {r['n_valid']:>4}/{r['n_samples']:<4} "
                      f"{med:>12} {mad:>7} {r['fit_rms']:>8.4f} {r['seconds']:>6.2f}")
            if args.grid_report:
                with open(args.grid_report, "w") as fh:
                    json.dump(rows_, fh, indent=2)
            grid_png = qc_png
            if grid_png:
                grid_figure(rows_, grid_png)
                qc_png = None                         # the QC figure is the grid figure in this mode
                print(f"grid figure -> {grid_png}")
        res = refine_mask_frame(None, frame, params, stack=stack)
        if stream is None:                            # single-frame input: write the refined frame back
            with np.load(args.mask) as z:
                use = args.key if args.key in z.files else z.files[0]
            np.savez_compressed(args.out, **{use: res.mask}, **meta)
        else:                                         # per-degree input: replace the refined entry, copy the rest
            nz, rows_, cols_ = stream.shape
            writer = NpzStreamWriter(args.out, nz, rows_, cols_)
            for k, fr in stream.iter_frames():
                writer.write(res.mask if k == entry else fr)
            writer.close(meta)
        if qc_png:
            exp = res.expansions[np.isfinite(res.expansions)]
            med = float(np.median(exp)) if exp.size else float("nan")
            item = dict(label="refine-debug", entry=0 if entry is None else entry, mask_angle=angle,
                        files=list(args.images), orig=frame, refined=res.mask, median_expansion=med,
                        n_fitted=res.n_valid, n_samples=int(res.shifts.size), result=res)
            refine_qc_figure([item], params, qc_png, max_display_px=args.qc_max_px,
                             zoom_px=args.qc_zoom_px, n_traces=args.qc_n_traces,
                             title=f"refine-debug - {Path(args.mask).name} entry "
                                   f"{'(single frame)' if entry is None else entry} vs {len(args.images)} image(s)")
        exp = res.expansions[np.isfinite(res.expansions)]
        med = float(np.median(exp)) if exp.size else float("nan")
        print(f"refine-debug: {'entry ' + str(entry) + ', ' if entry is not None else ''}{len(args.images)} image(s)"
              f" stacked, {res.n_valid}/{res.shifts.size} edge samples fitted, "
              f"median shift {med:.2f} px -> {args.out}"
              + (f"; QC image -> {qc_png}" if qc_png else ""))
        print("  stats: " + " ".join(f"{k}={v}" for k, v in res.stats.items()))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
