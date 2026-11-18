# xdsmask

Detect per-frame low-signal detector regions (e.g. the moving shadow of a
metal goniometer support) from an XDS `INTEGRATE.HKL` file, and mask the
affected reflections either through a generated `FILTER.HKL` (native XDS
exclusion during CORRECT) or a filtered copy of `INTEGRATE.HKL`.

XDS's own masking (`UNTRUSTED_RECTANGLE`/`UNTRUSTED_ELLIPSE` in DEFPIX) is
static per detector pixel and cannot follow a shadow that moves across frames.
`INTEGRATE.HKL` already contains the full 3D point cloud of the experiment —
every predicted reflection with its calculated detector position
(`XCAL, YCAL`) and image number (`ZCAL`) — so the moving shadow appears as a
connected tube of low-signal voxels in `(frame, y, x)` space and can be found
without any shape prior.

## Pipeline

1. **Parse** `INTEGRATE.HKL`: header `!`-lines (key=value) plus free-format
   records; the item order is read from the `!H,K,L,IOBS,...` header line.
2. **Voxelize** reflections into a `(n_frames, NY, NX)` grid using the
   calculated coordinates: frame index `f = round(ZCAL) - STARTING_FRAME`,
   pixel indices 1-based (`ix = round(XCAL)`, `iy = round(YCAL)`).
   Per voxel: reflection count, mean `log1p(MAXC)` (or `IOBS`), fraction
   unobserved (`XOBS == 0`).
3. **Local-contrast score** per frame:
   `score = value - median_filter(value, size=(1, K, K))`. Resolution and
   Lorentz/polarization falloff are smooth in `(x, y)`, the shadow is
   localized, so the shadow stands out as a strongly negative residual
   anywhere on the detector.
4. **Robust threshold** per frame: `score < median - k * MAD`, then 3D
   connected-component labeling (26-connectivity in `(frame, y, x)`); only
   components above a minimum voxel count and frame span are kept, which
   enforces continuity across frames. Before labeling, the low mask is
   dilated by one voxel in all directions (`--no-pre-dilate` to disable) so
   a fast shadow that moves more than one block per frame still forms a
   single connected tube.
5. **Geometric prior** (detector origin top-left): by default the support
   shadow enters from the bottom (max Y) and, with increasing frame number,
   moves left (toward min X, `dx/dz < 0`). For inverse geometry use
   `--entry-edge top --motion right` (enters from min Y, moves toward max X).
   Components are scored by their per-frame centroid trajectory; static
   components near `(ORGX, ORGY)` (the beamstop) are rejected.
6. **Mask construction**: dilation (radius `r`) of the selected component,
   restricted to a band around the trajectory `x(z), y(z)`. The trajectory
   is interpolated (clamped) only within the frame span where the component
   was detected — outside that span nothing is masked and the report's
   `trajectory_px` is `null`, so a shadow crossing the detector in only part
   of the dataset is never extrapolated to absurd off-detector positions
   (`span_frames` in the report gives the covered frame range).
7. **Apply**: a reflection is masked if
   `mask[f, iy-1, ix-1] == 0` with `f, ix, iy` from its `ZCAL, XCAL, YCAL`.
   Reflections off-detector or outside the mask stack are treated as masked.
   Masked reflections are written to `FILTER.HKL` as
   `H K L XCAL YCAL ZCAL DX DY DZ` (XDS matches these against the calculated
   columns, so tolerances of 0.5 are sufficient) — place the file in the XDS
   working directory and rerun CORRECT.

## Install

```bash
pip install -e .          # runtime deps: numpy, scipy
pip install -e .[test]    # adds pytest
```

## Usage

```bash
xdsmask detect INTEGRATE.HKL --masks-out masks.npz --report report.json \
    --kernel 31 --k 5.0 --entry-frac 0.8 --radius 3 --band 15 \
    --entry-edge bottom --motion left        # inverse geometry: --entry-edge top --motion right
xdsmask apply INTEGRATE.HKL --masks masks.npz --filter-out FILTER.HKL \
    --out INTEGRATE.masked.HKL
xdsmask plot --masks masks.npz --integrate-hkl INTEGRATE.HKL \
    --report report.json --out report.html
```

`plot` writes a self-contained interactive HTML (Plotly): a frame slider
(plus Play button) showing the mask overlay (red = masked) and the
reflections of each frame (red = masked, blue = kept, hover for hkl), with
the fitted shadow trajectory and its per-frame centroid.

Masks are stored as a boolean array (`True` = keep) in an `.npz` file under
the key `masks`, together with the integer `block` (pixel block size).

## Orientation: --flip-x / --flip-y

XDS pixel coordinates are 1-based; by default mask/npz array column/row 0
corresponds to `XCAL=1` / `YCAL=1` (origin top-left as displayed). If your
mask stack is stored with an inverted axis, pass `--flip-x` and/or
`--flip-y` to `detect` and `apply`: the block indices are mirrored
(`bx = nxb-1-bx`, `by = nyb-1-by`) before lookup, so the array is indexed
directly without pre-flipping it. With `--flip-y`, `--entry-edge top/bottom`
refers to the mask array's rows (top = min array row). The 3D `plot` always
shows the mask array as stored.

## Cyclical z offset

`z` (frame number) is cyclical: a dataset with `STARTING_ANGLE= 326.1` and
`OSCILLATION_RANGE= 0.1` has frame `f` at absolute angle `326.1 + 0.1·f`. To
align reflections with a mask stack whose frame 0 is angle 0, pass
`--z-offset-deg 326.1` to `detect`/`apply`: the offset is converted to frames
via `OSCILLATION_RANGE` and wrapped modulo the number of frames
(`f_mask = (round(ZCAL) − STARTING_FRAME + offset_frames) mod n_frames`), so
wraparound at 360° is handled. Requires `OSCILLATION_RANGE` in the header.

## Memory

The voxel grid is `(n_frames, NY/block, NX/block)`. `block` is chosen
automatically (powers of two) so the grid stays within `--voxel-budget`
voxels (default 2^25 ≈ 33M, ~1 GB total); override with `--block` /
`--voxel-budget`. This keeps large detector × many-frame datasets in RAM;
the shadow of a support is tens of pixels wide, so block binning does not
lose it. `apply` reads `block` from the `.npz` and maps each reflection to
its block, so masks are applied at full pixel precision without ever
materializing a full-resolution `(n_frames, NY, NX)` array.

## Tests

```bash
python -m pytest
```

The tests build a synthetic `INTEGRATE.HKL` and a synthetic 3D voxel volume
containing a moving low-signal tube plus a static beamstop blob, and verify
that the moving tube is selected and masked while the beamstop is not.
