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
   enforces continuity across frames.
5. **Geometric prior** (mini-kappa, detector origin top-left): the support
   shadow enters from max Y and, with increasing frame number, moves toward
   min X (`dx/dz < 0`). Components are scored by their per-frame centroid
   trajectory; static components near `(ORGX, ORGY)` (the beamstop) are
   rejected.
6. **Mask construction**: dilation (radius `r`) of the selected component,
   restricted to a band around the fitted trajectory `x(z), y(z)`; frames
   where the component is missing are filled from the linear fit.
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
    --kernel 31 --k 5.0 --entry-frac 0.8 --radius 3 --band 15
xdsmask apply INTEGRATE.HKL --masks masks.npz --filter-out FILTER.HKL \
    --out INTEGRATE.masked.HKL
```

Masks are stored as a boolean array of shape `(n_frames, NY, NX)`
(`True` = keep) in an `.npz` file under the key `masks`.

## Tests

```bash
python -m pytest
```

The tests build a synthetic `INTEGRATE.HKL` and a synthetic 3D voxel volume
containing a moving low-signal tube plus a static beamstop blob, and verify
that the moving tube is selected and masked while the beamstop is not.
