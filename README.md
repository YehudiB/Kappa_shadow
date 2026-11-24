# xdsmask

Tools for handling the detector shadow cast by the sample pin and the phi arm
of a mini-kappa goniometer: visualise the shadow in the browser, refine the
theoretical shadow mask against the collected diffraction images, and exclude
the shadowed reflections from XDS.

## Why

On a mini-kappa goniometer, when opening up the kappa arm, the mounting pin 
and the phi arm can end up in the beam path and cast a moving shadow on the 
detector as omega advances. The figure below shows the combinations of kappa
and omega that will end up casting a shadow on a 16M Eiger2 at 154 mm distance.

<img src="example_output/shadow_map.png" alt="Shadow map" width="400" height="400">

Reflections that diffract into this shadow will not reach the detector, which
distorts scaling and merging. XDS by itself can onlydeal with static shadows 
(e.g. beamstop), so the fix is to tell it exactly which reflections to ignore.
XDS supports this natively: place a `FILTER.HKL` next to the data and CORRECT 
ignores the listed reflections.

This repository provides the three steps of that workflow:

| File | Role |
| --- | --- |
| `mini-kappa-goniometer-pin-shadow-visualizer.html` | Self-contained browser tool for the omega / kappa / phi geometry of a mini-kappa goniometer and the shadow the pin and phi arm cast on the detector; exports the shadow as a rotation-resolved theoretical mask (`.npz`, one entry per degree of omega). |
| `pinshadow_profile.py` | Pure-stdlib Python module (no NumPy) that summarises the pin-only shadow for one kappa and detector distance over the full omega sweep: any shadow, the shadow-free omega zone and an ASCII profile; useful to plan a kappa / detector-distance strategy before collecting. |
| `xdsmask_refine.py` | Refines the theoretical mask against the raw diffraction images: the shadow edge is moved to where the fitted low-to-high background transition actually lies. Also finds the rotation-angle offset, writes masked image copies and offers single-entry debugging. |
| `xdsmask_filter.py` | Reads `INTEGRATE.HKL`, looks every reflection up in the (refined) shadow mask and writes a `FILTER.HKL` that makes XDS ignore the shadowed reflections in CORRECT. |

## Pipeline

1. **Theoretical mask.** The
   [visualizer webpage](mini-kappa-goniometer-pin-shadow-visualizer.html)
   draws the omega / kappa / phi three-circle geometry and the pin + phi-arm
   shadow for your beam / pin / detector parameters, and writes the shadow out
   as a theoretical mask: an `.npz` with boolean array `mask[omega_deg, row, col]`
   (`True` = shadowed, one entry per degree of the omega sweep), e.g.
   `mask_omscan_Eiger2X16M_k200.0.npz`.
2. **Angular offset.** The mask must first be aligned with the data in
   omega. `xdsmask_refine.py tune-offset` renders a figure to judge / tune the
   angle offset by eye (`--angle-offset`, plus orientation flags).
3. **Refine. `xdsmask_refine.py refine`** adapts the mask to reality: the
   shadow boundary is walked as a curve, the low-to-high background profile
   across the edge is fitted with a sigmoid from the raw images (aligned
   shift-and-add, so the 1-degree mask sampling and low backgrounds do not
   limit the fit) and the mask edge is moved to where the high plateau is
   reached. Result: a refined `.npz` with the same layout, a JSON report and a
   QC figure.
4. **Filter. `xdsmask_filter.py`** looks every reflection of
   `INTEGRATE.HKL` up in the refined mask (with a little play in omega and in
   x/y) and writes `FILTER.HKL`; re-run CORRECT and the shadowed reflections
   are ignored.

## Example output

Both scripts write a QC (quality-control) figure so the result can be judged
at a glance. The two below are the examples shipped in `example_output/`.

### Mask refinement QC (`example_output/refine_qc.png`)

Written by `xdsmask_refine.py refine` (default name `<masks-out>_qc.png`) for a
few representative mask entries. Three rows per entry:

* **top:** whole detector (binned) with the original mask outline (cyan)
  against the refined outline (red);
* **middle:** a full-resolution zoom around the middle of the refined edge,
  marked in the overview;
* **bottom:** the background profiles kept along the shadow boundary with
  their sigmoid fits and the refined edge position.

![Refinement QC figure](example_output/refine_qc.png)

### FILTER.HKL QC (`example_output/filter_qc.png`)

Written by `xdsmask_filter.py` (default name `filter_qc.png`):

* three panels of diffraction images summed at the 25th / 50th / 75th
  percentile of the shadow travel, with the shadowed mask entry outlined
  (cyan), every `INTEGRATE.HKL` spot as a blue dot and every reflection that
  `FILTER.HKL` removes as a yellow dot;
* below them, the sorted-intensity (iobs) scatter of all excluded
  reflections, with the mean / median of the excluded spots and the mean of
  the kept spots for comparison.

![FILTER.HKL QC figure](example_output/filter_qc.png)

## Install

Python 3 with NumPy is required for the two XDS scripts; SciPy additionally
for `xdsmask_refine.py`:

    pip install numpy scipy

Optional, depending on what you use:

    pip install matplotlib     # QC figures
    pip install fabio          # CBF images
    pip install h5py           # HDF5 images
    pip install imageio        # TIFF / PNG images (or pillow)

The visualizer is a single self-contained HTML file: just open it in a
browser, nothing to install. `pinshadow_profile.py` needs no third-party
packages either, pure Python 3 is enough.

## Usage

### 1. Write the theoretical mask

Open `mini-kappa-goniometer-pin-shadow-visualizer.html` in a browser, set the
motor positions and the beam / pin / phi-arm / detector parameters, and
download the rotation-resolved mask `.npz`. Preparing and downloading the mask
takes a minute or two.

### 2. Find the angular offset

    python xdsmask_refine.py tune-offset INTEGRATE.HKL --images RAW_DIR --mask mask_omscan.npz --flip-vertical --flip-time --angle-offset 0.0 --out tune.png

Due to minor discrepancies between the model and reality, the offset should 
probably be tweaked by a few degrees. Judge this offset from the figure and 
repeat with an updated `--angle-offset` until mask and data mostly agree. 
(`--images` accepts a directory, glob, `.txt` list or single image; frame 
numbers are read from the file names.)

### 3. Refine the mask

    python xdsmask_refine.py refine INTEGRATE.HKL --images RAW_DIR --mask mask_omscan.npz --flip-vertical --flip-time --angle-offset 3.7 --masks-out refined.npz --report refine.json --jobs 4

Outputs `refined.npz` (same layout and orientation as the input mask, so the
same flags stay valid for it), a JSON `--report` of the per-entry statistics
and the QC figure.

### 4. Exclude the shadowed reflections from XDS

    python xdsmask_filter.py INTEGRATE.HKL refined.npz --images RAW_DIR -o FILTER.HKL --qc-image filter_qc.png

Copy `FILTER.HKL` into the XDS processing directory and re-run CORRECT; each
listed reflection is excluded with a small box around it (DX, DY, DZ, default
0.5). Pass `--no-qc-image` to skip the figure, `--omega-margin` (deg) and
`--xy-margin` (px) to control the play allowed when matching reflections to
the mask.

### Planning a kappa / detector-distance strategy (`pinshadow_profile.py`)

Before collecting, you may want to know which kappa and detector distance
keep the pin out of the diffraction pattern. `pinshadow_profile.py` computes
the pin-only shadow for the full 360 deg of omega in a single call (pure
Python 3, ~10 ms per query, so it is computed ad hoc, no lookup table
needed), with the same geometry and conventions as the visualizer with the
phi-arm box unchecked and omega axis 0..360:

    python pinshadow_profile.py 200.5 154
    kappa 200.5 deg, detector 154 mm: pin shadow in 137 of 360 omega steps (worst 12.15 % at omega 261.5 deg)
    shadow-free zone: ([350, 213] deg, 223 deg wide)
    Omega 0: ......................................................:--=+*##%@@@@@@@@@@@@@@%%##*+=-:.... 360

The shadow-free zone wraps around 360 deg: here omega 350 -> 360 followed by
0 -> 213 is clear, a 223 deg wedge. Use it the same way from Python:

    >>> from pinshadow_profile import shadow_profile
    >>> p = shadow_profile(200.5, 154)   # beam / pin / detector parameters can be overridden
    >>> p.any_shadow, p.free, p.free[1] / 360.0
    (True, ([350.0, 213.0], 223.0), 0.6194444444444444)

`p.free` is `([start_deg, end_deg], width_deg)`; kappa is validated against
the mini-kappa range (-4 to 250 deg). Run `python pinshadow_profile.py --help`
for the overridable parameters (pin half-cone angle and mounting rotation,
beam offset, detector dimensions, sampling resolution).

### Helper subcommands (`xdsmask_refine.py`)

* `refine-debug`: refine a single mask entry against one or more images, in
  place, with a QC figure of the fit (`--grid-search` to sweep refinement
  parameters);
* `mask-images`: Alternative to the 'FILTER.HKL' route, this will remove the
  masked regions in the diffraction images. It will write copies of the raw
  images with the shadowed pixels set to the XDS bad-pixel value (default -1).
  XDS should then be rerun on these images. This should yield a better results
  because the INTEGRATE spot profiles will not be contaminated by the masked
  reflections.

## Notes

* **Geometry.** omega(z) = STARTING_ANGLE + OSCILLATION_RANGE * (z - (STARTING_FRAME - 1));
  the mask entry is omega mod 360 (one entry per degree). Values are taken from
  the `INTEGRATE.HKL` header, from `.lp` files, or overridden on the command
  line.
* **Orientation flags.** `--flip-vertical`, `--flip-horizontal`,
  `--transpose`, `--flip-time` and `--angle-offset` mean the same thing in both
  scripts; pass the same ones to both. `xdsmask_refine.py` detects a transposed
  mask automatically. For a normal MD-3 goniometer, no flipping of axes should be
  required. For a hanging goniometer (P14), '--flip-vertical --flip-horizontal --flip-time
  --angle-offset -90' are required.
* **Large masks are streamed.** A 360-entry Eiger2X 16M mask is about 6.5 GB;
  the scripts read it one entry at a time and never load it whole.
* **Image formats.** `.npy`, `.npz`, `.h5`, `.cbf`, `.tif`, `.png`, transparently
  `.gz` / `.bz2` compressed (`xdsmask_refine.py`); one 2D frame per file.


### Contributions

Concept is mine, made possible by DESY. Code mostly by Mistral Medium 3.5 and Large 4.0. 
as well as GLM 5.3 and Sonnet 5.5.
