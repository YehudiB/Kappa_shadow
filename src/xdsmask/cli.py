from __future__ import annotations

import argparse
import json
from pathlib import Path

from .apply import apply_masks, load_masks, load_masks_meta, save_masks
from .detect import DetectionParams, detect_shadow
from .hkl import IntegrateHKL


def _add_refine_params(p):
    p.add_argument("--rect-width", type=int, default=9, help="integration rectangle width along the mask tangent (px)")
    p.add_argument("--rect-length", type=int, default=100, help="integration rectangle length along the normal (px)")
    p.add_argument("--step", type=float, default=2.0, help="spacing of samples along the mask curve (px)")
    p.add_argument("--edge-frac", type=float, default=0.95, help="sigmoid level taken as 'plateau reached'")
    p.add_argument("--max-shift", type=float, default=25.0, help="clamp for the local edge shift (px)")
    p.add_argument("--smooth", type=int, default=5, help="median window along the curve smoothing the shifts (samples); <=1 off")
    p.add_argument("--min-amp", type=float, default=0.0, help="minimum sigmoid amplitude (integrated counts) to accept a sample")
    p.add_argument("--bad-value", type=float, default=-1.0, help="defective pixel value excluded from the rectangle integration (NaN always excluded)")
    p.add_argument("--min-valid-frac", type=float, default=0.5, help="min fraction of valid width samples for a profile point to be used")


def build_parser():
    p = argparse.ArgumentParser(
        prog="xdsmask",
        description="Detect moving low-signal detector regions in XDS INTEGRATE.HKL and mask them via FILTER.HKL.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("detect", help="detect the moving shadow and write per-frame masks")
    d.add_argument("integrate_hkl")
    d.add_argument("--masks-out", required=True)
    d.add_argument("--report", default=None)
    d.add_argument("--value", default="maxc", help="record item used as signal (default: maxc)")
    d.add_argument("--kernel", type=int, default=31, help="median-filter kernel in x,y (made odd)")
    d.add_argument("--k", type=float, default=5.0, help="robust threshold in units of MAD")
    d.add_argument("--min-count", type=int, default=1)
    d.add_argument("--min-voxels", type=int, default=10)
    d.add_argument("--min-span", type=int, default=3)
    d.add_argument("--entry-frac", type=float, default=0.8, help="entry edge: centroid y beyond entry_frac of the edge")
    d.add_argument("--entry-edge", choices=["bottom", "top"], default="bottom",
                   help="edge the shadow enters from (origin top-left: bottom = max Y, top = min Y); default bottom")
    d.add_argument("--motion", choices=["left", "right"], default="left",
                   help="horizontal motion of the shadow with increasing frame (left = toward min X, right = toward max X); default left")
    d.add_argument("--beamstop-radius", type=float, default=50.0)
    d.add_argument("--radius", type=int, default=3, help="dilation radius of the mask (pixels)")
    d.add_argument("--band", type=float, default=15.0, help="band around fitted trajectory (pixels)")
    d.add_argument("--block", type=int, default=0, help="pixel block size for the voxel grid; 0 = auto from --voxel-budget")
    d.add_argument("--voxel-budget", type=int, default=2 ** 25, help="max voxels in the (frames, y, x) grid when --block=0")
    d.add_argument("--no-pre-dilate", action="store_true", help="disable the 3x3x3 pre-dilation that keeps fast-moving shadows connected across frames")
    d.add_argument("--flip-x", action="store_true", help="mirror the x axis between XCAL and the mask array (mask col 0 = max XCAL)")
    d.add_argument("--flip-y", action="store_true", help="mirror the y axis between YCAL and the mask array (mask row 0 = max YCAL); --entry-edge then refers to the mask array's axes")
    d.add_argument("--z-offset-deg", type=float, default=0.0,
                   help="cyclical z shift in degrees (via OSCILLATION_RANGE, wrapped mod n_frames)")
    d.add_argument("--trend", choices=["local", "global", "radial"], default="local",
                   help="reference for the contrast score: local = per-frame median filter over neighbours (default); "
                        "global = per-frame median of all spot intensities; radial = ring median around (ORGX, ORGY)")
    d.add_argument("--radial-bin", type=float, default=0.0, help="radial bin width in px for --trend radial (0 = block size)")
    a = sub.add_parser("apply", help="apply masks: emit FILTER.HKL and/or a filtered INTEGRATE.HKL")
    a.add_argument("integrate_hkl")
    a.add_argument("--masks", required=True, help=".npz written by 'detect'")
    a.add_argument("--filter-out", default=None)
    a.add_argument("--out", default=None)
    a.add_argument("--flip-x", action="store_true", help="mirror the x axis when indexing the mask array")
    a.add_argument("--flip-y", action="store_true", help="mirror the y axis when indexing the mask array")
    a.add_argument("--z-offset-deg", type=float, default=0.0,
                   help="cyclical z shift in degrees when mapping reflections to mask frames (e.g. STARTING_ANGLE if the npz starts at angle 0)")
    a.add_argument("--deg-per-entry", type=float, default=None,
                   help="mask npz z axis is degree-indexed with this many degrees per entry (overrides npz metadata)")
    a.add_argument("--invert", action="store_true", help="mask npz True = masked (overrides npz metadata)")
    e = sub.add_parser("from-npz", help="build FILTER.HKL / masks / HTML from an external per-degree mask npz")
    e.add_argument("integrate_hkl")
    e.add_argument("--masks-in", required=True, help="external npz with array 'mask', shape (nz, y, x), True = masked")
    e.add_argument("--key", default="mask", help="array name in the npz (default: mask)")
    e.add_argument("--deg-per-entry", type=float, default=None,
                   help="degrees covered per z entry of the external npz (default: 360/nz)")
    e.add_argument("--no-invert", action="store_true", help="external True = keep (default: True = masked)")
    e.add_argument("--flip-x", action="store_true", help="external x axis is mirrored relative to XCAL")
    e.add_argument("--flip-y", action="store_true", help="external y axis is mirrored relative to YCAL")
    e.add_argument("--filter-out", default=None)
    e.add_argument("--out", default=None)
    e.add_argument("--masks-out", default=None, help="also save the frame-indexed masks in our npz format")
    e.add_argument("--report", default=None)
    e.add_argument("--plot", default=None, help="write the HTML visualization here")
    r = sub.add_parser("refine", help="refine an external per-degree mask against the raw diffraction images "
                                       "(walk the mask curve, fit a sigmoid to the perpendicular background edge, shift the mask locally)")
    r.add_argument("integrate_hkl", nargs="?", default=None,
                   help="INTEGRATE.HKL supplying STARTING_ANGLE / OSCILLATION_RANGE / STARTING_FRAME / NX / NY (optional)")
    r.add_argument("--images", required=True, help="directory, glob, list .txt or single image of raw diffraction frames")
    r.add_argument("--mask", required=True, help="external npz with array 'mask' (nz, y, x), True = shadowed")
    r.add_argument("--key", default="mask", help="array name in the mask npz (default: mask)")
    r.add_argument("--masks-out", default=None, help="write the refined per-degree mask npz (array 'mask', True = shadowed)")
    r.add_argument("--per-image-out", default=None, help="also write one refined mask per image (npz: mask, entry, frame, path)")
    r.add_argument("--report", default=None)
    r.add_argument("--plot", default=None, help="HTML QC plot (image + theoretical vs refined edge) for the first refined entry")
    r.add_argument("--deg-per-entry", type=float, default=None, help="degrees per z entry of the mask npz (default: 360/nz)")
    r.add_argument("--starting-angle", type=float, default=None)
    r.add_argument("--oscillation-range", type=float, default=None)
    r.add_argument("--starting-frame", type=float, default=None)
    r.add_argument("--z-offset-deg", type=float, default=0.0, help="added to omega before mapping to mask entries")
    r.add_argument("--flip-x", action="store_true", help="external x axis is mirrored relative to the images")
    r.add_argument("--flip-y", action="store_true", help="external y axis is mirrored relative to the images")
    r.add_argument("--frame-regex", default=None, help="regex for the frame number in image file names (default: last digit group)")
    _add_refine_params(r)
    d = sub.add_parser("refine-debug", help="debug: refine a single black/white mask.png against one diffraction image (no npz)")
    d.add_argument("mask_png", help="black/white mask image; the shadowed region is white by default")
    d.add_argument("images", nargs="+",
                   help="1, 3 or 5 sequential diffraction images (.npy/.npz/.h5/.cbf/.tif/.png); they are summed before refining to raise the SNR")
    d.add_argument("-o", "--out", default="refined_mask.png", help="output mask png, same polarity as the input mask")
    d.add_argument("--mask-polarity", choices=["white", "black"], default="white",
                   help="which pixels of mask.png are the shadowed region (default: white; use black for a black-on-white mask)")
    d.add_argument("--plot", default=None, help="HTML QC plot (image + theoretical vs refined edge)")
    _add_refine_params(d)
    v = sub.add_parser("plot", help="render masks and reflections as an interactive HTML report")
    v.add_argument("--masks", required=True, help=".npz written by 'detect'")
    v.add_argument("--report", default=None, help="report.json written by 'detect' (trajectory, block, pixel shape)")
    v.add_argument("--out", required=True, help="output .html file")
    v.add_argument("--max-voxels", type=int, default=200000, help="subsample cap for plotted masked voxels")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "detect":
        hkl = IntegrateHKL.read(args.integrate_hkl)
        params = DetectionParams(
            value=args.value,
            kernel=args.kernel,
            k=args.k,
            min_count=args.min_count,
            min_voxels=args.min_voxels,
            min_span=args.min_span,
            entry_frac=args.entry_frac,
            entry_edge="max_y" if args.entry_edge == "bottom" else "min_y",
            slope_sign=-1 if args.motion == "left" else 1,
            beamstop_radius=args.beamstop_radius,
            radius=args.radius,
            band=args.band,
            block=args.block,
            voxel_budget=args.voxel_budget,
            pre_dilate=not args.no_pre_dilate,
            flip_x=args.flip_x,
            flip_y=args.flip_y,
            z_offset_deg=args.z_offset_deg,
            trend=args.trend,
            radial_bin=args.radial_bin,
        )
        masks, report = detect_shadow(hkl.records, hkl.header, params)
        save_masks(args.masks_out, masks, block=report["block"])
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2))
        sel = report.get("selected")
        print(f"components: {report['n_components']}; block: {report['block']} px; selected label: {sel['label'] if sel else None}; masks -> {args.masks_out}")
        return 0
    if args.command == "apply":
        hkl = IntegrateHKL.read(args.integrate_hkl)
        meta = load_masks_meta(args.masks)
        masks = meta["masks"]
        block = int(meta["block"]) if "block" in meta else 1
        dpe = args.deg_per_entry
        if dpe is None and "deg_per_entry" in meta:
            dpe = float(meta["deg_per_entry"])
        invert = args.invert or bool(meta.get("invert", False))
        masked = apply_masks(hkl, masks, block=block, flip_x=args.flip_x, flip_y=args.flip_y,
                             z_offset_deg=args.z_offset_deg, invert=invert, deg_per_entry=dpe,
                             filter_out=args.filter_out, out=args.out)
        print(f"masked {int(masked.sum())} of {masked.size} reflections (block={block}, flip_x={args.flip_x}, flip_y={args.flip_y}, z_offset_deg={args.z_offset_deg}, deg_per_entry={dpe}, invert={invert})")
        return 0
    if args.command == "from-npz":
        from .external import block_reduce_any, load_external_masks, z_entries
        from .plot import build_figure
        from .voxelize import choose_block, geometry
        hkl = IntegrateHKL.read(args.integrate_hkl)
        arr, dpe = load_external_masks(args.masks_in, hkl.header, key=args.key,
                                       flip_x=args.flip_x, flip_y=args.flip_y,
                                       deg_per_entry=args.deg_per_entry)
        invert = not args.no_invert
        masked = apply_masks(hkl, arr, block=1, invert=invert, deg_per_entry=dpe,
                             filter_out=args.filter_out, out=args.out)
        nx, ny, sf = geometry(hkl.header)
        if args.masks_out or args.plot:
            b = choose_block(arr.shape[0], ny, nx, 2 ** 22)
            ds = block_reduce_any(arr, b)
            iz, nf = z_entries(hkl.header, hkl.records, arr.shape[0], dpe)
            frames_keep = ~ds[iz]
        if args.masks_out:
            save_masks(args.masks_out, frames_keep, block=b)
        report = {
            "source": args.masks_in,
            "deg_per_entry": dpe,
            "invert": invert,
            "pixel_shape": [ny, nx],
            "starting_frame": sf,
            "masked_reflections": int(masked.sum()),
        }
        if masked.any():
            report["masked_xy_px"] = [
                float(hkl.records["xcal"][masked].min()), float(hkl.records["xcal"][masked].max()),
                float(hkl.records["ycal"][masked].min()), float(hkl.records["ycal"][masked].max()),
            ]
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2))
        if args.plot:
            rep = dict(report)
            rep["block"] = b
            rep["masked_voxels_per_frame"] = (~frames_keep).sum(axis=(1, 2)).tolist()
            build_figure(frames_keep, report=rep).write_html(args.plot)
        print(f"masked {int(masked.sum())} of {masked.size} reflections from {args.masks_in}")
        return 0
    if args.command == "refine":
        import numpy as np
        from .refine import RefineParams, collect_images, frame_number_from_name, image_omega, load_image, refine_figure, refine_stack
        if args.integrate_hkl:
            hkl = IntegrateHKL.read(args.integrate_hkl)
            header = hkl.header
            from .external import load_external_masks
            arr, dpe = load_external_masks(args.mask, header, key=args.key,
                                           flip_x=args.flip_x, flip_y=args.flip_y,
                                           deg_per_entry=args.deg_per_entry)
            sa = args.starting_angle if args.starting_angle is not None else float(header.get("starting_angle", 0.0))
            osc = args.oscillation_range if args.oscillation_range is not None else float(header.get("oscillation_range", 0.0))
            sf = args.starting_frame if args.starting_frame is not None else float(header.get("starting_frame", 1.0))
        else:
            with np.load(args.mask) as data:
                if args.key not in data:
                    raise KeyError(f"{args.mask}: no array named {args.key!r} (available: {sorted(data.files)})")
                arr = np.asarray(data[args.key])
            dpe = args.deg_per_entry if args.deg_per_entry else 360.0 / arr.shape[0]
            sa = args.starting_angle
            osc = args.oscillation_range
            sf = args.starting_frame if args.starting_frame is not None else 1.0
            if sa is None or osc is None:
                raise SystemExit("refine without INTEGRATE.HKL requires --starting-angle and --oscillation-range")
        if not osc:
            raise SystemExit("OSCILLATION_RANGE is zero/missing; cannot map images to mask entries")
        paths = collect_images(args.images)
        frames = [frame_number_from_name(p, args.frame_regex) for p in paths]
        omegas = np.array([image_omega(f, sa, osc, sf) + args.z_offset_deg for f in frames])
        params = RefineParams(rect_width=args.rect_width, rect_length=args.rect_length, step=args.step,
                              edge_frac=args.edge_frac, max_shift=args.max_shift, smooth=args.smooth,
                              min_amp=args.min_amp, bad_value=args.bad_value, min_valid_frac=args.min_valid_frac)
        refined, per_image, report = refine_stack(paths, arr, dpe, omegas, params)
        report.update(dict(source=args.mask, images=str(args.images), starting_angle=float(sa),
                           oscillation_range=float(osc), starting_frame=float(sf),
                           z_offset_deg=args.z_offset_deg, flip_x=args.flip_x, flip_y=args.flip_y,
                           params={k: getattr(params, k) for k in vars(params)}))
        if args.masks_out:
            with Path(args.masks_out).open("wb") as fh:
                np.savez(fh, mask=refined, deg_per_entry=np.float64(dpe), invert=np.bool_(True))
        if args.per_image_out:
            with Path(args.per_image_out).open("wb") as fh:
                np.savez(fh, mask=np.stack([p["result"].mask for p in per_image]),
                         entry=np.array([p["entry"] for p in per_image], dtype=np.int64),
                         frame=np.array([p["frame"] for p in per_image], dtype=np.int64),
                         path=np.array([p["path"] for p in per_image]))
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2))
        if args.plot and per_image:
            first = per_image[0]
            refine_figure(load_image(first["path"]), first["result"]).write_html(args.plot)
        print(f"refined {report['n_entries_refined']} of {arr.shape[0]} mask entries from {len(paths)} images"
              + (f" -> {args.masks_out}" if args.masks_out else ""))
        return 0
    if args.command == "refine-debug":
        import numpy as np
        from .refine import RefineParams, load_image, refine_figure, refine_mask_frame, save_mask_png, sum_images
        mimg = load_image(args.mask_png)
        mask = (mimg > 127.0) if args.mask_polarity == "white" else (mimg <= 127.0)
        img = sum_images(args.images, bad_value=args.bad_value)
        if img.shape != mask.shape:
            raise SystemExit(f"mask {args.mask_png} is {mask.shape} but summed image is {img.shape}")
        params = RefineParams(rect_width=args.rect_width, rect_length=args.rect_length, step=args.step,
                              edge_frac=args.edge_frac, max_shift=args.max_shift, smooth=args.smooth,
                              min_amp=args.min_amp, bad_value=args.bad_value, min_valid_frac=args.min_valid_frac)
        res = refine_mask_frame(img, mask, params)
        save_mask_png(args.out, res.mask if args.mask_polarity == "white" else ~res.mask)
        if args.plot:
            refine_figure(img, res).write_html(args.plot)
        exp = res.expansions[np.isfinite(res.expansions)]
        med = float(np.median(exp)) if exp.size else float("nan")
        print(f"refine-debug: {len(args.images)} image(s) summed, {res.n_valid}/{res.shifts.size} edge samples fitted, "
              f"median shift {med:.2f} px -> {args.out}")
        return 0
    if args.command == "plot":
        from .plot import build_figure
        masks, block = load_masks(args.masks)
        report = json.loads(Path(args.report).read_text()) if args.report else {"block": block}
        fig = build_figure(masks, report=report, max_voxels=args.max_voxels)
        fig.write_html(args.out)
        print(f"wrote {args.out} ({int((~masks).sum())} masked voxels)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
