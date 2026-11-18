from __future__ import annotations

import argparse
import json
from pathlib import Path

from .apply import apply_masks, load_masks, load_masks_meta, save_masks
from .detect import DetectionParams, detect_shadow
from .hkl import IntegrateHKL


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
        if args.masks_out:
            save_masks(args.masks_out, arr, block=1, deg_per_entry=dpe, invert=invert)
        report = {
            "source": args.masks_in,
            "deg_per_entry": dpe,
            "invert": invert,
            "pixel_shape": [ny, nx],
            "starting_frame": sf,
            "masked_reflections": int(masked.sum()),
        }
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2))
        if args.plot:
            b = choose_block(arr.shape[0], ny, nx, 2 ** 22)
            ds = block_reduce_any(arr, b)
            iz, nf = z_entries(hkl.header, hkl.records, arr.shape[0], dpe)
            frames_ds = ds[iz]
            rep = dict(report)
            rep["block"] = b
            rep["masked_voxels_per_frame"] = (~frames_ds).sum(axis=(1, 2)).tolist()
            build_figure(frames_ds, report=rep).write_html(args.plot)
        print(f"masked {int(masked.sum())} of {masked.size} reflections from {args.masks_in}")
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
