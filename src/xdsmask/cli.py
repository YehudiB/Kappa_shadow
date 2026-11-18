from __future__ import annotations

import argparse
import json
from pathlib import Path

from .apply import apply_masks, load_masks, save_masks
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
    d.add_argument("--entry-frac", type=float, default=0.8, help="entry edge: centroid y > entry_frac * NY")
    d.add_argument("--beamstop-radius", type=float, default=50.0)
    d.add_argument("--radius", type=int, default=3, help="dilation radius of the mask (pixels)")
    d.add_argument("--band", type=float, default=15.0, help="band around fitted trajectory (pixels)")
    d.add_argument("--block", type=int, default=0, help="pixel block size for the voxel grid; 0 = auto from --voxel-budget")
    d.add_argument("--voxel-budget", type=int, default=2 ** 25, help="max voxels in the (frames, y, x) grid when --block=0")
    a = sub.add_parser("apply", help="apply masks: emit FILTER.HKL and/or a filtered INTEGRATE.HKL")
    a.add_argument("integrate_hkl")
    a.add_argument("--masks", required=True, help=".npz written by 'detect'")
    a.add_argument("--filter-out", default=None)
    a.add_argument("--out", default=None)
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
            beamstop_radius=args.beamstop_radius,
            radius=args.radius,
            band=args.band,
            block=args.block,
            voxel_budget=args.voxel_budget,
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
        masks, block = load_masks(args.masks)
        masked = apply_masks(hkl, masks, block=block, filter_out=args.filter_out, out=args.out)
        print(f"masked {int(masked.sum())} of {masked.size} reflections (block={block})")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
