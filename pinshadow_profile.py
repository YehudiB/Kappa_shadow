#!/usr/bin/env python3
"""
pinshadow_profile.py -- pin-cone shadow profile along omega for the mini-kappa goniometer.

For a given kappa and detector distance, computes how much of the detector the
sample-mounting pin shadows at every omega of a full 360 deg sweep, and returns
a compact summary:

  1. a boolean: is there any shadow at all;
  2. the single shadow-free stretch of omega as ([start_deg, end_deg], width_deg),
     wrapping around 360 deg (e.g. ([117.0, 44.0], 287.0) means omega 117 - 360
     followed by 0 - 44 is shadow-free, a 287 deg wedge);
  3. a 90-character ASCII profile, one character per 4 deg of omega, with the
     relative amount of shadow on the ramp ".:-=+*#%@" (none -> most):
     Omega 0: ...characters... 360

The geometry is a line-by-line port of the shadow map of
mini-kappa-goniometer-pin-shadow-visualizer.html (computeColumn with the phi arm
switched off), so the numbers agree with the browser tool with the phi-arm box
unchecked, omega axis 0..360 (degree bin i covers omega [i, i+1), evaluated at
its centre). The pin is a cone of half-angle 19.84 deg with its apex at the
sample around the phi-head direction (phiDirKappa x pinSign, mounting rotation
applied), exactly as in the visualizer. Only the pin cone is modelled; the phi
arm is NOT included.

The detector is sampled with an n_rays x n_rays grid (24 x 24 by default), so
the shadowed fraction carries the same ~0.2 % quantisation as the shadow map.
Pure standard library, no third-party imports needed.

Usage
-----
Command line:

    $ python3 pinshadow_profile.py 200.5 154
    kappa 200.5 deg, detector 154 mm: pin shadow in 137 of 360 omega steps (worst 12.15 % at omega 261.5 deg)
    shadow-free zone: ([350, 213] deg, 223 deg wide)
    Omega 0: ......................................................:--=+*##%@@@@@@@@@@@@@@%%##*+=-:.... 360

Python:

    from pinshadow_profile import shadow_profile
    p = shadow_profile(200.5, 154.0)
    p.any_shadow      # True
    p.free            # ([350.0, 213.0], 223.0): omega 350->360 + 0->213 is shadow-free
    p.sparkline       # "Omega 0: ... 360" (see above)
    print(p)          # the three-line report shown above

Conventions
-----------
* free zone: ([0.0, 360.0], 360.0) when nothing is ever shadowed; None when
  something is shadowed at EVERY omega; otherwise the widest shadow-free
  stretch (ties resolved deterministically, first one after a shadowed bin).
* If more than one disjoint shadow-free stretch exists (two shadowed arcs per
  revolution), the widest one is returned.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

DEG = math.pi / 180.0

# Defaults mirror the visualizer's built-in state (mini-kappa at alpha = 24 deg,
# mounting rotation 90 deg, Eiger2X 16M at 154 mm).
ALPHA_DEG = 24.0          # kappa/phi tilt from the omega axis
MOUNT_DEG = 90.0          # mounting rotation, constant offset added to omega
CONE_DEG = 19.84          # pin shadow cone half-angle (fixed, as in the visualizer)
KAPPA_SIGN = -1.0         # left-handed kappa motor
PIN_SIGN = -1.0           # pin points opposite the phi-head direction
KAPPA_MIN_DEG = -4.0       # kappa travel limits
KAPPA_MAX_DEG = 250.0
DETECTOR_HALF_MM = (155.55, 163.575)   # half-width (y) and half-height (z), Eiger2X 16M
BEAM_MM = (0.0, 0.0)                   # beam position, visualizer beamX / beamY inputs
N_RAYS_DEFAULT = 24                    # ray grid per axis (24 x 24 = 576 rays)

RAMP = ".:-=+*#%@"        # 9 levels: no shadow -> most shadow
SPARK_BINS = 90          # 90 characters, 4 deg each
N_OMEGA = 360            # one bin per degree


# --------------------------------------------------------------------------
# geometry kernel (port of computeColumn with S.armOn = false)
# --------------------------------------------------------------------------
def _pin_frame(kappa_deg: float, alpha_deg: float = ALPHA_DEG,
               kappa_sign: float = KAPPA_SIGN, pin_sign: float = PIN_SIGN
               ) -> Tuple[float, float, float]:
    """Pin-cone axis before the omega rotation, in the omega frame.

    Port of phiDirKappa(kappa) * pinSign: the phi-axis direction at kappa,
    scaled by the pin sign so the cone opens around the actual pin.
    """
    a = alpha_deg * DEG
    sa, ca = math.sin(a), math.cos(a)
    k = kappa_deg * DEG
    ck, sk = math.cos(k), math.sin(k)
    ph = (kappa_sign * sa * ca * (1.0 - ck), kappa_sign * sa * sk, ca * ca + sa * sa * ck)
    return (ph[0] * pin_sign, ph[1] * pin_sign, ph[2] * pin_sign)


def shadow_fractions(kappa_deg: float, det_dist_mm: float,
                     alpha_deg: float = ALPHA_DEG,
                     mount_deg: float = MOUNT_DEG,
                     cone_deg: float = CONE_DEG,
                     detector_half_mm: Tuple[float, float] = DETECTOR_HALF_MM,
                     beam_mm: Tuple[float, float] = BEAM_MM,
                     n_rays: int = N_RAYS_DEFAULT) -> list:
    """Fraction of the detector shadowed by the pin cone, for omega 0..360.

    Returns n_omega = 360 fractions: bin i covers omega [i, i+1) deg on the
    0..360 axis and is evaluated at its centre, omega i+0.5 deg (+ mounting
    rotation), exactly like row i of the visualizer shadow map. Ray sampling
    with an n_rays x n_rays grid; a ray exactly on the cone counts as blocked.
    """
    if not (KAPPA_MIN_DEG <= kappa_deg <= KAPPA_MAX_DEG):
        raise ValueError("kappa_deg = %g outside the motor travel [%g, %g] deg"
                         % (kappa_deg, KAPPA_MIN_DEG, KAPPA_MAX_DEG))
    if det_dist_mm <= 0:
        raise ValueError("det_dist_mm must be positive")
    if n_rays < 1:
        raise ValueError("n_rays must be >= 1")
    wy, hz = float(detector_half_mm[0]), float(detector_half_mm[1])
    if wy <= 0 or hz <= 0:
        raise ValueError("detector half-sizes must be positive")
    x = float(det_dist_mm)
    cy0, cz0 = -float(beam_mm[0]), -float(beam_mm[1])
    cos_delta = math.cos(cone_deg * DEG)

    # detector ray table (port of startMap): the pin test only needs dY, dZ
    # and dT = cos(cone) * |(x, y, z)|, since the JS tests the scaled
    # inequality (x*frx + y*fry)*c + (y*frx - x*fry)*s + z*frz >= dT.
    rays_y, rays_z, rays_dt = [], [], []
    step_y = 2.0 * wy / n_rays
    step_z = 2.0 * hz / n_rays
    for iy in range(n_rays):
        y = cy0 - wy + (iy + 0.5) * step_y
        for iz in range(n_rays):
            z = cz0 - hz + (iz + 0.5) * step_z
            length = math.sqrt(x * x + y * y + z * z)
            rays_y.append(y)
            rays_z.append(z)
            rays_dt.append(cos_delta * length)

    frx, fry, frz = _pin_frame(kappa_deg, alpha_deg)
    a_ray = [x * frx + y * fry for y in rays_y]
    b_ray = [y * frx - x * fry for y in rays_y]
    c_ray = [z * frz for z in rays_z]
    n_rays_tot = n_rays * n_rays

    fracs = []
    for i in range(N_OMEGA):
        omega = (i + 0.5 + mount_deg) * DEG
        c, s = math.cos(omega), math.sin(omega)
        cnt = 0
        for a_, b_, c_, dt_ in zip(a_ray, b_ray, c_ray, rays_dt):
            if a_ * c + b_ * s + c_ >= dt_:
                cnt += 1
        fracs.append(cnt / n_rays_tot)
    return fracs


# --------------------------------------------------------------------------
# summary outputs
# --------------------------------------------------------------------------
def free_zone(fracs: list) -> Optional[Tuple[list, float]]:
    """Widest shadow-free omega stretch as ([start_deg, end_deg], width_deg).

    Bin i is free when fracs[i] == 0. Angles are bin boundaries on the 0..360
    axis and may wrap: ([117.0, 44.0], 287.0) means bins 117..359 plus 0..43.
    Returns ([0.0, 360.0], 360.0) if nothing is shadowed, None if something
    is shadowed at every omega.
    """
    free = [f <= 0.0 for f in fracs]
    try:
        k = free.index(False)                # any shadowed bin
    except ValueError:
        return ([0.0, 360.0], 360.0)        # no shadow anywhere
    # rotate the circle so bin 0 is shadowed, then the longest run of free
    # bins in the rotated array is the widest circular free stretch
    rot = free[k + 1:] + free[:k + 1]
    best_len = 0
    best_start = 0
    run_len = 0
    run_start = 0
    for idx, is_free in enumerate(rot):
        if is_free:
            if run_len == 0:
                run_start = idx
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0
    if best_len == 0:
        return None                         # shadow at every omega
    start = (best_start + k + 1) % N_OMEGA
    end = (start + best_len) % N_OMEGA
    return ([float(start), float(end)], float(best_len))


def sparkline(fracs: list) -> str:
    """One-line ASCII profile: 90 characters, one per 4 deg of omega.

    Each character averages the 4 one-degree bins it covers and maps the value
    RELATIVE to the worst bin of this sweep onto the ramp ".:-=+*#%@"
    (no shadow -> '.', the sweep's maximum -> '@').
    """
    vmax = max(fracs)
    chars = []
    for b in range(SPARK_BINS):
        chunk = fracs[4 * b:4 * b + 4]
        v = sum(chunk) / 4.0
        level = 0 if vmax <= 0.0 else min(8, int(round(8.0 * v / vmax)))
        chars.append(RAMP[level])
    return "Omega 0: " + "".join(chars) + " 360"


@dataclass(frozen=True)
class ShadowProfile:
    """Summary of the pin shadow along a full omega sweep (see module docstring)."""
    any_shadow: bool                       # 1) is anything shadowed
    free: Optional[Tuple[list, float]]     # 2) ([start_deg, end_deg], width_deg) or None
    sparkline: str                         # 3) "Omega 0: ....... 360"
    omega_deg: tuple                        # bin centres 0.5 .. 359.5
    shadow_fraction: tuple                  # fraction of detector shadowed per bin
    kappa_deg: float
    det_dist_mm: float

    def __str__(self) -> str:
        n_shadow = sum(1 for f in self.shadow_fraction if f > 0.0)
        if self.any_shadow:
            fmax = max(self.shadow_fraction)
            om_at = self.omega_deg[self.shadow_fraction.index(fmax)]
            head = ("kappa %g deg, detector %g mm: pin shadow in %d of %d omega steps "
                    "(worst %.2f %% at omega %.1f deg)"
                    % (self.kappa_deg, self.det_dist_mm, n_shadow, len(self.shadow_fraction),
                       100.0 * fmax, om_at))
        else:
            head = ("kappa %g deg, detector %g mm: no pin shadow at any omega"
                    % (self.kappa_deg, self.det_dist_mm))
        if self.free is None:
            zone = "shadow-free zone: none (shadow at every omega)"
        else:
            zone = ("shadow-free zone: ([%s, %s] deg, %g deg wide)"
                    % (("%g" % self.free[0][0]), ("%g" % self.free[0][1]), self.free[1]))
        return "\n".join([head, zone, self.sparkline])


def shadow_profile(kappa_deg: float, det_dist_mm: float,
                   alpha_deg: float = ALPHA_DEG,
                   mount_deg: float = MOUNT_DEG,
                   cone_deg: float = CONE_DEG,
                   detector_half_mm: Tuple[float, float] = DETECTOR_HALF_MM,
                   beam_mm: Tuple[float, float] = BEAM_MM,
                   n_rays: int = N_RAYS_DEFAULT) -> ShadowProfile:
    """Pin shadow along omega: any-shadow flag, free zone, ASCII profile.

    Convenience wrapper around shadow_fractions(); see the module docstring
    for the meaning of the three summary outputs and the conventions.
    """
    fracs = shadow_fractions(kappa_deg, det_dist_mm, alpha_deg=alpha_deg,
                             mount_deg=mount_deg, cone_deg=cone_deg,
                             detector_half_mm=detector_half_mm, beam_mm=beam_mm,
                             n_rays=n_rays)
    return ShadowProfile(any_shadow=any(f > 0.0 for f in fracs),
                         free=free_zone(fracs),
                         sparkline=sparkline(fracs),
                         omega_deg=tuple(i + 0.5 for i in range(N_OMEGA)),
                         shadow_fraction=tuple(fracs),
                         kappa_deg=float(kappa_deg),
                         det_dist_mm=float(det_dist_mm))


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def _main(argv=None):
    p = argparse.ArgumentParser(
        description="Pin-cone shadow profile along omega for the mini-kappa goniometer "
                    "(pin only, phi arm excluded), matching the visualizer shadow map.")
    p.add_argument("kappa_deg", type=float, help="kappa motor position (deg)")
    p.add_argument("det_dist_mm", type=float, help="detector distance (mm)")
    p.add_argument("--alpha-deg", type=float, default=ALPHA_DEG,
                   help="kappa/phi tilt from the omega axis (default %(default)s deg)")
    p.add_argument("--mount-deg", type=float, default=MOUNT_DEG,
                   help="mounting rotation added to omega (default %(default)s deg)")
    p.add_argument("--cone-deg", type=float, default=CONE_DEG,
                   help="pin shadow cone half-angle (default %(default)s deg)")
    p.add_argument("--detector-mm", type=float, nargs=2, metavar=("HALF_Y", "HALF_Z"),
                   default=list(DETECTOR_HALF_MM),
                   help="detector half-width y / half-height z in mm "
                        "(default Eiger2X 16M: %(default)s)")
    p.add_argument("--beam-mm", type=float, nargs=2, metavar=("Y", "Z"),
                   default=list(BEAM_MM),
                   help="beam position offsets, visualizer beamX/beamY (default %(default)s)")
    p.add_argument("--n-rays", type=int, default=N_RAYS_DEFAULT,
                   help="ray grid per axis (default %(default)s)")
    args = p.parse_args(argv)
    prof = shadow_profile(args.kappa_deg, args.det_dist_mm, alpha_deg=args.alpha_deg,
                          mount_deg=args.mount_deg, cone_deg=args.cone_deg,
                          detector_half_mm=tuple(args.detector_mm),
                          beam_mm=tuple(args.beam_mm), n_rays=args.n_rays)
    print(str(prof))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
