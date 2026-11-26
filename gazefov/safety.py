"""Eye-safety budget for the 850 nm LED ring against IEC 62471 (photobiological safety of lamps)
exempt-group limits. This is a design calculation with stated assumptions — NOT a certification;
a real headset must be measured (radiometer at the eye position) and assessed by a lab.

Limits used (IEC 62471:2006, exempt group, for an IR-A source with no visible stimulus):

* Infrared hazard to the cornea/lens, 780-3000 nm:  E_IR <= 18000 * t^-0.75 W/m^2 for t <= 1000 s,
  and E_IR <= 100 W/m^2 for t > 1000 s (an all-day wearable is the t > 1000 s case).
* Retinal thermal hazard, weak visual stimulus, 780-1400 nm:
  L_IR = sum L(lambda) R(lambda) <= 6000 / alpha  W m^-2 sr^-1, alpha = apparent source angle in
  rad, clamped to >= 0.011 rad; burn-hazard weighting R(lambda) = 10^((700 - lambda)/500) for
  700-1050 nm (R(850 nm) = 0.50).

Model: each LED is a point-like emitter with on-axis radiant intensity I (W/sr) aimed at the eye;
irradiance at distance d is I/d^2 (inverse-square), summed over LEDs, times the duty cycle (the
tracker strobes LEDs with the camera exposure). Radiance of one LED = I / (projected emitting area).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CORNEA_LIMIT_LONG = 100.0  # W/m^2, t > 1000 s


def cornea_limit(t_s: float) -> float:
    return CORNEA_LIMIT_LONG if t_s > 1000 else 18000.0 * t_s ** -0.75


def burn_weight(wavelength_nm: float) -> float:
    if 700 <= wavelength_nm <= 1050:
        return 10 ** ((700 - wavelength_nm) / 500)
    if 1050 < wavelength_nm <= 1150:
        return 0.2
    if 1150 < wavelength_nm <= 1200:
        return 0.2 * 10 ** (0.02 * (1150 - wavelength_nm))
    if 1200 < wavelength_nm <= 1400:
        return 0.02
    return 1.0


@dataclass(frozen=True)
class LedSpec:
    radiant_intensity_mw_sr: float = 4.0  # on-axis, at the drive current used
    wavelength_nm: float = 850.0
    half_angle_deg: float = 30.0


def irradiance_at_eye(leds: list[LedSpec], distance_mm: float, duty: float = 1.0) -> float:
    """Total time-averaged irradiance (W/m^2) at the cornea, all LEDs aimed at the eye."""
    d = distance_mm / 1000.0
    return float(sum(led.radiant_intensity_mw_sr * 1e-3 for led in leds) / d ** 2 * duty)


def check_exposure(leds: list[LedSpec], distance_mm: float, duty: float, source_size_mm: float, t_s: float = 8 * 3600) -> dict:
    E = irradiance_at_eye(leds, distance_mm, duty)
    E_lim = cornea_limit(t_s)
    alpha = max(source_size_mm / distance_mm, 0.011)
    area = np.pi * (source_size_mm / 2000.0) ** 2
    worst = max(leds, key=lambda x: x.radiant_intensity_mw_sr)
    L = burn_weight(worst.wavelength_nm) * worst.radiant_intensity_mw_sr * 1e-3 / area * duty
    L_lim = 6000.0 / alpha
    return {
        "cornea_irradiance_W_m2": E, "cornea_limit_W_m2": E_lim, "cornea_margin": E_lim / max(E, 1e-30),
        "retina_radiance_W_m2_sr": float(L), "retina_limit_W_m2_sr": L_lim, "retina_margin": L_lim / max(L, 1e-30),
        "alpha_rad": alpha, "exempt_cornea": E <= E_lim, "exempt_retina": L <= L_lim,
    }


if __name__ == "__main__":  # python -m gazefov.safety
    for I, duty in [(4.0, 1.0), (4.0, 0.25), (10.0, 0.25)]:
        r = check_exposure([LedSpec(I)] * 8, distance_mm=30.0, duty=duty, source_size_mm=1.0)
        print(f"8 x {I} mW/sr @ 30 mm, duty {duty:.2f}: E = {r['cornea_irradiance_W_m2']:.1f} W/m^2 "
              f"(limit {r['cornea_limit_W_m2']:.0f}, margin x{r['cornea_margin']:.1f}); "
              f"L_R = {r['retina_radiance_W_m2_sr']:.0f} W/m^2/sr (limit {r['retina_limit_W_m2_sr']:.0f}, margin x{r['retina_margin']:.0f})")
