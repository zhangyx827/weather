"""Physical indicators for Saudi multi-hazard screening."""

from __future__ import annotations

import math
from typing import Any

from mazu_saudi.utils.math import clamp, is_missing, map_values


def _nan() -> float:
    return float("nan")


def compute_vpd_kpa(temp_c: Any, rh_percent: Any) -> Any:
    """Compute vapor pressure deficit in kPa.

    Args:
        temp_c: Air temperature in degrees Celsius.
        rh_percent: Relative humidity in percent, expected 0-100.
    """

    def scalar(t, rh):
        if is_missing(t) or is_missing(rh):
            return _nan()
        rh_clamped = max(0.0, min(100.0, float(rh)))
        es = 0.6108 * math.exp((17.27 * float(t)) / (float(t) + 237.3))
        ea = es * rh_clamped / 100.0
        return max(0.0, es - ea)

    return map_values(scalar, temp_c, rh_percent)


def compute_heat_index_c(temp_c: Any, rh_percent: Any) -> Any:
    """Compute apparent heat index in degrees Celsius."""

    def scalar(t, rh):
        if is_missing(t) or is_missing(rh):
            return _nan()
        t_c = float(t)
        rh_f = max(0.0, min(100.0, float(rh)))
        if t_c < 26.7:
            return t_c
        t_f = t_c * 9.0 / 5.0 + 32.0
        hi_f = (
            -42.379
            + 2.04901523 * t_f
            + 10.14333127 * rh_f
            - 0.22475541 * t_f * rh_f
            - 0.00683783 * t_f * t_f
            - 0.05481717 * rh_f * rh_f
            + 0.00122874 * t_f * t_f * rh_f
            + 0.00085282 * t_f * rh_f * rh_f
            - 0.00000199 * t_f * t_f * rh_f * rh_f
        )
        return (hi_f - 32.0) * 5.0 / 9.0

    return map_values(scalar, temp_c, rh_percent)


def compute_pwat_placeholder(temp_c: Any, rh_percent: Any, pressure_hpa: Any | None = None) -> Any:
    """Estimate precipitable water placeholder in mm."""

    def scalar(t, rh, p):
        if is_missing(t) or is_missing(rh):
            return _nan()
        pressure_factor = 1.0 if is_missing(p) else clamp((float(p) - 700.0) / 350.0, 0.5, 1.2)
        return max(0.0, 0.18 * (float(t) + 5.0) * (float(rh) / 100.0) * pressure_factor)

    return map_values(scalar, temp_c, rh_percent, 1010.0 if pressure_hpa is None else pressure_hpa)


def compute_ivt_placeholder(wind_speed_mps: Any, pwat_mm: Any) -> Any:
    """Estimate integrated vapor transport placeholder in kg m-1 s-1."""

    def scalar(wind, pwat):
        if is_missing(wind) or is_missing(pwat):
            return _nan()
        return max(0.0, float(wind) * float(pwat) * 2.5)

    return map_values(scalar, wind_speed_mps, pwat_mm)


def compute_cape_placeholder(temp_c: Any, rh_percent: Any) -> Any:
    """Estimate CAPE placeholder in J/kg from hot and humid instability proxy."""

    def scalar(t, rh):
        if is_missing(t) or is_missing(rh):
            return _nan()
        heat_term = max(0.0, float(t) - 28.0)
        moisture_term = max(0.0, float(rh) - 35.0)
        return heat_term * moisture_term * 6.0

    return map_values(scalar, temp_c, rh_percent)


def compute_flash_flood_screening_score(
    precip_1h_mm: float | None,
    precip_6h_mm: float | None,
    precip_24h_mm: float | None,
    slope_deg: float | None = None,
    soil_moisture_frac: float | None = None,
    impervious_frac: float | None = None,
) -> float:
    """Compute 0-1 flash-flood screening score."""

    p1 = 0.0 if is_missing(precip_1h_mm) else float(precip_1h_mm)
    p6 = 0.0 if is_missing(precip_6h_mm) else float(precip_6h_mm)
    p24 = 0.0 if is_missing(precip_24h_mm) else float(precip_24h_mm)
    slope = 5.0 if is_missing(slope_deg) else float(slope_deg)
    soil = 0.25 if is_missing(soil_moisture_frac) else float(soil_moisture_frac)
    impervious = 0.15 if is_missing(impervious_frac) else float(impervious_frac)
    rain_component = max(p1 / 35.0, p6 / 80.0, p24 / 140.0)
    terrain_component = 0.20 * clamp(slope / 25.0) + 0.20 * clamp(soil) + 0.15 * clamp(impervious)
    return clamp(0.75 * rain_component + terrain_component)


def compute_dust_potential_score(
    wind_speed_mps: float | None,
    wind_gust_mps: float | None = None,
    soil_moisture_frac: float | None = None,
    vegetation_index: float | None = None,
    visibility_km: float | None = None,
) -> float:
    """Compute 0-1 wind-blown dust potential score."""

    wind = 0.0 if is_missing(wind_speed_mps) else float(wind_speed_mps)
    gust = wind if is_missing(wind_gust_mps) else float(wind_gust_mps)
    soil = 0.15 if is_missing(soil_moisture_frac) else float(soil_moisture_frac)
    veg = 0.2 if is_missing(vegetation_index) else float(vegetation_index)
    vis = 20.0 if is_missing(visibility_km) else float(visibility_km)
    wind_component = max(wind / 16.0, gust / 22.0)
    erodibility = 0.35 * (1.0 - clamp(soil / 0.35)) + 0.25 * (1.0 - clamp(veg / 0.45))
    observed_dust = 0.20 * (1.0 - clamp(vis / 10.0))
    return clamp(0.65 * wind_component + erodibility + observed_dust)


def compute_dry_heat_stress_score(
    temp_c: float | None,
    rh_percent: float | None,
    wind_speed_mps: float | None = None,
    vegetation_index: float | None = None,
) -> float:
    """Compute 0-1 dry-heat agricultural stress score."""

    if is_missing(temp_c) or is_missing(rh_percent):
        return 0.0
    vpd = compute_vpd_kpa(temp_c, rh_percent)
    temp_component = clamp((float(temp_c) - 35.0) / 12.0)
    vpd_component = clamp(float(vpd) / 5.0)
    wind_component = 0.15 * clamp((0.0 if is_missing(wind_speed_mps) else float(wind_speed_mps)) / 10.0)
    veg_component = 0.15 * (1.0 - clamp((0.25 if is_missing(vegetation_index) else float(vegetation_index)) / 0.5))
    return clamp(0.45 * temp_component + 0.45 * vpd_component + wind_component + veg_component)
