"""
SolSpecs — Heat Stress Algorithm
WBGT estimation and multi-signal heat stress tier scoring.

All inputs use SI units unless noted.
Tier output: "green" | "yellow" | "orange" | "red"
"""

import math
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_wbgt_estimate(
    temp_c: float,
    humidity_pct: float,
    in_direct_sun: bool,
) -> float:
    """
    Estimate Wet Bulb Globe Temperature from ambient temperature and humidity.

    Full WBGT requires a wet bulb thermometer and black globe thermometer.
    This uses the Stull (2011) wet bulb approximation and a solar offset for
    the globe temperature.

    Liljegren simplified form:
        WBGT ≈ 0.7·Tw + 0.2·Tg + 0.1·Ta

    Where:
        Tw  = wet bulb temperature (Stull 2011 formula)
        Tg  = globe temperature ≈ Ta + solar_offset
        Ta  = ambient air temperature (°C)
        solar_offset = 7°C in direct sun, 1°C in shade

    Args:
        temp_c:       Ambient air temperature in °C (DHT11 reading).
        humidity_pct: Relative humidity 0–100 (DHT11 reading).
        in_direct_sun: True when photoresistor confirms direct solar exposure.

    Returns:
        WBGT estimate in °C.

    Reference:
        Stull, R. (2011). Wet-bulb temperature from relative humidity and
        air temperature. J. Appl. Meteor. Climatol., 50, 2267–2269.
    """
    h = humidity_pct

    # Stull (2011) — valid for 5–99 % RH and -20 to 50 °C
    tw = (
        temp_c * math.atan(0.151977 * (h + 8.313659) ** 0.5)
        + math.atan(temp_c + h)
        - math.atan(h - 1.676331)
        + 0.00391838 * h ** 1.5 * math.atan(0.023101 * h)
        - 4.686035
    )

    solar_offset = 7.0 if in_direct_sun else 1.0
    tg = temp_c + solar_offset

    wbgt = 0.7 * tw + 0.2 * tg + 0.1 * temp_c
    return wbgt


def compute_heat_stress_tier(
    wbgt: float,
    heart_rate: float,
    spo2: float,
    skin_temp: float,
    sun_exposure_minutes: float,
    breathing_rate: Optional[float] = None,
    sweat_level: Optional[float] = None,
    exertion_level: Optional[float] = None,
) -> str:
    """
    Fuse environmental and physiological signals into a heat stress tier.

    Each signal contributes risk points to a cumulative score.  The tier is
    determined by which band that score falls into.

    OSHA WBGT thresholds (moderate workload):
        < 24°C  → green
        24–26°C → low-yellow
        26–28°C → moderate-yellow/orange
        28–30°C → orange
        > 30°C  → red

    Args:
        wbgt:                  WBGT estimate in °C (from compute_wbgt_estimate).
        heart_rate:            Beats per minute.
        spo2:                  Blood oxygen saturation 0–100 %.
        skin_temp:             Skin surface temperature in °C.
        sun_exposure_minutes:  Cumulative direct-sun minutes in the current hour.
        breathing_rate:        Breaths per minute (optional; None = not available).
        sweat_level:           Raw ADC sweat sensor reading (optional; reserved
                               for future calibration).
        exertion_level:        Normalised exertion 0.0–1.0 derived from IMU
                               acceleration variance (optional).

    Returns:
        "green", "yellow", "orange", or "red".
    """
    score = 0

    # ── Environmental: WBGT ──────────────────────────────────────────
    if wbgt > config.WBGT_CRITICAL:      # > 30 °C
        score += 4
    elif wbgt > config.WBGT_HIGH:        # > 28 °C
        score += 3
    elif wbgt > config.WBGT_MODERATE:    # > 26 °C
        score += 2
    elif wbgt > config.WBGT_LOW:         # > 24 °C
        score += 1

    # ── Environmental: sun exposure ──────────────────────────────────
    if sun_exposure_minutes > config.SUN_EXPOSURE_CRITICAL_MIN:    # > 45 min
        score += 3
    elif sun_exposure_minutes > config.SUN_EXPOSURE_CONCERN_MIN:   # > 30 min
        score += 2
    elif sun_exposure_minutes > config.SUN_EXPOSURE_MILD_MIN:      # > 15 min
        score += 1

    # ── Physiological: heart rate ────────────────────────────────────
    if heart_rate > config.HR_CRITICAL:     # > 140 bpm
        score += 4
    elif heart_rate > config.HR_ELEVATED:   # > 120 bpm
        score += 3
    elif heart_rate > config.HR_CONCERN:    # > 100 bpm
        score += 2
    elif heart_rate > config.HR_MILD:       # > 90 bpm
        score += 1

    # ── Physiological: SpO2 (inverted — lower is worse) ─────────────
    # spo2 == 0 means sensor not ready — skip to avoid false critical alarm
    if spo2 > 0:
        if spo2 < config.SPO2_CRITICAL:      # < 90 %
            score += 4
        elif spo2 < config.SPO2_CONCERN:     # < 93 %
            score += 2
        elif spo2 < config.SPO2_MILD:        # < 95 %
            score += 1

    # ── Physiological: skin temperature ─────────────────────────────
    if skin_temp > config.SKIN_TEMP_CRITICAL:   # > 38.5 °C
        score += 4
    elif skin_temp > config.SKIN_TEMP_CONCERN:  # > 37.5 °C
        score += 2
    elif skin_temp > config.SKIN_TEMP_MILD:     # > 37.0 °C
        score += 1

    # ── Physiological: breathing rate (if available) ─────────────────
    if breathing_rate is not None:
        if breathing_rate > config.BREATHING_RATE_HIGH:   # > 25 breaths/min
            score += 2
        elif breathing_rate > 20:
            score += 1

    # ── Physical: exertion level (if available) ──────────────────────
    # Derived from IMU acceleration variance; 0.0 = rest, 1.0 = max exertion.
    if exertion_level is not None:
        if exertion_level > 0.8:
            score += 2
        elif exertion_level > 0.6:
            score += 1

    # sweat_level is accepted but not scored until sensor is calibrated.

    # ── Map cumulative score to tier ─────────────────────────────────
    if score >= config.TIER_RED:      # >= 10
        return "red"
    elif score >= config.TIER_ORANGE: # >= 6
        return "orange"
    elif score >= config.TIER_YELLOW: # >= 3
        return "yellow"
    else:
        return "green"


def thermistor_raw_to_celsius(
    raw_adc: int,
    adc_max: int = 1023,
    r_divider: float = 10_000.0,
    r_nominal: float = 2430.0,
    t_nominal: float = 25.0,
    b_coefficient: float = 3950.0,
) -> float:
    """
    Convert Arduino UNO Q 10-bit ADC reading to skin temperature in °C.

    Wiring (inverted divider — lower ADC = hotter):
        3.3 V → 10 kΩ fixed resistor → A1 (ADC) → NTC thermistor → GND

    Voltage divider:
        V_A1 = 3.3 * R_therm / (R_fixed + R_therm)
        R_therm = R_fixed * raw / (adc_max - raw)

    Lower raw values → lower V_A1 → lower R_therm → higher temperature (NTC).

    Args:
        raw_adc:       10-bit ADC reading (0–1023) from Arduino pin A1.
        adc_max:       ADC full-scale (1023 for 10-bit Arduino).
        r_divider:     Fixed resistor in Ω (upper leg, between VCC and ADC).
        r_nominal:     Thermistor nominal resistance at t_nominal °C (Ω).
        t_nominal:     Thermistor nominal temperature in °C (default 25 °C).
        b_coefficient: Steinhart-Hart B coefficient (K).

    Returns:
        Temperature in °C, or nan if raw_adc is out of range.
    """
    if raw_adc <= 0 or raw_adc >= adc_max:
        return float("nan")
    # Inverted divider: thermistor in lower leg (between ADC pin and GND)
    r_thermistor = r_divider * raw_adc / (adc_max - raw_adc)
    t0_k = t_nominal + 273.15
    temp_k = 1.0 / (
        1.0 / t0_k + math.log(r_thermistor / r_nominal) / b_coefficient
    )
    return temp_k - 273.15
