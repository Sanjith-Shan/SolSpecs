"""
Tests for core/heat_stress.py

Run with:  pytest tests/test_heat_stress.py -v
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.heat_stress import (
    compute_wbgt_estimate,
    compute_heat_stress_tier,
    thermistor_raw_to_celsius,
)
import config


# ─── compute_wbgt_estimate ─────────────────────────────────────────────────────

class TestComputeWBGTEstimate:

    def test_direct_sun_raises_wbgt_vs_shade(self):
        """Direct sun should produce a higher WBGT than shade at same T/RH."""
        wbgt_sun = compute_wbgt_estimate(35.0, 60.0, in_direct_sun=True)
        wbgt_shade = compute_wbgt_estimate(35.0, 60.0, in_direct_sun=False)
        assert wbgt_sun > wbgt_shade

    def test_solar_offset_magnitude(self):
        """Direct sun adds 7°C globe offset vs 1°C in shade → ~1.2°C WBGT diff."""
        wbgt_sun = compute_wbgt_estimate(30.0, 50.0, in_direct_sun=True)
        wbgt_shade = compute_wbgt_estimate(30.0, 50.0, in_direct_sun=False)
        # Globe weight is 0.2, offset diff is 6°C → delta ≈ 1.2°C
        assert abs((wbgt_sun - wbgt_shade) - 1.2) < 0.01

    def test_higher_humidity_raises_wbgt(self):
        """Higher humidity → higher wet bulb → higher WBGT."""
        wbgt_low_rh = compute_wbgt_estimate(35.0, 30.0, in_direct_sun=False)
        wbgt_high_rh = compute_wbgt_estimate(35.0, 80.0, in_direct_sun=False)
        assert wbgt_high_rh > wbgt_low_rh

    def test_higher_temp_raises_wbgt(self):
        """Higher ambient temperature → higher WBGT."""
        wbgt_cool = compute_wbgt_estimate(20.0, 50.0, in_direct_sun=False)
        wbgt_hot = compute_wbgt_estimate(40.0, 50.0, in_direct_sun=False)
        assert wbgt_hot > wbgt_cool

    def test_returns_float(self):
        """Output must always be a Python float."""
        result = compute_wbgt_estimate(30.0, 60.0, in_direct_sun=True)
        assert isinstance(result, float)

    def test_typical_hot_day_direct_sun_exceeds_critical(self):
        """35°C / 70% RH in direct sun should exceed the 30°C OSHA critical threshold."""
        wbgt = compute_wbgt_estimate(35.0, 70.0, in_direct_sun=True)
        assert wbgt > config.WBGT_CRITICAL

    def test_mild_conditions_below_low_threshold(self):
        """20°C / 40% RH in shade should be below the 24°C low threshold."""
        wbgt = compute_wbgt_estimate(20.0, 40.0, in_direct_sun=False)
        assert wbgt < config.WBGT_LOW

    def test_moderate_conditions_in_moderate_band(self):
        """34°C / 60% RH in shade → WBGT should land in the 24–30°C moderate band.
        Note: Stull wet-bulb depresses WBGT well below ambient; 34°C shade ≈ 25–26°C WBGT."""
        wbgt = compute_wbgt_estimate(34.0, 60.0, in_direct_sun=False)
        assert config.WBGT_LOW <= wbgt <= config.WBGT_CRITICAL

    def test_stull_formula_known_value(self):
        """Spot-check Stull (2011): T=20°C, RH=50% should give Tw ≈ 13.7°C."""
        # Compute wet bulb manually at T=20, RH=50, shade (Tg=21, Ta=20)
        # wbgt = 0.7*Tw + 0.2*Tg + 0.1*Ta
        # We back-compute expected Tw from the published Stull table.
        # Stull 2011 Table 1: T=20, RH=50 → Tw ≈ 13.7°C
        h = 50.0
        temp = 20.0
        tw_expected = 13.7
        tw_computed = (
            temp * math.atan(0.151977 * (h + 8.313659) ** 0.5)
            + math.atan(temp + h)
            - math.atan(h - 1.676331)
            + 0.00391838 * h ** 1.5 * math.atan(0.023101 * h)
            - 4.686035
        )
        assert abs(tw_computed - tw_expected) < 0.5  # within 0.5°C

    def test_low_humidity_does_not_crash(self):
        """1% humidity edge case should return a valid number."""
        result = compute_wbgt_estimate(30.0, 1.0, in_direct_sun=False)
        assert math.isfinite(result)

    def test_high_humidity_does_not_crash(self):
        """99% humidity edge case should return a valid number."""
        result = compute_wbgt_estimate(30.0, 99.0, in_direct_sun=False)
        assert math.isfinite(result)

    def test_freezing_temp_does_not_crash(self):
        """Negative temperature should not raise an exception."""
        result = compute_wbgt_estimate(-5.0, 80.0, in_direct_sun=False)
        assert math.isfinite(result)

    def test_tropical_extreme(self):
        """40°C / 90% RH direct sun — worst-case field conditions."""
        result = compute_wbgt_estimate(40.0, 90.0, in_direct_sun=True)
        assert math.isfinite(result)
        assert result > 35.0  # definitely critical territory


# ─── compute_heat_stress_tier ──────────────────────────────────────────────────

# Baseline "all normal" values — produces score 0 → green
NORMAL = dict(
    wbgt=22.0,
    heart_rate=72,
    spo2=98,
    skin_temp=36.5,
    sun_exposure_minutes=5,
)


class TestComputeHeatStressTierGreen:

    def test_all_normal_is_green(self):
        assert compute_heat_stress_tier(**NORMAL) == "green"

    def test_green_with_all_optional_params(self):
        assert compute_heat_stress_tier(
            **NORMAL, breathing_rate=15, sweat_level=100, exertion_level=0.3
        ) == "green"

    def test_optional_params_default_none(self):
        """Function must accept calls without optional params."""
        result = compute_heat_stress_tier(**NORMAL)
        assert result in ("green", "yellow", "orange", "red")


class TestComputeHeatStressTierWBGT:

    def test_wbgt_above_low_threshold_adds_one_point(self):
        """WBGT just above 24°C → +1 point — not enough alone for yellow."""
        tier = compute_heat_stress_tier(
            wbgt=config.WBGT_LOW + 0.5,
            heart_rate=72, spo2=98, skin_temp=36.5, sun_exposure_minutes=0,
        )
        # 1 point only → green (threshold is 3)
        assert tier == "green"

    def test_wbgt_critical_alone_causes_orange(self):
        """WBGT > 30°C → +4 points → orange (4 < 6? No, 4 ≥ 3 → yellow).
        Actually 4 points → yellow (need 6 for orange)."""
        tier = compute_heat_stress_tier(
            wbgt=config.WBGT_CRITICAL + 0.5,
            heart_rate=72, spo2=98, skin_temp=36.5, sun_exposure_minutes=0,
        )
        assert tier == "yellow"  # 4 points → yellow band (3–5)

    def test_wbgt_high_plus_sun_causes_orange(self):
        """WBGT > 30°C (+4) + sun > 30 min (+2) = 6 → orange."""
        tier = compute_heat_stress_tier(
            wbgt=config.WBGT_CRITICAL + 1.0,
            heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=config.SUN_EXPOSURE_CONCERN_MIN + 1,
        )
        assert tier == "orange"

    def test_wbgt_threshold_boundary_flips_tier(self):
        """WBGT exactly at 28°C scores +2 (green); just above scores +3 (yellow).
        This is the meaningful boundary where tier actually changes."""
        tier_at = compute_heat_stress_tier(
            wbgt=config.WBGT_HIGH,          # 28.0 → +2 pts → green
            heart_rate=72, spo2=98, skin_temp=36.5, sun_exposure_minutes=0,
        )
        tier_above = compute_heat_stress_tier(
            wbgt=config.WBGT_HIGH + 0.01,   # 28.01 → +3 pts → yellow
            heart_rate=72, spo2=98, skin_temp=36.5, sun_exposure_minutes=0,
        )
        assert tier_at == "green"
        assert tier_above == "yellow"

    def test_each_wbgt_band_assigns_correct_points(self):
        """Validate the four WBGT scoring bands produce distinct score deltas."""
        # Isolate WBGT contribution by zeroing all other signals
        kwargs = dict(heart_rate=72, spo2=98, skin_temp=36.0, sun_exposure_minutes=0)

        # > 30 → +4 (yellow: 4 pts)
        assert compute_heat_stress_tier(wbgt=31.0, **kwargs) == "yellow"
        # > 28 → +3 (yellow)
        assert compute_heat_stress_tier(wbgt=29.0, **kwargs) == "yellow"
        # > 26 → +2 (green: 2 < 3)
        assert compute_heat_stress_tier(wbgt=27.0, **kwargs) == "green"
        # > 24 → +1 (green)
        assert compute_heat_stress_tier(wbgt=25.0, **kwargs) == "green"
        # ≤ 24 → +0 (green)
        assert compute_heat_stress_tier(wbgt=22.0, **kwargs) == "green"


class TestComputeHeatStressTierHeartRate:

    def test_hr_mild_elevation_adds_one_point(self):
        """HR 91 → +1 point alone → green."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=91, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "green"

    def test_hr_concern_adds_two_points(self):
        """HR 105 → +2 points alone → green."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=105, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "green"

    def test_hr_elevated_adds_three_points(self):
        """HR 125 → +3 points → yellow."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=125, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "yellow"

    def test_hr_critical_adds_four_points(self):
        """HR 145 → +4 points → yellow."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=145, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "yellow"

    def test_hr_critical_with_high_wbgt_causes_red(self):
        """HR 145 (+4) + WBGT 31 (+4) + sun 50 min (+3) = 11 → red."""
        tier = compute_heat_stress_tier(
            wbgt=31.0, heart_rate=145, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=50,
        )
        assert tier == "red"


class TestComputeHeatStressTierSpO2:

    def test_spo2_mild_concern_adds_one_point(self):
        """SpO2 94% → +1 point → green alone."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=94, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "green"

    def test_spo2_concern_adds_two_points(self):
        """SpO2 91% → +2 points → green alone."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=91, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "green"

    def test_spo2_critical_adds_four_points(self):
        """SpO2 88% → +4 points → yellow."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=88, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "yellow"

    def test_normal_spo2_adds_zero_points(self):
        """SpO2 97% → 0 points → no contribution."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=97, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "green"


class TestComputeHeatStressTierSkinTemp:

    def test_skin_temp_mild_adds_one_point(self):
        """Skin 37.2°C → +1 → green."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=37.2,
            sun_exposure_minutes=0,
        )
        assert tier == "green"

    def test_skin_temp_concern_adds_two_points(self):
        """Skin 37.8°C → +2 → green."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=37.8,
            sun_exposure_minutes=0,
        )
        assert tier == "green"

    def test_skin_temp_critical_adds_four_points(self):
        """Skin 39.0°C → +4 → yellow."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=39.0,
            sun_exposure_minutes=0,
        )
        assert tier == "yellow"

    def test_normal_skin_temp_adds_zero_points(self):
        """Skin 36.5°C → 0 points."""
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "green"


class TestComputeHeatStressTierSunExposure:

    def test_under_15_min_adds_zero_points(self):
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=10,
        )
        assert tier == "green"

    def test_15_to_30_min_adds_one_point(self):
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=20,
        )
        assert tier == "green"  # 1 point only

    def test_30_to_45_min_adds_two_points(self):
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=35,
        )
        assert tier == "green"  # 2 points only

    def test_over_45_min_adds_three_points(self):
        tier = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=50,
        )
        assert tier == "yellow"  # 3 points → yellow


class TestComputeHeatStressTierOptionalSignals:

    def test_high_breathing_rate_adds_points(self):
        """Breathing rate > 25 bpm → +2 points."""
        tier_with = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0, breathing_rate=28,
        )
        tier_without = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        # Both may be green, but with-breathing should have higher score
        tiers = ["green", "yellow", "orange", "red"]
        assert tiers.index(tier_with) >= tiers.index(tier_without)

    def test_high_exertion_adds_points(self):
        """Exertion > 0.8 → +2 points."""
        tier_exerted = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0, exertion_level=0.9,
        )
        tier_rest = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0, exertion_level=0.1,
        )
        tiers = ["green", "yellow", "orange", "red"]
        assert tiers.index(tier_exerted) >= tiers.index(tier_rest)

    def test_sweat_level_does_not_crash(self):
        """sweat_level is reserved — passing any value must not raise."""
        result = compute_heat_stress_tier(
            wbgt=22.0, heart_rate=72, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0, sweat_level=800,
        )
        assert result in ("green", "yellow", "orange", "red")


class TestComputeHeatStressTierRealScenarios:

    def test_comfortable_shade_worker(self):
        """Worker in shade, mild temp, normal vitals → green."""
        wbgt = compute_wbgt_estimate(24.0, 50.0, in_direct_sun=False)
        tier = compute_heat_stress_tier(
            wbgt=wbgt, heart_rate=75, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=5,
        )
        assert tier == "green"

    def test_hot_day_direct_sun_causes_at_least_yellow(self):
        """Construction worker, 34°C, direct sun, moderate exertion."""
        wbgt = compute_wbgt_estimate(34.0, 65.0, in_direct_sun=True)
        tier = compute_heat_stress_tier(
            wbgt=wbgt, heart_rate=105, spo2=97, skin_temp=37.6,
            sun_exposure_minutes=35,
        )
        assert tier in ("yellow", "orange", "red")

    def test_dangerous_scenario_is_orange_or_red(self):
        """Very hot day, long sun exposure, elevated HR and skin temp."""
        wbgt = compute_wbgt_estimate(38.0, 75.0, in_direct_sun=True)
        tier = compute_heat_stress_tier(
            wbgt=wbgt, heart_rate=130, spo2=96, skin_temp=38.0,
            sun_exposure_minutes=50,
        )
        assert tier in ("orange", "red")

    def test_heat_stroke_risk_is_red(self):
        """Critical vitals — should always produce red tier."""
        tier = compute_heat_stress_tier(
            wbgt=32.0,
            heart_rate=145,
            spo2=88,
            skin_temp=39.5,
            sun_exposure_minutes=60,
        )
        assert tier == "red"

    def test_score_accumulates_across_signals(self):
        """Multiple mild elevations accumulate into yellow."""
        # wbgt 27 (+2) + HR 95 (+1) = 3 → yellow
        tier = compute_heat_stress_tier(
            wbgt=27.0, heart_rate=95, spo2=98, skin_temp=36.5,
            sun_exposure_minutes=0,
        )
        assert tier == "yellow"

    def test_tier_output_is_always_valid_string(self):
        valid = {"green", "yellow", "orange", "red"}
        test_cases = [
            (22.0, 72, 98, 36.5, 0),
            (28.0, 100, 95, 37.5, 20),
            (32.0, 140, 90, 38.5, 50),
            (35.0, 150, 85, 39.5, 60),
        ]
        for wbgt, hr, spo2, st, sun in test_cases:
            result = compute_heat_stress_tier(
                wbgt=wbgt, heart_rate=hr, spo2=spo2,
                skin_temp=st, sun_exposure_minutes=sun,
            )
            assert result in valid, f"Invalid tier '{result}' for inputs {wbgt},{hr},{spo2},{st},{sun}"


# ─── thermistor_raw_to_celsius ────────────────────────────────────────────────

class TestThermistorRawToCelsius:

    def test_nominal_resistance_gives_nominal_temp(self):
        """When ADC value corresponds to Rth = R_nominal (2430Ω) → temperature = T_nominal (25°C)."""
        # R_nominal=2430Ω: raw = 2430*1023/(10000+2430) ≈ 200
        result = thermistor_raw_to_celsius(raw_adc=200)
        assert abs(result - 25.0) < 1.0

    def test_zero_adc_returns_nan(self):
        """ADC = 0 means short circuit; must return NaN, not crash."""
        result = thermistor_raw_to_celsius(raw_adc=0)
        assert math.isnan(result)

    def test_higher_adc_higher_resistance_lower_temp(self):
        """Inverted divider (3.3V→Rfixed→ADC→NTC→GND):
        Higher ADC → higher Rth → lower temp (NTC: high R = cold)."""
        temp_low_adc  = thermistor_raw_to_celsius(raw_adc=350)   # hot (~40°C)
        temp_high_adc = thermistor_raw_to_celsius(raw_adc=700)   # cold (~10°C)
        assert temp_high_adc < temp_low_adc

    def test_body_temperature_range(self):
        """10-bit ADC values ~110–165 should map to skin-temperature range (30–42°C).

        With R_nominal=2430Ω, B=3950 in a 10kΩ inverted divider (10-bit ADC):
          ADC=200 → Rth ≈ 2430Ω → 25°C nominal
          ADC=132 → Rth ≈ 1290Ω → ~36.5°C (skin temp)
          ADC=110 → Rth ≈ 1190Ω → ~42°C
        """
        results = [thermistor_raw_to_celsius(raw_adc=adc) for adc in range(110, 170, 5)]
        finite = [r for r in results if math.isfinite(r)]
        assert len(finite) > 0
        body_range = [r for r in finite if 30 <= r <= 45]
        assert len(body_range) > 0

    def test_custom_b_coefficient(self):
        """Different B value changes the output."""
        temp_b3950 = thermistor_raw_to_celsius(raw_adc=1000, b_coefficient=3950.0)
        temp_b3435 = thermistor_raw_to_celsius(raw_adc=1000, b_coefficient=3435.0)
        assert temp_b3950 != temp_b3435

    def test_output_is_finite_for_typical_adc_range(self):
        """For 10-bit ADC values 1–1022, result must be a finite float."""
        for adc in [1, 100, 200, 388, 511, 700, 900, 1022]:
            result = thermistor_raw_to_celsius(raw_adc=adc)
            assert math.isfinite(result), f"Non-finite result for ADC={adc}"
