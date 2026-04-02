"""
generate_data.py
----------------
Synthesises realistic ICU vital-sign time series for n_patients.

Each patient has:
  - A random ICU stay length (12–96 hours)
  - Vitals sampled every ~30 min with realistic noise and missingness
  - A binary deterioration label (1 if the patient deteriorated within
    the next 6 hours of any given window)
  - Deteriorating patients show physiologically consistent trends
    (rising HR, falling SpO2, rising RR, unstable BP) in the hours
    before the event.

Usage:
    python scripts/generate_data.py --n_patients 5000 --output data/raw/vitals.parquet
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VITALS = ["hr", "sbp", "dbp", "spo2", "rr", "temp"]

# (mean, std) for a stable ICU patient
STABLE_PARAMS = {
    "hr":   (82,  10),
    "sbp":  (120, 15),
    "dbp":  (75,  10),
    "spo2": (97,   1.5),
    "rr":   (16,   2.5),
    "temp": (37.0, 0.3),
}

# drift applied per hour in the 6h before an event (positive = rising)
DETERIORATION_DRIFT = {
    "hr":   +2.5,
    "sbp":  -3.0,
    "dbp":  -2.0,
    "spo2": -0.8,
    "rr":   +1.2,
    "temp": +0.15,
}

RNG = np.random.default_rng(42)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _generate_patient(patient_id: str, deteriorates: bool) -> pd.DataFrame:
    stay_hours = RNG.integers(12, 96)
    event_hour = int(stay_hours * RNG.uniform(0.5, 0.85)) if deteriorates else None

    rows = []
    # Readings roughly every 30 min with some jitter
    times = np.arange(0, stay_hours * 60, 30) + RNG.integers(-5, 5, size=stay_hours * 2)
    times = np.clip(times, 0, stay_hours * 60)

    state = {v: RNG.normal(*STABLE_PARAMS[v]) for v in VITALS}

    for t in times:
        hours_elapsed = t / 60.0
        drifted = dict(state)

        if deteriorates and event_hour is not None:
            hours_to_event = event_hour - hours_elapsed
            if 0 <= hours_to_event <= 6:
                # Progressive deterioration in the 6h window
                drift_scale = (6 - hours_to_event) / 6
                for v in VITALS:
                    drifted[v] += DETERIORATION_DRIFT[v] * drift_scale

        # Add per-reading noise
        reading = {
            "patient_id": patient_id,
            "timestamp_minutes": float(t),
            "hours_in_icu": hours_elapsed,
            "label": int(deteriorates and event_hour is not None and 0 <= (event_hour - hours_elapsed) <= 6),
        }
        for v in VITALS:
            noise = RNG.normal(0, STABLE_PARAMS[v][1] * 0.3)
            val = drifted[v] + noise
            # Physiological clamps
            clamps = {
                "hr": (25, 220), "sbp": (60, 220), "dbp": (30, 140),
                "spo2": (70, 100), "rr": (4, 50), "temp": (34.0, 42.0),
            }
            reading[v] = round(_clamp(val, *clamps[v]), 2)
            # ~8% missingness per vital
            if RNG.random() < 0.08:
                reading[v] = float("nan")

        rows.append(reading)

    return pd.DataFrame(rows)


def generate(n_patients: int, deterioration_rate: float = 0.25) -> pd.DataFrame:
    log.info("Generating data for %d patients (%.0f%% deterioration rate)...", n_patients, deterioration_rate * 100)
    frames = []
    n_deteriorating = int(n_patients * deterioration_rate)

    patient_ids = [f"P-{i:05d}" for i in range(n_patients)]
    labels = [True] * n_deteriorating + [False] * (n_patients - n_deteriorating)
    RNG.shuffle(labels)

    for pid, det in tqdm(zip(patient_ids, labels), total=n_patients):
        frames.append(_generate_patient(pid, det))

    df = pd.concat(frames, ignore_index=True)
    log.info("Generated %d vital-sign readings across %d patients.", len(df), n_patients)
    log.info("Deterioration rate in data: %.2f%%", df.groupby("patient_id")["label"].max().mean() * 100)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ICU vitals data.")
    parser.add_argument("--n_patients", type=int, default=5000)
    parser.add_argument("--deterioration_rate", type=float, default=0.25)
    parser.add_argument("--output", type=str, default="data/raw/vitals.parquet")
    args = parser.parse_args()

    df = generate(args.n_patients, args.deterioration_rate)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("Saved → %s  (%.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
