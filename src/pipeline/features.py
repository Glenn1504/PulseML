"""
src/pipeline/features.py
------------------------
Computes sliding-window aggregate features from raw vital sign readings.

For each patient, we roll a 6-hour window across their time series and
compute a rich set of statistical and clinical composite features.
These become the inputs to the XGBoost model; the raw sequences feed the LSTM.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

VITALS = ["hr", "sbp", "dbp", "spo2", "rr", "temp"]
WINDOW_HOURS = 6
MIN_READINGS = 3  # drop windows with fewer readings than this


def compute_window_features(df: pd.DataFrame, window_hours: float = WINDOW_HOURS) -> pd.DataFrame:
    """
    Vectorised window feature computation using pandas rolling API.

    Uses time-based rolling on timestamp_minutes grouped by patient_id.
    ~50-100x faster than the nested-loop approach.

    Parameters
    ----------
    df : pd.DataFrame
        Raw vitals with columns: patient_id, timestamp_minutes, hours_in_icu,
        hr, sbp, dbp, spo2, rr, temp, label.

    Returns
    -------
    pd.DataFrame
        One row per reading with engineered features + patient_id + label.
    """
    log.info(
        "Computing window features (window=%.0fh) for %d patients...",
        window_hours,
        df["patient_id"].nunique(),
    )

    window_minutes = int(window_hours * 60)
    parts = []

    for pid, pat in df.groupby("patient_id", sort=False):
        pat = pat.sort_values("timestamp_minutes").reset_index(drop=True)

        # Build a DatetimeIndex from timestamp_minutes so we can use
        # pandas time-based rolling (much faster than manual loops)
        pat.index = pd.to_timedelta(pat["timestamp_minutes"], unit="min")

        roller = pat[VITALS].rolling(f"{window_minutes}min", min_periods=MIN_READINGS)

        feat = pd.DataFrame(index=pat.index)
        feat["patient_id"] = pid
        feat["timestamp_minutes"] = pat["timestamp_minutes"].values
        feat["hours_in_icu"] = pat["hours_in_icu"].values
        feat["label"] = pat["label"].values
        feat["n_readings"] = roller[VITALS[0]].count().values

        for v in VITALS:
            col = pat[v]
            r = roller[v]

            feat[f"{v}_mean"] = r.mean().values
            feat[f"{v}_std"] = r.std().values
            feat[f"{v}_min"] = r.min().values
            feat[f"{v}_max"] = r.max().values

            feat[f"{v}_missing_rate"] = (
                col.isna()
                .rolling(f"{window_minutes}min", min_periods=1)
                .mean()
                .values
            )

            # Trend: rolling corr between vital and a synthetic 0..N index
            # Can yield NaN for constant windows; that's fine.
            ix_series = pd.Series(np.arange(len(col), dtype=float), index=col.index)
            feat[f"{v}_trend"] = (
                col.rolling(f"{window_minutes}min", min_periods=MIN_READINGS)
                .corr(ix_series)
                .values
            )

        # Drop rows where the window had too few readings
        feat = feat[feat["n_readings"] >= MIN_READINGS].reset_index(drop=True)

        # Clinical composites
        # Guard against tiny or zero SBP means to avoid exploding shock index
        safe_sbp = feat["sbp_mean"].where(feat["sbp_mean"].abs() > 1e-6, np.nan)
        feat["shock_index"] = feat["hr_mean"] / safe_sbp

        feat["pulse_pressure_mean"] = feat["sbp_mean"] - feat["dbp_mean"]
        feat["map_mean"] = feat["dbp_mean"] + feat["pulse_pressure_mean"] / 3.0
        feat["overall_missing_rate"] = feat[[f"{v}_missing_rate" for v in VITALS]].mean(axis=1)

        # Final numeric sanitization: replace inf/-inf with NaN
        numeric_cols = feat.select_dtypes(include=[np.number]).columns

        inf_count_before = int(np.isinf(feat[numeric_cols].to_numpy()).sum())
        if inf_count_before > 0:
            log.warning("Patient %s produced %d inf values in window features", pid, inf_count_before)

        huge_cols = []
        for c in numeric_cols:
            vals = feat[c].replace([np.inf, -np.inf], np.nan)
            if vals.notna().any():
                max_abs = vals.abs().max()
                if pd.notna(max_abs) and max_abs > 1e6:
                    huge_cols.append((c, float(max_abs)))

        if huge_cols:
            preview = ", ".join([f"{c}={v:.3g}" for c, v in huge_cols[:10]])
            log.warning("Patient %s has unusually large feature values: %s", pid, preview)

        feat[numeric_cols] = feat[numeric_cols].replace([np.inf, -np.inf], np.nan)

        parts.append(feat)

    result = pd.concat(parts, ignore_index=True)

    # Dataset-level sanitization as one final guardrail
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    total_inf = int(np.isinf(result[numeric_cols].to_numpy()).sum())
    if total_inf > 0:
        log.warning("Final feature table still had %d inf values; converting to NaN.", total_inf)
        result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)

    log.info("Feature table: %d rows × %d columns.", len(result), len(result.columns))
    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes ID, timestamp, label)."""
    exclude = {"patient_id", "timestamp_minutes", "label"}
    return [c for c in df.columns if c not in exclude]


def train_test_split_by_patient(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by patient_id to prevent data leakage.
    Returns (train_df, val_df, test_df).
    """
    rng = np.random.default_rng(seed)
    patients = df["patient_id"].unique()
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)

    test_ids  = set(patients[:n_test])
    val_ids   = set(patients[n_test:n_test + n_val])
    train_ids = set(patients[n_test + n_val:])

    train = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val   = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    test  = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)

    log.info(
        "Split: train=%d rows (%d pts) | val=%d rows (%d pts) | test=%d rows (%d pts)",
        len(train), len(train_ids),
        len(val),   len(val_ids),
        len(test),  len(test_ids),
    )
    return train, val, test