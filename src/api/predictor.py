"""
src/api/predictor.py
--------------------
Loads the trained ensemble from disk and runs inference for a PredictRequest.

The Predictor reconstructs the same tabular features that the pipeline
produces, then calls the ensemble.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.schemas import PredictRequest, VitalReading
from src.models.ensemble import DeteriorationEnsemble

log = logging.getLogger(__name__)

VITALS = ["hr", "sbp", "dbp", "spo2", "rr", "temp"]
DEFAULT_MODEL_DIR = "models/latest"
DEFAULT_MODEL_VERSION = "v1.0.0"


def _reading_to_dict(r: VitalReading, idx: int) -> dict:
    return {
        "timestamp_minutes": idx * 30.0,  # treat as 30-min intervals
        "hr": r.hr, "sbp": r.sbp, "dbp": r.dbp,
        "spo2": r.spo2, "rr": r.rr, "temp": r.temp,
    }


def _compute_features_from_request(request: PredictRequest) -> pd.DataFrame:
    """Reconstruct window features from a PredictRequest on-the-fly."""
    rows = [_reading_to_dict(r, i) for i, r in enumerate(request.vitals_window)]
    df = pd.DataFrame(rows)

    feat: dict = {"hours_in_icu": len(df) * 0.5}  # rough estimate

    for v in VITALS:
        col = df[v].dropna()
        feat[f"{v}_mean"] = col.mean() if len(col) > 0 else np.nan
        feat[f"{v}_std"]  = col.std()  if len(col) > 1 else 0.0
        feat[f"{v}_min"]  = col.min()  if len(col) > 0 else np.nan
        feat[f"{v}_max"]  = col.max()  if len(col) > 0 else np.nan

        # Trend: OLS slope
        y = col.values
        if len(y) >= 2:
            x = np.arange(len(y), dtype=float)
            x -= x.mean()
            denom = (x ** 2).sum()
            feat[f"{v}_trend"] = float((x * (y - y.mean())).sum() / denom) if denom > 0 else 0.0
        else:
            feat[f"{v}_trend"] = 0.0

        feat[f"{v}_missing_rate"] = df[v].isna().mean()

    hr_mean  = feat.get("hr_mean", np.nan)
    sbp_mean = feat.get("sbp_mean", np.nan)
    feat["shock_index"]        = hr_mean / sbp_mean if sbp_mean and sbp_mean > 0 else np.nan
    feat["pulse_pressure_mean"] = feat.get("sbp_mean", np.nan) - feat.get("dbp_mean", np.nan)
    feat["map_mean"]           = feat.get("dbp_mean", np.nan) + feat.get("pulse_pressure_mean", np.nan) / 3
    feat["overall_missing_rate"] = df[VITALS].isna().mean().mean()
    feat["n_readings"] = len(df)

    return pd.DataFrame([feat])


def _build_sequence(request: PredictRequest, seq_len: int = 12) -> np.ndarray:
    """Build a (1, seq_len, n_vitals) array from the last seq_len readings."""
    df = pd.DataFrame([
        {v: getattr(r, v) for v in VITALS}
        for r in request.vitals_window
    ])
    # Pad or truncate to seq_len
    if len(df) < seq_len:
        pad = pd.DataFrame([df.iloc[0].to_dict()] * (seq_len - len(df)))
        df = pd.concat([pad, df], ignore_index=True)
    df = df.tail(seq_len).reset_index(drop=True)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return df[VITALS].values[np.newaxis, :, :].astype(np.float32)  # (1, 12, 6)


class Predictor:
    def __init__(self, ensemble: DeteriorationEnsemble, model_version: str):
        self.ensemble      = ensemble
        self.model_version = model_version

    @classmethod
    def from_config(
        cls,
        model_dir: str = DEFAULT_MODEL_DIR,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> "Predictor":
        model_path = Path(model_dir)
        if not (model_path / "ensemble.pkl").exists():
            # No trained model found — use a stub that returns random predictions.
            # This lets the API start up for development without a trained model.
            log.warning(
                "No ensemble.pkl found at %s. Using stub predictor (dev mode).",
                model_dir,
            )
            return cls(_StubEnsemble(), model_version="dev-stub")

        ensemble = DeteriorationEnsemble.load(model_path)
        log.info("Loaded ensemble from %s", model_path)
        return cls(ensemble, model_version)

    def predict(self, request: PredictRequest) -> dict:
        tabular = _compute_features_from_request(request)
        seq     = _build_sequence(request)
        return self.ensemble.predict(tabular, seq, return_shap=True)


class _StubEnsemble:
    """Fake ensemble that returns plausible-looking random predictions.
    Used in development when no trained model exists."""

    threshold = 0.35

    def predict(self, tabular_features, sequences, return_shap=True):
        import random
        prob = round(random.uniform(0.1, 0.9), 4)
        risk = "HIGH" if prob >= 0.55 else "MEDIUM" if prob >= 0.35 else "LOW"
        result = {"probability": prob, "risk_level": risk}
        if return_shap:
            result["top_features"] = [
                {"name": "spo2_min",    "shap_value": round(random.uniform(0.1, 0.4), 4)},
                {"name": "hr_trend",    "shap_value": round(random.uniform(0.1, 0.3), 4)},
                {"name": "shock_index", "shap_value": round(random.uniform(0.05, 0.2), 4)},
            ]
        return result