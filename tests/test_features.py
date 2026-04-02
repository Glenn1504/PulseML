"""
tests/test_features.py
-----------------------
Unit tests for the feature engineering pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import (
    compute_window_features,
    get_feature_columns,
    train_test_split_by_patient,
)


def _make_raw_df(n_patients: int = 5, readings_per_patient: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_patients):
        pid = f"P-{i:03d}"
        for t in range(readings_per_patient):
            rows.append({
                "patient_id": pid,
                "timestamp_minutes": float(t * 30),
                "hours_in_icu": t * 0.5,
                "label": int(t >= readings_per_patient - 5),
                "hr":   float(rng.normal(82, 10)),
                "sbp":  float(rng.normal(120, 15)),
                "dbp":  float(rng.normal(75, 10)),
                "spo2": float(rng.normal(97, 1.5)),
                "rr":   float(rng.normal(16, 2.5)),
                "temp": float(rng.normal(37.0, 0.3)),
            })
    return pd.DataFrame(rows)


class TestComputeWindowFeatures:
    def test_returns_dataframe(self):
        raw = _make_raw_df()
        features = compute_window_features(raw, window_hours=3)
        assert isinstance(features, pd.DataFrame)

    def test_has_expected_columns(self):
        raw = _make_raw_df()
        features = compute_window_features(raw, window_hours=3)
        for vital in ["hr", "sbp", "spo2"]:
            assert f"{vital}_mean" in features.columns
            assert f"{vital}_std"  in features.columns
            assert f"{vital}_trend" in features.columns

    def test_clinical_composites_present(self):
        raw = _make_raw_df()
        features = compute_window_features(raw, window_hours=3)
        assert "shock_index" in features.columns
        assert "map_mean"    in features.columns

    def test_no_label_leakage_in_feature_cols(self):
        raw = _make_raw_df()
        features = compute_window_features(raw, window_hours=3)
        feat_cols = get_feature_columns(features)
        assert "label" not in feat_cols

    def test_missing_vitals_handled(self):
        raw = _make_raw_df()
        # Introduce 30% missingness
        for v in ["hr", "spo2"]:
            mask = np.random.rand(len(raw)) < 0.3
            raw.loc[mask, v] = np.nan
        features = compute_window_features(raw, window_hours=3)
        assert len(features) > 0


class TestTrainTestSplit:
    def test_no_patient_overlap(self):
        raw = _make_raw_df(n_patients=20)
        features = compute_window_features(raw, window_hours=3)
        train, val, test = train_test_split_by_patient(features)

        train_ids = set(train["patient_id"])
        val_ids   = set(val["patient_id"])
        test_ids  = set(test["patient_id"])

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_patients_accounted_for(self):
        raw = _make_raw_df(n_patients=20)
        features = compute_window_features(raw, window_hours=3)
        train, val, test = train_test_split_by_patient(features)

        all_ids = (
            set(train["patient_id"]) |
            set(val["patient_id"])   |
            set(test["patient_id"])
        )
        assert all_ids == set(features["patient_id"])
