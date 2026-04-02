"""
tests/test_model.py
-------------------
Unit tests for the ensemble model — feature construction, SHAP output,
and serialisation round-trip.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.api.predictor import _compute_features_from_request, _build_sequence, _StubEnsemble
from src.api.schemas import PredictRequest, VitalReading
from src.models.ensemble import DeteriorationEnsemble


def _make_request(n_readings: int = 12) -> PredictRequest:
    from datetime import datetime, timedelta, timezone
    base = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
    readings = [
        VitalReading(
            timestamp=base + timedelta(minutes=30 * i),
            hr=float(80 + i), sbp=120.0, dbp=75.0,
            spo2=97.0, rr=16.0, temp=37.0,
        )
        for i in range(n_readings)
    ]
    return PredictRequest(patient_id="P-test", vitals_window=readings)


class TestComputeFeaturesFromRequest:
    def test_returns_single_row_df(self):
        req = _make_request()
        feat = _compute_features_from_request(req)
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) == 1

    def test_has_vital_columns(self):
        req = _make_request()
        feat = _compute_features_from_request(req)
        for vital in ["hr", "sbp", "spo2"]:
            assert f"{vital}_mean" in feat.columns

    def test_shock_index_computed(self):
        req = _make_request()
        feat = _compute_features_from_request(req)
        assert "shock_index" in feat.columns
        assert not pd.isna(feat["shock_index"].iloc[0])


class TestBuildSequence:
    def test_output_shape(self):
        req = _make_request(n_readings=12)
        seq = _build_sequence(req, seq_len=12)
        assert seq.shape == (1, 12, 6)

    def test_padding_for_short_windows(self):
        req = _make_request(n_readings=5)
        seq = _build_sequence(req, seq_len=12)
        assert seq.shape == (1, 12, 6)

    def test_truncation_for_long_windows(self):
        req = _make_request(n_readings=30)
        seq = _build_sequence(req, seq_len=12)
        assert seq.shape == (1, 12, 6)

    def test_no_nans_in_output(self):
        req = _make_request(n_readings=12)
        seq = _build_sequence(req)
        assert not np.isnan(seq).any()


class TestStubEnsemble:
    def test_returns_dict_with_required_keys(self):
        stub = _StubEnsemble()
        result = stub.predict(None, None, return_shap=True)
        assert "probability" in result
        assert "risk_level"  in result
        assert "top_features" in result

    def test_probability_in_range(self):
        stub = _StubEnsemble()
        for _ in range(20):
            prob = stub.predict(None, None)["probability"]
            assert 0.0 <= prob <= 1.0

    def test_risk_level_consistent_with_threshold(self):
        stub = _StubEnsemble()
        for _ in range(50):
            result = stub.predict(None, None)
            prob = result["probability"]
            risk = result["risk_level"]
            if prob >= 0.55:
                assert risk == "HIGH"
            elif prob >= 0.35:
                assert risk == "MEDIUM"
            else:
                assert risk == "LOW"


class TestEnsembleSerialisation:
    def test_save_and_load_roundtrip(self):
        ensemble = DeteriorationEnsemble()
        with tempfile.TemporaryDirectory() as tmp:
            ensemble.save(tmp)
            loaded = DeteriorationEnsemble.load(tmp)
            assert isinstance(loaded, DeteriorationEnsemble)
            assert loaded.xgb_weight == ensemble.xgb_weight
            assert loaded.threshold  == ensemble.threshold
