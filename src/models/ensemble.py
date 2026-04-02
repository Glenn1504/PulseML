"""
src/models/ensemble.py
----------------------
Two-model ensemble: XGBoost on tabular window features + LSTM on raw time series.

The final probability is a weighted blend:
    P(deterioration) = 0.55 * P_xgb + 0.45 * P_lstm

SHAP values from the XGBoost component make predictions interpretable
without sacrificing the temporal dynamics the LSTM captures.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XGBoost wrapper
# ---------------------------------------------------------------------------

class XGBoostDetector:
    """Thin wrapper around xgboost.XGBClassifier with SHAP support."""

    def __init__(self, **xgb_kwargs):
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("xgboost is required: pip install xgboost") from e

        defaults = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=3,   # handles class imbalance (~25% positive)
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(xgb_kwargs)
        self.model = XGBClassifier(**defaults)
        self.feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series) -> None:
        self.feature_names = list(X_train.columns)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        log.info("XGBoost best iteration: %d", self.model.best_iteration)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values for the positive class."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(X[self.feature_names])
        except ImportError:
            log.warning("shap not installed — returning zeros.")
            return np.zeros((len(X), len(self.feature_names)))

    def top_features(self, X: pd.DataFrame, n: int = 5) -> list[dict]:
        """Return top-n features by |SHAP value| for a single-row DataFrame."""
        shap_vals = self.shap_values(X)[0]
        pairs = sorted(
            zip(self.feature_names, shap_vals),
            key=lambda t: abs(t[1]),
            reverse=True,
        )
        return [{"name": name, "shap_value": round(float(val), 4)} for name, val in pairs[:n]]


# ---------------------------------------------------------------------------
# LSTM wrapper (PyTorch)
# ---------------------------------------------------------------------------

class LSTMDetector:
    """
    A 2-layer LSTM that ingests raw vital-sign sequences.

    Input shape: (batch, seq_len, n_vitals)  — we use seq_len=12, n_vitals=6
    Output: scalar deterioration probability per patient window.
    """

    SEQ_LEN  = 12
    N_VITALS = 6
    VITALS   = ["hr", "sbp", "dbp", "spo2", "rr", "temp"]

    def __init__(self, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, lr: float = 1e-3):
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr
        self._model      = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal PyTorch model (defined lazily to avoid import-time errors
    # when torch is not installed)
    # ------------------------------------------------------------------

    def _build_model(self):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_vitals, hidden, layers, drop):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_vitals, hidden_size=hidden,
                    num_layers=layers, batch_first=True, dropout=drop,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden, 32),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):                       # x: (B, T, V)
                out, _ = self.lstm(x)                   # (B, T, H)
                return self.head(out[:, -1, :]).squeeze(-1)  # (B,)

        return _Net(self.N_VITALS, self.hidden_size, self.num_layers, self.dropout)

    def _raw_to_sequences(self, raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert raw vitals DataFrame into fixed-length sequences per window.
        For simplicity we sample the last SEQ_LEN readings per 6h window
        (indexed by patient_id + timestamp). Returns X (N, SEQ, V) and y (N,).
        """
        sequences, labels = [], []
        for pid, pat in raw_df.groupby("patient_id"):
            pat = pat.sort_values("timestamp_minutes")
            vitals = pat[self.VITALS].values           # (T, 6)
            y_vals = pat["label"].values               # (T,)

            # Stride through the series with stride=1
            for end in range(self.SEQ_LEN, len(vitals) + 1):
                seq = vitals[end - self.SEQ_LEN: end]  # (12, 6)
                # Simple forward-fill then mean imputation
                seq = pd.DataFrame(seq).ffill().bfill().fillna(0).values
                sequences.append(seq)
                labels.append(y_vals[end - 1])

        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        return X, y

    def fit(self, raw_train: pd.DataFrame, raw_val: pd.DataFrame,
            epochs: int = 20, batch_size: int = 256) -> None:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:
            raise ImportError("torch is required: pip install torch") from e

        X_tr, y_tr = self._raw_to_sequences(raw_train)
        X_va, y_va = self._raw_to_sequences(raw_val)

        # Fit scaler on train
        self._scaler_mean = X_tr.reshape(-1, self.N_VITALS).mean(axis=0)
        self._scaler_std  = X_tr.reshape(-1, self.N_VITALS).std(axis=0) + 1e-8

        X_tr = (X_tr - self._scaler_mean) / self._scaler_std
        X_va = (X_va - self._scaler_mean) / self._scaler_std

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("LSTM training on: %s", device)

        self._model = self._build_model().to(device)
        opt      = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn  = nn.BCELoss()

        tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            self._model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(self._model(xb), yb).backward()
                opt.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_va).to(device)
                yv = torch.tensor(y_va).to(device)
                val_loss = loss_fn(self._model(xv), yv).item()

            log.info("LSTM epoch %2d/%2d — val_loss=%.4f", epoch, epochs, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Keep best weights in memory
                self._best_state = {k: v.clone() for k, v in self._model.state_dict().items()}

        self._model.load_state_dict(self._best_state)
        log.info("LSTM training complete. Best val_loss=%.4f", best_val_loss)

    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        sequences : np.ndarray, shape (N, SEQ_LEN, N_VITALS)
            Already normalized sequences.
        """
        import torch
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(sequences, dtype=torch.float32)
            return self._model(x).numpy()

    def preprocess(self, sequences: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler."""
        if self._scaler_mean is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return (sequences - self._scaler_mean) / self._scaler_std


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

@dataclass
class DeteriorationEnsemble:
    """Weighted ensemble of XGBoost + LSTM detectors."""

    xgb_weight: float = 0.55
    lstm_weight: float = 0.45
    threshold:   float = 0.35

    xgb:  XGBoostDetector = field(default_factory=XGBoostDetector)
    lstm: LSTMDetector     = field(default_factory=LSTMDetector)

    def predict(
        self,
        tabular_features: pd.DataFrame,
        sequences: np.ndarray,
        return_shap: bool = False,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
            probability   float  — ensemble deterioration probability
            risk_level    str    — LOW / MEDIUM / HIGH
            top_features  list   — top SHAP features (if return_shap=True)
        """
        p_xgb = self.xgb.predict_proba(tabular_features)

        lstm_trained = self.lstm._scaler_mean is not None
        if lstm_trained and self.lstm_weight > 0:
            p_lstm = self.lstm.predict_proba(self.lstm.preprocess(sequences))
            total  = self.xgb_weight + self.lstm_weight
            prob   = (self.xgb_weight / total) * p_xgb + (self.lstm_weight / total) * p_lstm
        else:
            prob = p_xgb

        # Scalar for single-patient inference
        p_scalar = float(np.atleast_1d(prob)[0])

        risk = (
            "HIGH"   if p_scalar >= self.threshold + 0.20 else
            "MEDIUM" if p_scalar >= self.threshold else
            "LOW"
        )

        result = {"probability": round(p_scalar, 4), "risk_level": risk}

        if return_shap:
            result["top_features"] = self.xgb.top_features(tabular_features, n=5)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "ensemble.pkl", "wb") as f:
            pickle.dump(self, f)
        log.info("Ensemble saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DeteriorationEnsemble":
        with open(Path(path) / "ensemble.pkl", "rb") as f:
            obj = pickle.load(f)
        log.info("Ensemble loaded from %s", path)
        return obj