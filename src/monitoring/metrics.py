"""
src/monitoring/metrics.py
--------------------------
Prometheus metrics definitions shared across the application.

Import these in src/api/main.py rather than re-defining them — Prometheus
raises a ValueError if you register the same metric name twice.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary

# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------

PREDICTIONS_TOTAL = Counter(
    "pulseml_predictions_total",
    "Total number of predictions served",
    ["risk_level", "model_version"],
)

PREDICTION_LATENCY = Histogram(
    "pulseml_prediction_latency_seconds",
    "End-to-end prediction latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_PROBABILITY = Histogram(
    "pulseml_prediction_probability_distribution",
    "Distribution of predicted deterioration probabilities",
    buckets=[i / 10 for i in range(11)],
)

# ---------------------------------------------------------------------------
# Model health metrics
# ---------------------------------------------------------------------------

MODEL_DRIFT_SCORE = Gauge(
    "pulseml_feature_drift_score",
    "Latest Evidently drift score for a given feature",
    ["feature_name"],
)

DRIFT_ALERTS_TOTAL = Counter(
    "pulseml_drift_alerts_total",
    "Number of times drift exceeded the alert threshold",
)

MODEL_LOADED_AT = Gauge(
    "pulseml_model_loaded_timestamp_seconds",
    "Unix timestamp when the current model was loaded",
)

# ---------------------------------------------------------------------------
# Data pipeline metrics
# ---------------------------------------------------------------------------

PIPELINE_RUNS_TOTAL = Counter(
    "pulseml_pipeline_runs_total",
    "Total feature pipeline runs",
    ["status"],   # success / failure
)

PIPELINE_DURATION = Summary(
    "pulseml_pipeline_duration_seconds",
    "Duration of each feature pipeline run",
)

FEATURES_COMPUTED = Counter(
    "pulseml_features_computed_total",
    "Total number of feature rows computed by the pipeline",
)
