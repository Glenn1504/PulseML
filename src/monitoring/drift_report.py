"""
src/monitoring/drift_report.py
------------------------------
Generates an Evidently AI data drift report comparing a reference
(training) dataset to a current (production) dataset.

If the drift score on any feature exceeds DRIFT_THRESHOLD, a warning
is logged (and in production this would fire a Slack / PagerDuty alert).

Usage:
    python -m src.monitoring.drift_report \
        --reference data/processed/train_features.parquet \
        --current   data/processed/recent_features.parquet \
        --output    reports/drift_report.html
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.pipeline.features import get_feature_columns

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DRIFT_THRESHOLD = 0.15  # Evidently drift score above this triggers alert


def generate_drift_report(
    reference_path: str,
    current_path: str,
    output_path: str = "reports/drift_report.html",
) -> dict:
    """
    Computes feature drift between reference and current datasets.

    Returns a dict with:
        drifted_features  list[str]   — features that crossed the threshold
        max_drift_score   float
        report_path       str
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric
    except ImportError as exc:
        raise ImportError("evidently is required: pip install evidently") from exc

    log.info("Loading reference: %s", reference_path)
    reference = pd.read_parquet(reference_path)
    log.info("Loading current:   %s", current_path)
    current = pd.read_parquet(current_path)

    feat_cols = get_feature_columns(reference)

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(
        reference_data=reference[feat_cols].fillna(0),
        current_data=current[feat_cols].fillna(0),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    log.info("Drift report saved → %s", out)

    # Parse results dict to find drifted features
    result_dict = report.as_dict()
    drifted_features = []
    max_score = 0.0

    for metric in result_dict.get("metrics", []):
        result = metric.get("result", {})
        drift_by_col = result.get("drift_by_columns", {})
        for col, info in drift_by_col.items():
            score = info.get("drift_score", 0.0)
            max_score = max(max_score, score)
            if score > DRIFT_THRESHOLD:
                drifted_features.append(col)
                log.warning("DRIFT DETECTED: feature=%s score=%.3f (threshold=%.2f)",
                            col, score, DRIFT_THRESHOLD)

    if drifted_features:
        log.warning("⚠️  %d feature(s) drifted beyond threshold. "
                    "Consider retraining.", len(drifted_features))
        _fire_alert(drifted_features, max_score)
    else:
        log.info("✅  No significant drift detected (max score=%.3f).", max_score)

    return {
        "drifted_features": drifted_features,
        "max_drift_score": round(max_score, 4),
        "report_path": str(out),
    }


def _fire_alert(drifted_features: list[str], max_score: float) -> None:
    """
    Placeholder for alert integration (Slack, PagerDuty, etc.).
    In production: POST to Slack webhook or PagerDuty Events API.
    """
    log.warning(
        "ALERT: Data drift detected on %d feature(s). Max score=%.3f. "
        "Features: %s",
        len(drifted_features), max_score, ", ".join(drifted_features[:5]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current",   required=True)
    parser.add_argument("--output",    default="reports/drift_report.html")
    args = parser.parse_args()

    result = generate_drift_report(args.reference, args.current, args.output)
    print(result)


if __name__ == "__main__":
    main()
