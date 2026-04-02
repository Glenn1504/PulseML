"""
src/pipeline/run.py
-------------------
Entrypoint for the feature pipeline.

Reads raw vitals from Parquet, computes window features, and writes
train/val/test splits to the processed data directory.

Usage:
    python -m src.pipeline.run \
        --input data/raw/vitals.parquet \
        --output data/processed/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.pipeline.features import compute_window_features, train_test_split_by_patient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run(input_path: str, output_dir: str) -> None:
    inp = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading raw vitals from %s ...", inp)
    raw = pd.read_parquet(inp)
    log.info("Loaded %d rows.", len(raw))

    features = compute_window_features(raw)

    train, val, test = train_test_split_by_patient(features)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{split_name}_features.parquet"
        split_df.to_parquet(path, index=False)
        log.info("Saved %s → %s", split_name, path)

    log.info("Pipeline complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/vitals.parquet")
    parser.add_argument("--output", default="data/processed/")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
