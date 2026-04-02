"""
src/pipeline/ingest.py
----------------------
Simulated Kafka consumer for real-time vital sign ingestion.

In production this connects to a real Kafka broker. Here we simulate
the consumer with a generator that reads from Parquet and yields
records in timestamp order, rate-limited to simulate streaming.

The design is identical — swap `_simulate_consumer` for a real
`confluent_kafka.Consumer` and the rest of the pipeline is unchanged.

Usage (simulation):
    python -m src.pipeline.ingest \
        --source data/raw/vitals.parquet \
        --topic  icu.vitals \
        --rate   10          # records per second
"""

import argparse
import logging
import time
from typing import Generator

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TOPIC = "icu.vitals"


# ---------------------------------------------------------------------------
# Simulated consumer (no Kafka dependency required)
# ---------------------------------------------------------------------------

def _simulate_consumer(
    source_path: str,
    rate_per_second: float = 10.0,
) -> Generator[dict, None, None]:
    """
    Reads a Parquet file and yields rows as dicts at the specified rate,
    simulating a Kafka stream.
    """
    df = pd.read_parquet(source_path).sort_values(
        ["patient_id", "timestamp_minutes"]
    )
    interval = 1.0 / rate_per_second

    for _, row in df.iterrows():
        yield row.to_dict()
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Real Kafka consumer (uncomment and configure for production)
# ---------------------------------------------------------------------------

def _kafka_consumer(
    broker: str,
    topic: str,
    group_id: str = "pulseml-pipeline",
) -> Generator[dict, None, None]:
    """
    Reads from a real Kafka topic using confluent-kafka.
    Requires: pip install confluent-kafka
    """
    try:
        from confluent_kafka import Consumer, KafkaException
        import json
    except ImportError as e:
        raise ImportError("confluent-kafka is required: pip install confluent-kafka") from e

    consumer = Consumer({
        "bootstrap.servers": broker,
        "group.id":          group_id,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([topic])
    log.info("Subscribed to Kafka topic: %s @ %s", topic, broker)

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            record = json.loads(msg.value().decode("utf-8"))
            yield record
    finally:
        consumer.close()


# ---------------------------------------------------------------------------
# Windowed buffer — accumulates readings until we have a full 6h window
# ---------------------------------------------------------------------------

class VitalWindowBuffer:
    """
    Per-patient buffer that accumulates readings and flushes a complete
    window when enough time has elapsed.

    Parameters
    ----------
    window_minutes : float
        Width of the feature window (default: 360 = 6 hours).
    flush_interval_minutes : float
        How often to flush a window per patient (default: 30 = every reading).
    """

    def __init__(self, window_minutes: float = 360, flush_interval_minutes: float = 30):
        self.window_minutes          = window_minutes
        self.flush_interval_minutes  = flush_interval_minutes
        self._buffers: dict[str, list[dict]] = {}
        self._last_flush: dict[str, float]   = {}

    def add(self, record: dict) -> list[dict] | None:
        """
        Add a record to the patient's buffer.

        Returns a list of records in the current window if it's time to flush,
        otherwise returns None.
        """
        pid = record.get("patient_id")
        t   = record.get("timestamp_minutes", 0.0)

        if pid not in self._buffers:
            self._buffers[pid]    = []
            self._last_flush[pid] = t

        self._buffers[pid].append(record)

        # Evict records outside the window
        self._buffers[pid] = [
            r for r in self._buffers[pid]
            if r["timestamp_minutes"] >= t - self.window_minutes
        ]

        # Flush?
        if t - self._last_flush[pid] >= self.flush_interval_minutes:
            self._last_flush[pid] = t
            return list(self._buffers[pid])

        return None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_simulation(source_path: str, rate: float = 10.0) -> None:
    buffer  = VitalWindowBuffer()
    n_total = 0
    n_flush = 0

    log.info("Starting simulated Kafka consumer from %s at %.0f records/sec ...", source_path, rate)

    for record in _simulate_consumer(source_path, rate_per_second=rate):
        n_total += 1
        window = buffer.add(record)

        if window is not None:
            n_flush += 1
            pid = record["patient_id"]
            log.debug("Flushed window for %s: %d readings", pid, len(window))

        if n_total % 1000 == 0:
            log.info("Processed %d records, %d windows flushed.", n_total, n_flush)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/vitals.parquet")
    parser.add_argument("--topic",  default=TOPIC)
    parser.add_argument("--rate",   type=float, default=10.0,
                        help="Records per second (simulation mode)")
    args = parser.parse_args()
    run_simulation(args.source, args.rate)


if __name__ == "__main__":
    main()
