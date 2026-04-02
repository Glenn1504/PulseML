.PHONY: install data pipeline train serve test lint docker-up docker-down eval monitor clean

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	python3 -m pip install --upgrade pip
	pip3 install -r requirements.txt

# ── Data ───────────────────────────────────────────────────────────────────
data:
	python3 scripts/generate_data.py --n_patients 5000 --output data/raw/vitals.parquet

# ── Pipeline ───────────────────────────────────────────────────────────────
pipeline:
	python3 -m src.pipeline.run --input data/raw/vitals.parquet --output data/processed/

# ── Training ───────────────────────────────────────────────────────────────
train:
	python3 -m src.models.train --config configs/train_config.yaml

mlflow-ui:
	mlflow ui --port 5000

# ── Evaluation ─────────────────────────────────────────────────────────────
eval:
	python3 -m src.models.evaluate \
		--model_dir models/latest \
		--test_data data/processed/test_features.parquet \
		--output    reports/eval/

# ── Serving ────────────────────────────────────────────────────────────────
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Monitoring ─────────────────────────────────────────────────────────────
monitor:
	python3 -m src.monitoring.drift_report \
		--reference data/processed/train_features.parquet \
		--current   data/processed/val_features.parquet \
		--output    reports/drift_report.html

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --tb=short

# ── Lint ───────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/

# ── Docker ─────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

# ── Full run (dev) ──────────────────────────────────────────────────────────
run-all: install data pipeline train eval serve

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/ dist/ build/ *.egg-info/
