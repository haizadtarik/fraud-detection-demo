.PHONY: data data-full clean-data

data:            ## Features from the committed sample (fast, runs anywhere)
	python -m fraud_detection.pipelines.data_pipeline --source sample

data-full:       ## Features from the full Kaggle CSV (real metrics)
	python -m fraud_detection.pipelines.data_pipeline --source full

clean-data:      ## Remove generated data artifacts
	rm -f data/02_intermediate/* data/03_features/*

.PHONY: train mlflow-ui

train:           ## Train + evaluate on the current feature table (full-data by default)
	python -m fraud_detection.pipelines.training_pipeline --source full

mlflow-ui:       ## Browse experiment runs
	mlflow ui

.PHONY: serve

serve:           ## Run the scoring API locally
	uvicorn fraud_detection.serving.app:app --reload --port 8000

.PHONY: monitor

monitor:         ## Run drift monitoring (PSI + Evidently)
	python -m fraud_detection.monitoring.drift_report

.PHONY: lint format format-check check

lint:            ## Lint and auto-fix with ruff
	ruff check --fix .

format:          ## Format code with ruff
	ruff format .

format-check:    ## Check formatting without modifying files (CI)
	ruff format --check .
	ruff check .

check: format-check  ## Alias for format-check

.PHONY: install-hooks

install-hooks:   ## Install pre-commit git hooks
	pre-commit install