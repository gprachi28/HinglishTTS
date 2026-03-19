PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: install test lint format clean

install:
	python3.12 -m venv .venv
	$(PIP) install -r requirements.txt
	.venv/bin/pre-commit install

test:
	.venv/bin/pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	.venv/bin/flake8 . --max-line-length=100

format:
	.venv/bin/black . && .venv/bin/isort .

data:
	$(PYTHON) data/codeswitching.py --num_sentences 5000
	$(PYTHON) -m data.script_variants
	$(PYTHON) data/preprocess_streaming.py --max_samples 1000

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete