.PHONY: install format lint typecheck test test-cov all clean

install:
	pip install -e ".[dev]"

format:
	ruff format ai_researcher_assistant/ tests/

lint:
	ruff check ai_researcher_assistant/ tests/

typecheck:
	mypy ai_researcher_assistant/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov --cov-report=term-missing

all: format lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache .ruff_cache
