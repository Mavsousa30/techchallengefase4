.PHONY: setup run test lint ci clean help

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install project dependencies
	pip install -r requirements.txt

run: ## Run the main pipeline with default video
	python3 -m src.main --video data/input_video/video.mp4 --save-preview

web: ## Run web interface (Streamlit)
	./run_web.sh

test: ## Run tests with pytest
	pytest -q

lint: ## Run linting and type checking
	ruff check . && mypy src

ci: lint test ## Run CI pipeline (lint + test)

clean: ## Clean up generated files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf outputs/logs/* outputs/frames/*
