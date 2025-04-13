.PHONY: all setup run run-spapi test lint visualize docker docker-run clean help

# Default target
all: help

# Setup environment
setup:
	poetry install
	cp -n .env.template .env || true
	@echo "Setup complete. Edit .env file with your SP-API credentials."

# Run the application
run:
	poetry run python src/main.py

# Run with SP-API integration
run-spapi:
	poetry run python src/main.py --use-spapi

# Run with CLI and show help
cli-help:
	poetry run python src/cli.py --help

# Run tests
test:
	poetry run pytest

# Run oneAPI integration tests
test-oneapi:
	poetry run pytest tests/test_oneapi_integration.py -v

# Run linting
lint:
	poetry run black src tests
	poetry run isort src tests
	poetry run flake8 src tests

# Run visualization only
visualize:
	@if [ -f output/inventory_analysis.csv ]; then \
		poetry run python -c "from src.visualization import visualize_results; visualize_results()"; \
	else \
		echo "Error: Run analysis first to generate output/inventory_analysis.csv"; \
		exit 1; \
	fi

# Build Docker image
docker:
	docker build -t bibliophile-sp .

# Run with Docker
docker-run:
	docker run -it -v $(pwd)/output:/app/output --env-file .env bibliophile-sp

# Run with Docker Compose
docker-compose:
	docker-compose up --build

# Clean output files
clean:
	rm -rf output/*.csv output/*.png

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup            - Install dependencies and create .env file"
	@echo "  make run              - Run the application with dummy data"
	@echo "  make run-spapi        - Run the application with SP-API data"
	@echo "  make cli-help         - Show CLI help"
	@echo "  make test             - Run tests"
	@echo "  make test-oneapi      - Run oneAPI integration tests"
	@echo "  make lint             - Run linters"
	@echo "  make visualize        - Generate visualizations from existing analysis"
	@echo "  make docker           - Build Docker image"
	@echo "  make docker-run       - Run with Docker"
	@echo "  make docker-compose   - Run with Docker Compose"
	@echo "  make clean            - Remove output files"
