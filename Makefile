.PHONY: install format lint test docker-up docker-down

install:
	uv python install 3.11
	uv sync --group dev

format:
	uv run black .

lint:
	uv run ruff check src tests

test:
	uv run pytest --cov=src --cov-report=term-missing tests/ -v

docker-up:
	docker compose up -d

docker-down:
	docker compose down
