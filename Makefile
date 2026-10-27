# Compatibility aliases for `python -m tianximem.dev`.

ARGS ?=

.PHONY: help sync lock fmt format-check lint typecheck serve config-check \
        qdrant-up qdrant-down image-build up down clean

help:
	@uv run python -m tianximem.dev help

sync:
	uv run python -m tianximem.dev sync

lock:
	uv run python -m tianximem.dev lock

fmt:
	uv run python -m tianximem.dev format

format-check:
	uv run python -m tianximem.dev format-check

lint:
	uv run python -m tianximem.dev lint

typecheck:
	uv run python -m tianximem.dev typecheck

serve:
	uv run python -m tianximem.dev serve

config-check:
	uv run python -m tianximem.dev config-check

qdrant-up:
	uv run python -m tianximem.dev qdrant-up $(ARGS)

qdrant-down:
	uv run python -m tianximem.dev qdrant-down $(ARGS)

image-build:
	uv run python -m tianximem.dev image-build $(ARGS)

up:
	uv run python -m tianximem.dev up $(ARGS)

down:
	uv run python -m tianximem.dev down $(ARGS)

clean:
	uv run python -m tianximem.dev clean
