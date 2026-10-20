# Compatibility aliases for `python -m tianximem.dev`.

ARGS ?=
RUN_ARG = $(if $(RUN),--run-id $(RUN))
PROCESSES_ARG = $(if $(PROCESSES),--processes $(PROCESSES))
ROUTER_OUT_ARG = $(if $(ROUTER_OUT),--out $(ROUTER_OUT))
COQ_OUT_ARG = $(if $(COQ_OUT),--out $(COQ_OUT))
DATASET ?= locomo-refined
DATASET_FLAG = $(if $(filter all,$(DATASET)),--tier all,--dataset $(DATASET))

.PHONY: help sync check config-check fmt lint typecheck test qdrant-up qdrant-down serve contract-check probe-reranker \
        router-eval coq-eval coq-decompose order-probe export-multihop image-build up down deploy-check \
        eval baseline replay-official t1 t2 t2-dump smoke clean lock diagnose fetch-data data-check data-patch

help:
	@uv run python -m tianximem.dev help

sync:
	uv run python -m tianximem.dev sync

lock:
	uv run python -m tianximem.dev lock

fmt:
	uv run python -m tianximem.dev format

lint: typecheck
	uv run python -m tianximem.dev lint

typecheck:
	uv run python -m tianximem.dev typecheck

test:
	uv run python -m tianximem.dev test $(ARGS)

serve:
	uv run python -m tianximem.dev serve

config-check:
	uv run python -m tianximem.dev config-check

check:
	uv run python -m tianximem.dev check $(ARGS)

data-check:
	uv run python -m tianximem.dev data-check $(DATASET_FLAG)

data-patch:
	uv run python -m tianximem.dev data-patch $(DATASET_FLAG)

fetch-data:
	uv run python -m tianximem.dev data-fetch $(DATASET_FLAG)

contract-check:
	uv run python -m tianximem.dev contract-check $(ARGS)

diagnose:
	uv run python -m tianximem.dev diagnose $(RUN_ARG) $(ARGS)

probe-reranker:
	uv run python -m tianximem.dev probe-reranker $(ARGS)

router-eval:
	uv run python -m tianximem.dev router-eval $(ROUTER_OUT_ARG) $(ARGS)

coq-eval:
	uv run python -m tianximem.dev coq-eval $(COQ_OUT_ARG) $(ARGS)

coq-decompose:
	uv run python -m tianximem.dev coq-decompose $(ARGS)

order-probe:
	uv run python -m tianximem.dev order-probe $(PROCESSES_ARG) $(ARGS)

export-multihop:
	uv run python -m tianximem.dev export-multihop $(ARGS)

eval:
	uv run python -m tianximem.dev eval --dataset $(DATASET) $(ARGS)

baseline:
	uv run python -m tianximem.dev baseline --dataset $(DATASET) $(ARGS)

replay-official:
	uv run python -m tianximem.dev replay-official $(ARGS)

t1:
	uv run python -m tianximem.dev t1 $(ARGS)

t2:
	uv run python -m tianximem.dev t2 $(ARGS)

t2-dump:
	uv run python -m tianximem.dev t2-dump $(ARGS)

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

deploy-check:
	uv run python -m tianximem.dev deploy-check $(ARGS)

smoke:
	uv run python -m tianximem.dev smoke

clean:
	uv run python -m tianximem.dev clean
