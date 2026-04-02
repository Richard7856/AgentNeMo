# Insurance Claims Agent — convenience commands
#
# WHY PYTHONPATH="." is needed:
# The project lives in an iCloud path with spaces ("Mobile Documents").
# Python's editable install .pth file doesn't process paths with spaces when
# running via uv run. Using PYTHONPATH="." adds the project root directly.
# See docs/DECISIONS.md for the full explanation.

VENV_NAT  = PYTHONPATH="." .venv/bin/nat
VENV_PY   = PYTHONPATH="." .venv/bin/python
ENV_CMD   = source .env && export NVIDIA_API_KEY

.PHONY: run serve phoenix trace validate ingest generate-pdfs infra-up infra-down infra-reset help

## run INPUT="your query" — run the agent with a single query
run:
	$(ENV_CMD) && $(VENV_NAT) run --config_file configs/config.yml --input "$(INPUT)"

## serve — start the REST API on port 8000
serve:
	$(ENV_CMD) && $(VENV_NAT) serve --config_file configs/config.yml --host 0.0.0.0 --port 8000

## phoenix — start the Phoenix tracing dashboard on http://localhost:6006
phoenix:
	PHOENIX_PORT=6006 $(VENV_PY) -m phoenix.server.main serve

## trace — run 3 demo queries with Phoenix tracing (start 'make phoenix' first in another terminal)
trace:
	$(ENV_CMD) && $(VENV_PY) trace_demo.py

## infra-up — start Milvus + etcd + MinIO stack (Docker required)
infra-up:
	docker compose up -d
	@echo "Waiting for Milvus to be healthy (can take ~60s on first run)..."
	@docker compose wait milvus 2>/dev/null || docker compose ps milvus

## infra-down — stop containers (data preserved in Docker volumes)
infra-down:
	docker compose down

## infra-reset — stop containers AND delete all vector data (full wipe)
infra-reset:
	docker compose down -v
	@echo "All Milvus data deleted. Run 'make infra-up && make ingest' to rebuild."

## validate — validate config.yml (no API key needed)
validate:
	.venv/bin/nat validate --config_file configs/config.yml

## ingest — embed policy PDFs and write vectors to Milvus (requires infra-up first)
ingest:
	$(ENV_CMD) && $(VENV_PY) insurance_claims/data_prep/ingest_policies.py

## generate-pdfs — regenerate the sample policy PDFs
generate-pdfs:
	$(VENV_PY) insurance_claims/data_prep/generate_policies.py

help:
	@grep -E '^## ' Makefile | sed 's/## /  make /'
