#!/bin/bash
# nat-serve.sh — wrapper for preview_start compatibility.
#
# WHY this wrapper exists:
# The project lives in an iCloud path with spaces ("Mobile Documents").
# preview_start launches processes without PYTHONPATH set, so the editable
# install can't find the insurance_claims package.
# Setting PYTHONPATH to the absolute project dir fixes the import before nat runs.

PROJECT_DIR="/Users/richardfigueroa/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/insurance-claims-agent"

cd "$PROJECT_DIR" && \
  source .env && \
  export PYTHONPATH="$PROJECT_DIR" && \
  exec "$PROJECT_DIR/.venv/bin/python" \
    "$PROJECT_DIR/.venv/bin/nat" \
    serve \
    --config_file "$PROJECT_DIR/configs/config.yml" \
    --host 0.0.0.0 \
    --port 8000
