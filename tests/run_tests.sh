#!/bin/bash

set -e

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT"
pwd

export PYTHONPATH=$GIT_ROOT

echo "Running unit tests"
python -m pytest -v ./tests/unit/test_download_methods.py

echo "Running integration tests"
python -m pytest -v ./tests/integration/test_download_script.py
