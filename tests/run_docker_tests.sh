#!/bin/bash

set -e

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT"
pwd

export PYTHONPATH=$GIT_ROOT

echo "Running docker tests"
python -m pytest -v ./tests/integration/test_make_predictions.py
