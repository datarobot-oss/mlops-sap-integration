#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

set -e

echo "Starting Custom Model environment with DRUM prediction server"

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi

echo
echo "Downloading model info/jar/classes"
echo "Target type: ${TARGET_TYPE}"
echo

if [ "${TARGET_TYPE}" = "multiclass" ]; then
  export CLASS_LABELS_FILE="${CODE_DIR}/classLabels.txt"
fi
python3 download_model.py

echo
echo "Launching Nginx front-proxy (in the background)..."
nginx

echo
echo "Executing command: drum server $*"
echo
exec drum server "$@"
