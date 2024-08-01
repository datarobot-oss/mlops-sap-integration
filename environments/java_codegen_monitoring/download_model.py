"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import os
import sys

from datarobot_mlops.connected.client import MLOpsClient


model_package_id = os.environ.get("MLOPS_MODEL_PACKAGE_ID")
if not model_package_id:
    print("Missing model package id, env var MLOPS_MODEL_PACKAGE_ID not provided", file=sys.stderr)
    raise SystemExit(1)

output_dir = os.environ.get("CODE_DIR")
if not output_dir:
    print("Missing output dir, env var CODE_DIR not provided", file=sys.stderr)
    raise SystemExit(1)

target_type = os.environ.get("TARGET_TYPE")
if not target_type:
    print("Missing target type, env var TARGET_TYPE not provided", file=sys.stderr)
    raise SystemExit(1)

endpoint = os.environ["DATAROBOT_ENDPOINT"]
token = os.environ["DATAROBOT_API_TOKEN"]

print(f"Connecting to: {endpoint}")
client = MLOpsClient(endpoint, token)

model_package_details = client.get_model_package(model_package_id)
if not model_package_details:
    print("Fail to extract model package details", file=sys.stderr)
    raise SystemExit(1)

print(f"Downloading model jar from: {endpoint}")
client.download_model_package_from_registry(
    model_package_id,
    output_dir,
    download_scoring_code=True,
)

# When targetType == Multiclass, download the class names into a file
if "multiclass" == target_type:
    class_names_file = f"{output_dir}/classLabels.txt"
    class_names_file = os.environ.get("CLASS_LABELS_FILE", class_names_file)

    class_names = model_package_details["target"]["classNames"]
    print(f"Writing class names: {class_names} to : {class_names_file}")
    with open(class_names_file, mode="w") as f:
        f.write("\n".join(class_names))
