"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import os
import sys

from datarobot_mlops.connected.client import MLOpsClient


class ModelPackageUtils:

    REQUIRED_VARS = [
        "DATAROBOT_ENDPOINT",
        "DATAROBOT_API_TOKEN",
        "MLOPS_MODEL_PACKAGE_ID",
        "CODE_DIR",
        "TARGET_TYPE",
    ]

    def __init__(self):
        missing_vars = [v for v in self.REQUIRED_VARS if v not in os.environ]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {missing_vars}")

        self.model_package_id = os.environ["MLOPS_MODEL_PACKAGE_ID"]
        self.output_dir = os.environ["CODE_DIR"]
        self.target_type = os.environ["TARGET_TYPE"]

        endpoint = os.environ["DATAROBOT_ENDPOINT"]
        token = os.environ["DATAROBOT_API_TOKEN"]

        print(f"Connecting to: {endpoint}")
        self.client = MLOpsClient(endpoint, token)

    def _write_class_names(self, class_names):
        class_names_file = f"{self.output_dir}/classLabels.txt"
        class_names_file = os.environ.get("CLASS_LABELS_FILE", class_names_file)

        print(f"Writing class names: {class_names} to : {class_names_file}")
        with open(class_names_file, mode="w") as f:
            f.write("\n".join(class_names))

    def download(self):
        # When targetType == Multiclass, download the class names into a file
        if self.target_type.casefold() == "multiclass":
            model_package_details = self.client.get_model_package(self.model_package_id)
            class_names = model_package_details["target"]["classNames"]
            self._write_class_names(class_names)

        self.client.download_model_package_from_registry(
            self.model_package_id,
            self.output_dir,
            download_scoring_code=True,
        )


if __name__ == "__main__":
    try:
        ModelPackageUtils().download()
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
