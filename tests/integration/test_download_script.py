"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import logging
import os
import subprocess
from pathlib import Path


ENVS_DIR = Path(__file__).parent.parent.parent / "environments"
download_file = ENVS_DIR / "java_codegen_monitoring" / "download_model.py"

logger = logging.getLogger("datarobot.sap.integration")


def test_script(mock_dr_app, mock_dr_app_port, model_packages, tmpdir):
    model_package_id = "iris_multiclass_package"
    model_package = model_packages[model_package_id]["details"]

    env = os.environ.copy()
    extra_vars = {
        "MLOPS_MODEL_PACKAGE_ID": model_package_id,
        "TARGET_TYPE": model_package["target"]["type"],
        "CODE_DIR": tmpdir,
        "DATAROBOT_ENDPOINT":  f"http://localhost:{mock_dr_app_port}",
        "DATAROBOT_API_TOKEN": "secret_token",
    }
    env.update(extra_vars)

    subprocess.run(["python", str(download_file)], check=True, env=env)
    model_file = tmpdir.join("model.jar")
    assert model_file.check(file=True)

    class_names_file = tmpdir.join("classLabels.txt")
    assert class_names_file.check(file=True)
    actual_class_names = class_names_file.read().splitlines()
    class_names = model_package["target"]["classNames"]
    assert class_names == actual_class_names
