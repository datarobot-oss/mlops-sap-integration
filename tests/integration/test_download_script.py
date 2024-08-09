"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import logging
import multiprocessing
import os
import subprocess
import tempfile
from http import HTTPStatus
from multiprocessing import Process
from pathlib import Path

import pytest
from bson import ObjectId
from flask import Flask, send_file
from future.moves.urllib import parse

multiprocessing.set_start_method("fork")


ENVS_DIR = Path(__file__).parent.parent.parent / "environments"
download_file = ENVS_DIR / "java_codegen_monitoring" / "download_model.py"
logger = logging.getLogger("datarobot.sap.integration")


@pytest.fixture(scope="session")
def class_names():
    return ["virginica", "versicolor", "setosa"]


@pytest.fixture(scope="session")
def mock_dr_app(class_names):
    base_url = "127.0.0.1"
    port = "33993"
    app = Flask(__name__)

    @app.route("/api/v2/version/", methods=["GET"])
    def get_version():
        version = {"major": 2, "minor": 35, "versionString": "2.35.0"}
        return version, HTTPStatus.OK

    @app.route("/api/v2/modelPackages/<model_package_id>/", methods=["GET"])
    def get_model_packages(model_package_id):
        return {
            "id": str(model_package_id),
            "name": "Multiclass Model",
            "modelId": "668c763b57e16e20e29e67e",
            "target": {
                "name": "Species",
                "type": "Multiclass",
                "classNames": class_names,
            },
        }

    @app.route(
        "/api/v2/modelPackages/<model_package_id>/scoringCodeBuilds/status",
        methods=["GET"],
    )
    def model_pacakge_builds_status(model_package_id):
        return {}, HTTPStatus.SEE_OTHER, {"Location": "download"}

    @app.route(
        "/api/v2/modelPackages/<model_package_id>/scoringCodeBuilds/download",
        methods=["GET"],
    )
    def model_pacakge_builds_download(model_package_id):
        model_file = tempfile.NamedTemporaryFile(delete=False)
        model_file.write(b"this is the content")
        model_file.close()
        return send_file(model_file.name, as_attachment=True, download_name="model.jar")

    @app.route(
        "/api/v2/modelPackages/<model_package_id>/scoringCodeBuilds/", methods=["POST"]
    )
    def model_pacakge_builds(model_package_id):
        return "", HTTPStatus.ACCEPTED, {"location": "status"}

    def start_app():
        app.run(host=base_url, port=port)

    server = Process(target=start_app)
    server.start()
    yield parse.urlunparse(("http", f"{base_url}:{port}", "", "", "", ""))

    logger.info("Shutting down mock server")
    server.terminate()


def test_script(mock_dr_app, class_names, tmpdir):
    env = os.environ.copy()
    extra_vars = {
        "MLOPS_MODEL_PACKAGE_ID": str(ObjectId()),
        "TARGET_TYPE": "multiclass",
        "CODE_DIR": tmpdir,
        "DATAROBOT_ENDPOINT": mock_dr_app,
        "DATAROBOT_API_TOKEN": "secret_token",
    }
    env.update(extra_vars)

    subprocess.run(["python", str(download_file)], check=True, env=env)
    model_file = tmpdir.join("model.jar")
    assert model_file.check(file=True)
    assert model_file.read()

    class_names_file = tmpdir.join("classLabels.txt")
    assert class_names_file.check(file=True)
    actual_class_names = class_names_file.read().splitlines()
    assert class_names == actual_class_names
