"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import logging
import multiprocessing
from http import HTTPStatus
from multiprocessing import Process
from pathlib import Path

import pytest
from flask import Flask, request, send_file

multiprocessing.set_start_method("fork")


TEST_DIR = Path(__file__).parent.parent
logger = logging.getLogger("datarobot.sap.integration")


@pytest.fixture(scope="session")
def model_packages():
    return {
        "iris_multiclass_package": {
            "details": {
                "id": "iris_multiclass_package",
                "name": "Multiclass Model",
                "modelId": "668c763b57e16e20e29e67e",
                "target": {
                    "name": "Species",
                    "type": "multiclass",
                    "classNames": ["viginica c g", "versicolor", "setosa color"],
                },
            },
            "model_dataset": TEST_DIR / "data" / "mlops-example-iris-samples.csv",
            "model_jar": TEST_DIR / "data" / "ScoringCodeIrisSamples.jar",
        },
        "10k_diabetes_package": {
            "details": {
                "id": "10k_diabetes_package",
                "name": "Binary Model",
                "modelId": "668c763b57e16e20e29e67e",
                "target": {
                    "name": "readmitted",
                    "type": "binary",
                    "classNames": ["True", "False"],
                },
            },
            "model_dataset": TEST_DIR / "data" / "mlops-example-10k-diabetes.csv",
            "model_jar": TEST_DIR / "data" / "CodeGen10KDiabetes.jar",
        },
    }


@pytest.fixture(scope="session")
def mock_dr_app_port():
    return 33993


@pytest.fixture
def mock_dr_app(model_packages, mock_dr_app_port):
    app = Flask(__name__)

    @app.route("/api/v2/version/", methods=["GET"])
    def get_version():
        version = {"major": 2, "minor": 35, "versionString": "2.35.0"}
        return version, HTTPStatus.OK

    @app.route("/api/v2/modelPackages/<model_package_id>/", methods=["GET"])
    def get_model_packages(model_package_id):
        return model_packages.get(model_package_id, {}).get("details")

    @app.route(
        "/api/v2/status/<model_package_id>/",
        methods=["GET"],
    )
    def model_pacakge_builds_status(model_package_id):
        download_url = (
            f"{request.host_url}api/v2/modelPackages/{model_package_id}/download/"
        )
        return {}, HTTPStatus.SEE_OTHER, {"Location": download_url}

    @app.route(
        "/api/v2/modelPackages/<model_package_id>/download/",
        methods=["GET"],
    )
    def model_pacakge_builds_download(model_package_id):
        model_jar_filename = model_packages.get(model_package_id, {}).get("model_jar")
        return send_file(
            str(model_jar_filename), as_attachment=True, download_name="model.jar"
        )

    @app.route(
        "/api/v2/modelPackages/<model_package_id>/scoringCodeBuilds/", methods=["POST"]
    )
    def model_pacakge_builds(model_package_id):
        status_url = f"{request.host_url}api/v2/status/{model_package_id}/"
        return "", HTTPStatus.ACCEPTED, {"location": status_url}

    def start_app():
        app.run(host="0.0.0.0", port=mock_dr_app_port)

    server = Process(target=start_app)
    server.start()

    yield

    logger.info("Shutting down mock server")
    server.terminate()
