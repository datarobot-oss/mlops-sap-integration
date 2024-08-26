"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import contextlib
import logging
import time
from io import BytesIO

import pandas as pd
import pytest
import requests
from docker import from_env as docker_from_env

logger = logging.getLogger("datarobot.sap.integration")

SAP_MLOPS_INTEGRATION_IMAGE = (
    "ghcr.io/datarobot-oss/mlops-sap-monitoring-scoring-code:latest"
)
MAX_ROWS_PER_REQUEST = 1000


def _wait_for_container_healthy(container, ping_url, timeout=100, interval=2):
    """Wait `timeout` seconds for `container` become healthy."""
    end_time = time.monotonic() + timeout
    while time.monotonic() < end_time:
        try:
            response = requests.get(ping_url)
            if response.ok:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(interval)

    raise RuntimeError(container.logs())


@pytest.fixture
def run_sap_integration_image():
    """Run a Docker container with mlops SAP image"""

    @contextlib.contextmanager
    def runner(
        envvars,
        ping_url,
        network_mode="bridge",
        wait_healthy_timeout=200,
    ):
        docker_client = docker_from_env()
        container = docker_client.containers.run(
            SAP_MLOPS_INTEGRATION_IMAGE,
            ports={"9001/tcp": ("127.0.0.1", "9001")},
            environment=envvars,
            extra_hosts={"host.docker.internal": "host-gateway"},
            network_mode=network_mode,
            detach=True,
        )

        try:
            _wait_for_container_healthy(
                container, ping_url, timeout=wait_healthy_timeout
            )
            yield container
        finally:
            container.stop()
            container.remove()

    return runner


@pytest.mark.parametrize(
    "model_package_id, target_type",
    [
        pytest.param("iris_multiclass_package", "multiclass", id="multiclass"),
        pytest.param("10k_diabetes_package", "binary", id="binary"),
    ],
)
def test_run_server(
    mock_dr_app,
    mock_dr_app_port,
    run_sap_integration_image,
    model_package_id,
    target_type,
    model_packages,
):
    env_vars = {
        "MLOPS_MODEL_PACKAGE_ID": model_package_id,
        "TARGET_TYPE": target_type,
        "DATAROBOT_ENDPOINT": f"http://host.docker.internal:{mock_dr_app_port}",
        "DATAROBOT_API_TOKEN": "secret_token",
    }
    if target_type == "binary":
        env_vars["POSITIVE_CLASS_LABEL"] = "True"
        env_vars["NEGATIVE_CLASS_LABEL"] = "False"

    base_url = "http://localhost:9001"
    with run_sap_integration_image(env_vars, f"{base_url}/v1/ping/"):
        dataset = model_packages[model_package_id].get("model_dataset")
        data_chunks = pd.read_csv(dataset, chunksize=MAX_ROWS_PER_REQUEST)

        for chunk in data_chunks:
            # Convert the chunk to a CSV byte stream
            csv_chunk = BytesIO()
            chunk.to_csv(csv_chunk, index=False)
            csv_chunk.seek(0)

            # Make predictions
            response = requests.post(
                f"{base_url}/v1/predict/",
                data=csv_chunk,
            )
            assert response.status_code == 200

            # assert number of predictions
            predictions = response.json().get("predictions", [])
            assert len(predictions) == len(chunk)


def test_run_server_large_request(
    mock_dr_app,
    mock_dr_app_port,
    run_sap_integration_image,
    model_packages,
):
    env_vars = {
        "MLOPS_MODEL_PACKAGE_ID": "10k_diabetes_package",
        "TARGET_TYPE": "binary",
        "DATAROBOT_ENDPOINT": f"http://host.docker.internal:{mock_dr_app_port}",
        "DATAROBOT_API_TOKEN": "secret_token",
        "POSITIVE_CLASS_LABEL": "True",
        "NEGATIVE_CLASS_LABEL": "False",
    }

    base_url = "http://localhost:9001"
    with run_sap_integration_image(env_vars, f"{base_url}/v1/ping/"):
        # Make large number of predictions in a single request
        dataset = model_packages["10k_diabetes_package"].get("model_dataset")
        response = requests.post(
            f"{base_url}/v1/predict/", data=dataset.open(mode="rb")
        )
        assert response.status_code == 413
