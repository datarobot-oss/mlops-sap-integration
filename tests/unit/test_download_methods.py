"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import os
from unittest.mock import mock_open, patch

import pytest

from environments.java_codegen_monitoring.download_model import ModelPackageUtils


class TestModelPackageUtils:

    @pytest.fixture()
    def env_vars(self, tmpdir):
        return {
            "DATAROBOT_ENDPOINT": "https://mock_endpoint.com",
            "DATAROBOT_API_TOKEN": "mock_token",
            "MLOPS_MODEL_PACKAGE_ID": "mock_package_id",
            "CODE_DIR": str(tmpdir),
            "TARGET_TYPE": "mock_target_type",
        }

    @pytest.fixture(autouse=True)
    def setup_env_vars(self, env_vars):
        patcher = patch.dict(os.environ, env_vars)
        patcher.start()
        yield
        patcher.stop()

    @pytest.fixture
    def mock_mlops_client(self):
        with patch(
            "environments.java_codegen_monitoring.download_model.MLOpsClient",
            autospec=True,
        ) as m:
            yield m

    def test_init_successful(self, mock_mlops_client, env_vars):
        instance = ModelPackageUtils()

        assert instance.model_package_id == env_vars["MLOPS_MODEL_PACKAGE_ID"]
        assert instance.output_dir == env_vars["CODE_DIR"]
        assert instance.target_type == env_vars["TARGET_TYPE"]
        mock_mlops_client.assert_called_once_with(
            env_vars["DATAROBOT_ENDPOINT"], env_vars["DATAROBOT_API_TOKEN"]
        )

    def test_init_missing_env_vars(self, env_vars):
        del os.environ["DATAROBOT_ENDPOINT"]

        with pytest.raises(ValueError, match="Missing environment variables:"):
            ModelPackageUtils()

    def test_write_class_names(self, mock_mlops_client, env_vars):
        mock_class_names = ["class1", "class2", "class3"]
        mock_file_path = f"{env_vars['CODE_DIR']}/classLabels.txt"

        with patch("builtins.open", mock_open()) as mocked_file:
            instance = ModelPackageUtils()
            instance._write_class_names(mock_class_names)

            mocked_file.assert_called_once_with(mock_file_path, mode="w")
            mocked_file.return_value.write.assert_called_once_with(
                "\n".join(mock_class_names)
            )

    def test_write_class_names_with_custom_file(self, mock_mlops_client):
        mock_class_names = ["class1", "class2", "class3"]
        custom_file_path = "/custom/path/classLabels.txt"

        with patch.dict(os.environ, {"CLASS_LABELS_FILE": custom_file_path}), patch(
            "builtins.open", mock_open()
        ) as mocked_file:
            instance = ModelPackageUtils()
            instance._write_class_names(mock_class_names)

            mocked_file.assert_called_once_with(custom_file_path, mode="w")
            mocked_file.return_value.write.assert_called_once_with(
                "\n".join(mock_class_names)
            )

    @pytest.mark.parametrize("target_type", ["multiclass", "binary"])
    def test_download_multiclass(self, target_type, mock_mlops_client, env_vars):
        with patch.dict(os.environ, {"TARGET_TYPE": target_type}):

            instance = ModelPackageUtils()
            instance.download()
            mock = mock_mlops_client.return_value

            if target_type == "multiclass":
                mock.get_model_package.assert_called_once_with(
                    env_vars["MLOPS_MODEL_PACKAGE_ID"]
                )
            else:
                mock.get_model_package.assert_not_called()

            mock.download_model_package_from_registry.assert_called_once_with(
                env_vars["MLOPS_MODEL_PACKAGE_ID"],
                env_vars["CODE_DIR"],
                download_scoring_code=True,
            )
