# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Unit tests for MPI driver and utilities."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from sagemaker.train.container_drivers.distributed_drivers.mpi_driver import MPIDriver
from sagemaker.train.container_drivers.distributed_drivers.mpi_utils import (
    get_mpirun_command,
)


class TestGetMpirunCommand:
    """Tests for get_mpirun_command utility function."""

    def test_normal_command_construction(self):
        """Test normal mpirun command construction with valid inputs."""
        cmd = get_mpirun_command(
            host_count=2,
            host_list=["algo-1:1", "algo-2:1"],
            num_processes=2,
        )

        assert cmd[0] == "mpirun"
        assert "--host" in cmd
        host_idx = cmd.index("--host")
        assert cmd[host_idx + 1] == "algo-1:1,algo-2:1"
        assert "-np" in cmd
        np_idx = cmd.index("-np")
        assert cmd[np_idx + 1] == "2"
        assert "--allow-run-as-root" in cmd
        assert cmd[-1] == "train.py"

    def test_default_entry_script(self):
        """Test that default entry script is train.py."""
        cmd = get_mpirun_command(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
        )
        assert cmd[-1] == "train.py"

    def test_custom_entry_script(self):
        """Test custom entry script path."""
        cmd = get_mpirun_command(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
            entry_script_path="my_script.py",
        )
        assert cmd[-1] == "my_script.py"

    def test_additional_options(self):
        """Test that additional options are included in the command."""
        cmd = get_mpirun_command(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
            additional_options=["-x", "MY_VAR", "--oversubscribe"],
        )
        assert "-x" in cmd
        assert "MY_VAR" in cmd
        assert "--oversubscribe" in cmd

    @patch.dict("os.environ", {"SM_NETWORK_INTERFACE_NAME": "ens5"})
    def test_custom_network_interface(self):
        """Test that custom network interface from env var is used."""
        cmd = get_mpirun_command(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
        )
        assert "ens5" in cmd

    @patch.dict("os.environ", {"SM_SOURCE_DIR": "/custom/source"})
    def test_custom_source_dir(self):
        """Test that custom source dir from env var is used."""
        cmd = get_mpirun_command(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
        )
        wdir_idx = cmd.index("-wdir")
        assert cmd[wdir_idx + 1] == "/custom/source"

    def test_empty_host_list_raises_value_error(self):
        """Test that empty host list raises ValueError."""
        with pytest.raises(ValueError, match="host_list"):
            get_mpirun_command(
                host_count=0,
                host_list=[],
                num_processes=1,
            )

    def test_zero_num_processes_raises_value_error(self):
        """Test that zero num_processes raises ValueError."""
        with pytest.raises(ValueError, match="num_processes"):
            get_mpirun_command(
                host_count=1,
                host_list=["algo-1:1"],
                num_processes=0,
            )

    def test_negative_num_processes_raises_value_error(self):
        """Test that negative num_processes raises ValueError."""
        with pytest.raises(ValueError, match="num_processes"):
            get_mpirun_command(
                host_count=1,
                host_list=["algo-1:1"],
                num_processes=-1,
            )

    def test_zero_host_count_raises_value_error(self):
        """Test that zero host_count raises ValueError."""
        with pytest.raises(ValueError, match="host_count"):
            get_mpirun_command(
                host_count=0,
                host_list=["algo-1:1"],
                num_processes=1,
            )

    def test_negative_host_count_raises_value_error(self):
        """Test that negative host_count raises ValueError."""
        with pytest.raises(ValueError, match="host_count"):
            get_mpirun_command(
                host_count=-1,
                host_list=["algo-1:1"],
                num_processes=1,
            )

    def test_host_count_mismatch_raises_value_error(self):
        """Test that host_count not matching host_list length raises ValueError."""
        with pytest.raises(ValueError, match="host_count"):
            get_mpirun_command(
                host_count=3,
                host_list=["algo-1:1", "algo-2:1"],
                num_processes=2,
            )


class TestMPIDriver:
    """Tests for MPIDriver class."""

    def test_init_valid_params(self):
        """Test MPIDriver initialization with valid parameters."""
        driver = MPIDriver(
            host_count=2,
            host_list=["algo-1:1", "algo-2:1"],
            num_processes=4,
            entry_script_path="my_train.py",
            additional_options=["--oversubscribe"],
        )
        assert driver.host_count == 2
        assert driver.host_list == ["algo-1:1", "algo-2:1"]
        assert driver.num_processes == 4
        assert driver.entry_script_path == "my_train.py"
        assert driver.additional_options == ["--oversubscribe"]

    def test_init_default_values(self):
        """Test MPIDriver initialization with default values."""
        driver = MPIDriver(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
        )
        assert driver.entry_script_path == "train.py"
        assert driver.additional_options == []

    def test_init_empty_host_list_raises_value_error(self):
        """Test that empty host_list raises ValueError."""
        with pytest.raises(ValueError, match="host_list"):
            MPIDriver(
                host_count=0,
                host_list=[],
                num_processes=1,
            )

    def test_init_negative_host_count_raises_value_error(self):
        """Test that negative host_count raises ValueError."""
        with pytest.raises(ValueError, match="host_count"):
            MPIDriver(
                host_count=-1,
                host_list=["algo-1:1"],
                num_processes=1,
            )

    def test_init_zero_num_processes_raises_value_error(self):
        """Test that zero num_processes raises ValueError."""
        with pytest.raises(ValueError, match="num_processes"):
            MPIDriver(
                host_count=1,
                host_list=["algo-1:1"],
                num_processes=0,
            )

    def test_init_negative_num_processes_raises_value_error(self):
        """Test that negative num_processes raises ValueError."""
        with pytest.raises(ValueError, match="num_processes"):
            MPIDriver(
                host_count=1,
                host_list=["algo-1:1"],
                num_processes=-1,
            )

    @patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.subprocess.Popen")
    def test_run_success(self, mock_popen):
        """Test successful run() execution."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["Training started\n", "Training complete\n"])
        mock_process.returncode = 0
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        driver = MPIDriver(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
        )
        return_code = driver.run()

        assert return_code == 0
        mock_popen.assert_called_once()

    @patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.subprocess.Popen")
    def test_run_failure(self, mock_popen):
        """Test run() with non-zero return code."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.returncode = 1
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        driver = MPIDriver(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
        )
        return_code = driver.run()

        assert return_code == 1

    @patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.subprocess.Popen")
    def test_run_mpirun_not_found(self, mock_popen):
        """Test run() when mpirun is not found on the system."""
        mock_popen.side_effect = FileNotFoundError("No such file or directory: 'mpirun'")

        driver = MPIDriver(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
        )

        with pytest.raises(RuntimeError, match="Failed to execute mpirun"):
            driver.run()

    @patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.subprocess.Popen")
    def test_run_os_error(self, mock_popen):
        """Test run() when an OSError occurs."""
        mock_popen.side_effect = OSError("Permission denied")

        driver = MPIDriver(
            host_count=1,
            host_list=["algo-1:1"],
            num_processes=1,
        )

        with pytest.raises(RuntimeError, match="Failed to execute mpirun"):
            driver.run()
