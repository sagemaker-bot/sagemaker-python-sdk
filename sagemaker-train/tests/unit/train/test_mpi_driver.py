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
"""Unit tests for MPI driver, mpi_utils, and common/utils modules."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from sagemaker.train.container_drivers.distributed_drivers.mpi_driver import MpiDriver
from sagemaker.train.container_drivers.distributed_drivers.mpi_utils import get_mpirun_command
from sagemaker.train.container_drivers.common.utils import get_process_count


class TestMpiDriverInitialization:
    """Tests for MpiDriver initialization."""

    def test_basic_initialization(self):
        """Test basic MpiDriver initialization with valid parameters."""
        driver = MpiDriver(
            host_count=2,
            host_list=["algo-1", "algo-2"],
            process_count_per_node=4,
        )
        assert driver._host_count == 2
        assert driver._host_list == ["algo-1", "algo-2"]
        assert driver._process_count_per_node == 4
        assert driver._additional_options == []

    def test_initialization_with_additional_options(self):
        """Test MpiDriver initialization with additional MPI options."""
        options = ["--mca", "plm_rsh_no_tree_spawn", "1"]
        driver = MpiDriver(
            host_count=1,
            host_list=["algo-1"],
            process_count_per_node=2,
            additional_options=options,
        )
        assert driver.additional_options == options

    def test_initialization_with_zero_process_count_defaults_to_auto(self):
        """Test that process_count_per_node=0 triggers auto-detection."""
        with patch.dict(os.environ, {"SM_NUM_GPUS": "8"}):
            driver = MpiDriver(
                host_count=1,
                host_list=["algo-1"],
                process_count_per_node=0,
            )
            assert driver._process_count_per_node == 8

    def test_initialization_invalid_host_count_zero(self):
        """Test that host_count=0 raises ValueError."""
        with pytest.raises(ValueError, match="host_count must be a positive integer"):
            MpiDriver(
                host_count=0,
                host_list=["algo-1"],
                process_count_per_node=1,
            )

    def test_initialization_invalid_host_count_negative(self):
        """Test that negative host_count raises ValueError."""
        with pytest.raises(ValueError, match="host_count must be a positive integer"):
            MpiDriver(
                host_count=-1,
                host_list=["algo-1"],
                process_count_per_node=1,
            )

    def test_initialization_empty_host_list(self):
        """Test that empty host_list raises ValueError."""
        with pytest.raises(ValueError, match="host_list must be a non-empty list"):
            MpiDriver(
                host_count=1,
                host_list=[],
                process_count_per_node=1,
            )

    def test_initialization_negative_process_count(self):
        """Test that negative process_count_per_node raises ValueError."""
        with pytest.raises(ValueError, match="process_count_per_node must be non-negative"):
            MpiDriver(
                host_count=1,
                host_list=["algo-1"],
                process_count_per_node=-1,
            )


class TestMpiDriverGetCommand:
    """Tests for MpiDriver.get_command output."""

    def test_get_command_basic(self):
        """Test basic get_command output."""
        driver = MpiDriver(
            host_count=2,
            host_list=["algo-1:4", "algo-2:4"],
            process_count_per_node=4,
        )
        command = driver.get_command(entry_script_path="/opt/ml/code/train.py")

        assert command[0] == "mpirun"
        assert "--allow-run-as-root" in command
        assert "-np" in command
        np_index = command.index("-np")
        assert command[np_index + 1] == "8"
        assert "--host" in command
        host_index = command.index("--host")
        assert command[host_index + 1] == "algo-1:4,algo-2:4"
        assert command[-1] == "/opt/ml/code/train.py"

    def test_get_command_with_additional_options(self):
        """Test get_command includes additional options."""
        driver = MpiDriver(
            host_count=1,
            host_list=["algo-1"],
            process_count_per_node=2,
            additional_options=["-x", "NCCL_DEBUG=INFO"],
        )
        command = driver.get_command(entry_script_path="train.py")
        assert "-x" in command
        assert "NCCL_DEBUG=INFO" in command

    def test_get_command_single_host(self):
        """Test get_command with single host."""
        driver = MpiDriver(
            host_count=1,
            host_list=["algo-1"],
            process_count_per_node=1,
        )
        command = driver.get_command(entry_script_path="train.py")
        np_index = command.index("-np")
        assert command[np_index + 1] == "1"

    def test_get_command_network_interface(self):
        """Test that network interface is included in command."""
        with patch.dict(os.environ, {"SM_NETWORK_INTERFACE_NAME": "ens5"}):
            driver = MpiDriver(
                host_count=1,
                host_list=["algo-1"],
                process_count_per_node=1,
            )
            command = driver.get_command(entry_script_path="train.py")
            assert "ens5" in command


class TestMpiDriverProperties:
    """Tests for MpiDriver properties."""

    def test_num_processes_total(self):
        """Test num_processes_total calculation."""
        driver = MpiDriver(
            host_count=4,
            host_list=["algo-1", "algo-2", "algo-3", "algo-4"],
            process_count_per_node=8,
        )
        assert driver.num_processes_total == 32

    def test_additional_options_default(self):
        """Test additional_options defaults to empty list."""
        driver = MpiDriver(
            host_count=1,
            host_list=["algo-1"],
            process_count_per_node=1,
        )
        assert driver.additional_options == []


class TestGetMpirunCommand:
    """Tests for get_mpirun_command utility function."""

    def test_basic_command(self):
        """Test basic mpirun command generation."""
        command = get_mpirun_command(
            host_list=["algo-1", "algo-2"],
            num_processes=4,
            additional_options=[],
            entry_script_path="train.py",
        )
        assert command[0] == "mpirun"
        assert "--allow-run-as-root" in command
        assert "-np" in command
        np_index = command.index("-np")
        assert command[np_index + 1] == "4"
        assert command[-1] == "train.py"

    def test_command_with_options(self):
        """Test mpirun command with additional options."""
        command = get_mpirun_command(
            host_list=["algo-1"],
            num_processes=2,
            additional_options=["-x", "PATH"],
            entry_script_path="train.py",
        )
        assert "-x" in command
        assert "PATH" in command

    def test_command_default_network_interface(self):
        """Test default network interface is eth0."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove SM_NETWORK_INTERFACE_NAME if set
            os.environ.pop("SM_NETWORK_INTERFACE_NAME", None)
            command = get_mpirun_command(
                host_list=["algo-1"],
                num_processes=1,
                additional_options=[],
                entry_script_path="train.py",
            )
            assert "eth0" in command


class TestGetProcessCount:
    """Tests for get_process_count utility function."""

    def test_explicit_process_count(self):
        """Test that explicit process count is returned directly."""
        assert get_process_count(4) == 4

    def test_auto_detect_gpus(self):
        """Test auto-detection from GPU count."""
        with patch.dict(os.environ, {"SM_NUM_GPUS": "8", "SM_NUM_NEURONS": "0"}):
            assert get_process_count(0) == 8

    def test_auto_detect_neurons(self):
        """Test auto-detection from Neuron device count."""
        with patch.dict(os.environ, {"SM_NUM_GPUS": "0", "SM_NUM_NEURONS": "16"}):
            assert get_process_count(0) == 16

    def test_fallback_to_one(self):
        """Test fallback to 1 when no accelerators available."""
        with patch.dict(os.environ, {"SM_NUM_GPUS": "0", "SM_NUM_NEURONS": "0"}):
            assert get_process_count(0) == 1

    def test_negative_process_count_raises(self):
        """Test that negative process count raises ValueError."""
        with pytest.raises(ValueError, match="process_count_per_node must be non-negative"):
            get_process_count(-1)

    def test_gpu_priority_over_neuron(self):
        """Test that GPU count takes priority over Neuron count."""
        with patch.dict(os.environ, {"SM_NUM_GPUS": "4", "SM_NUM_NEURONS": "16"}):
            assert get_process_count(0) == 4
