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
"""Unit tests for MPI driver hyperparameter contract configuration."""
from __future__ import absolute_import

import pytest

from sagemaker.modules.train.container_drivers.mpi_driver import (
    MPIDriverConfig,
)


class TestMPIDriverConfig:
    """Tests for MPI driver configuration and hyperparameter contracts."""

    def test_mpi_driver_config_defaults(self):
        """Test that MPIDriverConfig has correct default values."""
        config = MPIDriverConfig()
        assert config.processes_per_host == 1
        assert config.custom_mpi_options == ""

    def test_mpi_driver_config_custom_values(self):
        """Test MPIDriverConfig with custom values."""
        config = MPIDriverConfig(
            processes_per_host=4,
            custom_mpi_options="--mca btl_vader_single_copy_mechanism none",
        )
        assert config.processes_per_host == 4
        assert config.custom_mpi_options == "--mca btl_vader_single_copy_mechanism none"

    def test_mpi_driver_config_processes_per_host_validation(self):
        """Test that processes_per_host must be a positive integer."""
        with pytest.raises((ValueError, TypeError)):
            MPIDriverConfig(processes_per_host=0)

        with pytest.raises((ValueError, TypeError)):
            MPIDriverConfig(processes_per_host=-1)
