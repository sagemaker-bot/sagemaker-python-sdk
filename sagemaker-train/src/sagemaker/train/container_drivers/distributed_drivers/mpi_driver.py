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
"""MPI driver for distributed training on SageMaker.

This module provides the MPI driver implementation that handles
MPI-based distributed training configuration and execution.
"""
from __future__ import absolute_import

import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# MPI-related hyperparameter keys
MP_PROCESSES_PER_HOST = "sagemaker_mpi_num_of_processes_per_host"
MP_CUSTOM_MPI_OPTIONS = "sagemaker_mpi_custom_mpi_options"
MP_ENABLED = "sagemaker_mpi_enabled"

# Default MPI options
DEFAULT_MPI_OPTIONS = "--allow-run-as-root"

# Environment variable keys
ENV_MASTER_ADDR = "MASTER_ADDR"
ENV_MASTER_PORT = "MASTER_PORT"
ENV_NETWORK_INTERFACE = "NCCL_SOCKET_IFNAME"


class MPIDriver:
    """Driver for MPI-based distributed training.

    This driver handles the configuration and execution of MPI-based
    distributed training jobs on SageMaker, including setting up the
    MPI environment, managing host communication, and launching the
    training script with mpirun.
    """

    def __init__(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        hosts: Optional[List[str]] = None,
        current_host: Optional[str] = None,
        network_interface: str = "eth0",
    ):
        """Initialize the MPI driver.

        Args:
            hyperparameters: Dictionary of hyperparameters for the training job.
                Expected to contain MPI-specific parameters such as
                sagemaker_mpi_num_of_processes_per_host and
                sagemaker_mpi_custom_mpi_options.
            hosts: List of hostnames participating in the distributed training.
            current_host: The hostname of the current instance.
            network_interface: The network interface to use for communication.
                Defaults to "eth0".
        """
        self._hyperparameters = hyperparameters or {}
        self._hosts = hosts or []
        self._current_host = current_host or ""
        self._network_interface = network_interface
        self._processes_per_host = self._get_processes_per_host()
        self._custom_mpi_options = self._get_custom_mpi_options()

    def _get_processes_per_host(self) -> int:
        """Get the number of MPI processes per host from hyperparameters.

        Returns:
            int: Number of processes per host. Defaults to 1 if not specified.
        """
        return int(self._hyperparameters.get(MP_PROCESSES_PER_HOST, 1))

    def _get_custom_mpi_options(self) -> str:
        """Get custom MPI options from hyperparameters.

        Returns:
            str: Custom MPI options string. Defaults to empty string if not specified.
        """
        return str(self._hyperparameters.get(MP_CUSTOM_MPI_OPTIONS, ""))

    @property
    def is_master(self) -> bool:
        """Check if the current host is the master node.

        Returns:
            bool: True if the current host is the first host in the list.
        """
        if not self._hosts:
            return True
        return self._current_host == self._hosts[0]

    @property
    def num_hosts(self) -> int:
        """Get the total number of hosts.

        Returns:
            int: Number of hosts in the training cluster.
        """
        return len(self._hosts) if self._hosts else 1

    @property
    def total_processes(self) -> int:
        """Get the total number of MPI processes across all hosts.

        Returns:
            int: Total number of processes (hosts * processes_per_host).
        """
        return self.num_hosts * self._processes_per_host

    def build_mpi_command(
        self,
        script: str,
        script_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the mpirun command for launching distributed training.

        Args:
            script: Path to the training script to execute.
            script_args: Optional list of arguments to pass to the training script.

        Returns:
            List[str]: The complete mpirun command as a list of strings.
        """
        mpi_command = [
            "mpirun",
            DEFAULT_MPI_OPTIONS,
            "-np",
            str(self.total_processes),
            "--npernode",
            str(self._processes_per_host),
        ]

        # Add host list
        if self._hosts:
            host_list = ",".join(
                [f"{host}:{self._processes_per_host}" for host in self._hosts]
            )
            mpi_command.extend(["-H", host_list])

        # Add network interface configuration
        mpi_command.extend(
            ["-mca", "btl_tcp_if_include", self._network_interface]
        )

        # Add custom MPI options
        if self._custom_mpi_options:
            mpi_command.extend(self._custom_mpi_options.split())

        # Add the script and its arguments
        mpi_command.append(script)
        if script_args:
            mpi_command.extend(script_args)

        return mpi_command

    def setup_environment(self) -> Dict[str, str]:
        """Set up environment variables for MPI distributed training.

        Returns:
            Dict[str, str]: Dictionary of environment variables to set.
        """
        env = {}

        if self._hosts:
            env[ENV_MASTER_ADDR] = self._hosts[0]
            env[ENV_MASTER_PORT] = "7777"

        env[ENV_NETWORK_INTERFACE] = self._network_interface

        return env

    def run(
        self,
        script: str,
        script_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        """Execute the MPI training job.

        This method builds and runs the mpirun command on the master node.
        Worker nodes will wait for the master to initiate the MPI job.

        Args:
            script: Path to the training script to execute.
            script_args: Optional list of arguments to pass to the training script.
            env: Optional dictionary of additional environment variables.

        Returns:
            int: The return code of the mpirun process.
        """
        if not self.is_master:
            logger.info(
                "Current host %s is not master. Waiting for MPI job from master.",
                self._current_host,
            )
            return 0

        # Set up environment
        run_env = os.environ.copy()
        run_env.update(self.setup_environment())
        if env:
            run_env.update(env)

        # Build and execute the MPI command
        command = self.build_mpi_command(script, script_args)
        logger.info("Running MPI command: %s", " ".join(command))

        try:
            process = subprocess.Popen(
                command,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Stream output
            if process.stdout:
                for line in iter(process.stdout.readline, b""):
                    sys.stdout.write(line.decode("utf-8"))

            process.wait()
            return process.returncode

        except Exception as e:
            logger.error("Error running MPI command: %s", str(e))
            raise


def is_mpi_enabled(hyperparameters: Dict[str, Any]) -> bool:
    """Check if MPI distributed training is enabled.

    Args:
        hyperparameters: Dictionary of hyperparameters for the training job.

    Returns:
        bool: True if MPI is enabled in the hyperparameters.
    """
    enabled = hyperparameters.get(MP_ENABLED, "false")
    if isinstance(enabled, bool):
        return enabled
    return str(enabled).lower() == "true"
