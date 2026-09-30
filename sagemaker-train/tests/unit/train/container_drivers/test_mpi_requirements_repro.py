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
"""Reproduction for issue #5886 - MPI distributed training does not resolve
the entry script / requirements.txt on worker nodes.

The ``environment.py`` script sets ``SM_ENTRY_SCRIPT`` to the *relative*
entry-script name (e.g. ``"train.py"``) - see
tests/unit/train/container_drivers/scripts/test_enviornment.py which asserts
``export SM_ENTRY_SCRIPT='train.py'``.

The MPI driver hands that value straight to ``mpirun``. On the master node this
happens to work because the training shell already did ``cd
/opt/ml/input/data/code``. But ``mpirun`` launches the training processes on the
*worker* nodes over a fresh SSH login whose working directory is the login
(home) directory - NOT ``/opt/ml/input/data/code``. A relative path such as
``train.py`` therefore cannot be found on the workers, which is why the
``test_hp_contract_mpi_script`` integ test was skipped with the note that
"MPI distributed training does not resolve requirements.txt on worker nodes".

For the command to work on every node, the entry-script argument passed to
``mpirun`` must be an absolute container path
(``/opt/ml/input/data/code/train.py``).
"""
from __future__ import absolute_import

import os
import sys
import json

from unittest.mock import patch, MagicMock

sys.modules["utils"] = MagicMock()
sys.modules["mpi_utils"] = MagicMock()

from sagemaker.train.container_drivers.distributed_drivers import mpi_driver  # noqa: E402

# environment.py writes SM_ENTRY_SCRIPT as the bare, relative entry-script name.
RELATIVE_ENTRY_SCRIPT = "train.py"

DUMMY_DISTRIBUTED = {
    "process_count_per_node": 2,
    "mpi_additional_options": [],
}


@patch.dict(
    os.environ,
    {
        "SM_CURRENT_HOST": "algo-1",
        "SM_HOSTS": '["algo-1", "algo-2"]',
        "SM_MASTER_ADDR": "algo-1",
        "SM_HOST_COUNT": "2",
        "SM_CURRENT_INSTANCE_TYPE": "ml.m5.xlarge",
        "SM_HPS": json.dumps({}),
        "SM_DISTRIBUTED_CONFIG": json.dumps(DUMMY_DISTRIBUTED),
        # As produced by environment.py, this is a *relative* path.
        "SM_ENTRY_SCRIPT": RELATIVE_ENTRY_SCRIPT,
    },
)
@patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.write_env_vars_to_file")
@patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.start_sshd_daemon")
@patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.bootstrap_master_node")
@patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.bootstrap_worker_node")
@patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.get_process_count")
@patch(
    "sagemaker.train.container_drivers.distributed_drivers.mpi_driver.hyperparameters_to_cli_args"
)
@patch("sagemaker.train.container_drivers.distributed_drivers.mpi_driver.execute_commands")
@patch(
    "sagemaker.train.container_drivers.distributed_drivers.mpi_driver.write_status_file_to_workers"
)
def test_mpi_entry_script_is_absolute_for_workers(
    mock_write_status_file_to_workers,
    mock_execute_commands,
    mock_hyperparameters_to_cli_args,
    mock_get_process_count,
    mock_bootstrap_worker_node,
    mock_bootstrap_master_node,
    mock_start_sshd_daemon,
    mock_write_env_vars_to_file,
):
    mock_hyperparameters_to_cli_args.return_value = []
    mock_get_process_count.return_value = 2
    mock_execute_commands.return_value = (0, "")

    # Run the real driver (get_mpirun_command is NOT mocked) to build the command.
    mpi_driver.main()

    mock_execute_commands.assert_called_once()
    mpi_command = mock_execute_commands.call_args[0][0]

    # The entry script argument is the token that follows "mpi4py".
    assert "mpi4py" in mpi_command, f"Unexpected mpirun command: {mpi_command}"
    entry_arg = mpi_command[mpi_command.index("mpi4py") + 1]

    # BUG: the driver forwards the relative path, which mpirun cannot resolve on
    # worker nodes (their SSH cwd is the home dir, not /opt/ml/input/data/code).
    assert os.path.isabs(entry_arg), (
        "mpirun entry-script path must be absolute so it resolves on worker "
        f"nodes, but got a relative path: {entry_arg!r}"
    )
