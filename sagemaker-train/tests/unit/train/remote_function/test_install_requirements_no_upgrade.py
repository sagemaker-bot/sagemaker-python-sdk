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
"""Reproduction for issue #5872.

The @remote decorator bootstrap installs the requirements.txt with a hardcoded
``-U`` (upgrade) flag. This forces pip to upgrade an already-compatible
sagemaker install to the latest version in the loose range, introducing a
serialization/deserialization version mismatch (DeserializationError).

The container must install requirements WITHOUT forcing an upgrade of already
satisfied dependencies, so the pip command must not contain ``-U``.
"""
from __future__ import absolute_import

from unittest.mock import patch

from sagemaker.train.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentManager,
)


@patch(
    "sagemaker.train.remote_function.runtime_environment.runtime_environment_manager._run_shell_cmd"
)
def test_install_requirements_txt_does_not_force_upgrade(mock_run_cmd):
    """The pip install command must NOT contain the -U/--upgrade flag.

    Fails on the buggy code because ``_install_requirements_txt`` hardcodes
    ``-U``, which silently upgrades an already-compatible sagemaker install
    and breaks deserialization in the container.
    """
    manager = RuntimeEnvironmentManager()
    manager._install_requirements_txt("/path/to/requirements.txt", "/usr/bin/python")

    mock_run_cmd.assert_called_once()
    cmd = mock_run_cmd.call_args[0][0]

    assert "-U" not in cmd and "--upgrade" not in cmd, (
        f"pip install command must not force an upgrade, got: {cmd}"
    )
