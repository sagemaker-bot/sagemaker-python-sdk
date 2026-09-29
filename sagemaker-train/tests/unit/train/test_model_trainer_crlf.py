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
"""Regression test for issue #5897.

On a Windows host, ``sm_train.sh`` is written with CRLF line endings because
``_prepare_train_script`` opens the file in text mode without ``newline="\\n"``,
which triggers platform newline translation (``\\n`` -> ``\\r\\n``). Inside the
Linux training container bash fails to parse the script (``$'\\r': command not
found``). The script must be written with LF-only endings regardless of host OS.
"""
from __future__ import absolute_import

import io
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.constants import TRAIN_SCRIPT
from sagemaker.train.configs import Compute, SourceCode, OutputDataConfig

from tests.unit import DATA_DIR

DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_REGION = "us-west-2"
DEFAULT_SOURCE_DIR = f"{DATA_DIR}/script_mode"


@pytest.fixture(autouse=True)
def modules_session():
    with patch("sagemaker.train.Session", spec=Session) as session_mock, patch(
        "sagemaker.train.defaults.resolve_and_validate_role", return_value=DEFAULT_ROLE
    ):
        session_instance = session_mock.return_value
        session_instance.default_bucket.return_value = DEFAULT_BUCKET
        session_instance.get_caller_identity_arn.return_value = DEFAULT_ROLE
        session_instance.default_bucket_prefix = None
        session_instance.boto_session = MagicMock(spec="boto3.session.Session")
        session_instance.boto_region_name = DEFAULT_REGION
        yield session_instance


class _WindowsTextFile(io.StringIO):
    """A StringIO that emulates Windows text-mode newline translation on write.

    When a text file is opened without an explicit ``newline`` argument, Python's
    default behavior translates every ``\\n`` to ``os.linesep`` on write. On
    Windows this is ``\\r\\n``. When ``newline=""`` or ``newline="\\n"`` is passed,
    no translation is performed. This faithfully reproduces the reported
    Windows-only behavior on any host.
    """

    def __init__(self, target_path, newline):
        super().__init__()
        self._target_path = target_path
        self._newline = newline

    def write(self, s):
        # Default text mode (newline unset) translates \n -> \r\n on Windows.
        # An explicit newline of "" or "\n" disables translation.
        if self._newline is None:
            s = s.replace("\n", "\r\n")
        return super().write(s)

    def close(self):
        data = self.getvalue()
        with io.open(self._target_path, "w", newline="") as real:
            real.write(data)
        super().close()


def test_sm_train_script_written_with_lf_on_windows_host(tmp_path, modules_session):
    """sm_train.sh must contain no CRLF, even when the host translates newlines."""

    import builtins

    real_open = builtins.open

    def fake_open(file, mode="r", *args, newline=None, **kwargs):
        # Only intercept the sm_train.sh text-mode write; delegate everything else.
        if "w" in mode and str(file).endswith(TRAIN_SCRIPT):
            return _WindowsTextFile(file, newline)
        return real_open(file, mode, *args, newline=newline, **kwargs)

    trainer = ModelTrainer(
        sagemaker_session=modules_session,
        training_image=DEFAULT_IMAGE,
        role=DEFAULT_ROLE,
        compute=Compute(instance_type="ml.m5.xlarge", instance_count=1),
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{DEFAULT_BUCKET}/output",
        ),
        source_code=SourceCode(
            source_dir=DEFAULT_SOURCE_DIR,
            entry_script="custom_script.py",
        ),
    )

    tmp_dir = tempfile.TemporaryDirectory(dir=str(tmp_path))
    try:
        with patch("builtins.open", side_effect=fake_open):
            trainer._prepare_train_script(
                tmp_dir=tmp_dir,
                source_code=trainer.source_code,
                distributed=None,
            )

        script_path = os.path.join(tmp_dir.name, TRAIN_SCRIPT)
        with io.open(script_path, "rb") as f:
            content = f.read()

        assert b"\r" not in content, (
            "sm_train.sh was written with CRLF line endings; bash in the Linux "
            "training container cannot parse it. The file must be opened with "
            'newline="\\n".'
        )
    finally:
        tmp_dir.cleanup()
