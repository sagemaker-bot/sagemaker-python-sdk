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
"""Regression test for issue #5904.

ModelTrainer generates ``sm_train.sh`` using ``open(path, "w")`` without an
explicit ``newline``. On Windows, text-mode writes translate ``\\n`` to
``\\r\\n`` (CRLF), which breaks the script when executed inside the Linux
training container.

This test simulates the Windows text-mode newline translation and asserts that
the generated ``sm_train.sh`` never contains carriage returns. It fails on the
unfixed code (which relies on the platform default) and passes once the SDK
forces LF line endings via ``newline="\\n"``.
"""
from __future__ import absolute_import

import builtins
import io
import os
import tempfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.constants import TRAIN_SCRIPT
from sagemaker.train.configs import SourceCode

from tests.unit import DATA_DIR

DEFAULT_SOURCE_DIR = f"{DATA_DIR}/script_mode"


class _WindowsNewlineFile(io.StringIO):
    """StringIO that flushes to disk translating LF -> CRLF like Windows text mode."""

    def __init__(self, path):
        super().__init__()
        self._path = path

    def close(self):
        contents = self.getvalue()
        with io.open(self._path, "wb") as raw:
            raw.write(contents.replace("\n", "\r\n").encode("utf-8"))
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def _windows_open(real_open):
    """Return an ``open`` replacement emulating Windows CRLF text-mode writes.

    Mirrors CPython's text-mode behaviour: ``\\n`` is translated to the platform
    line separator on write *unless* ``newline`` is passed explicitly. Only the
    default (no/None ``newline``) write path is emulated as Windows.
    """

    def _open(file, mode="r", *args, **kwargs):
        newline = kwargs.get("newline", None)
        if "w" in mode and "b" not in mode and newline is None:
            return _WindowsNewlineFile(file)
        return real_open(file, mode, *args, **kwargs)

    return _open


def test_sm_train_script_has_lf_line_endings_on_windows():
    trainer = ModelTrainer.__new__(ModelTrainer)

    source_code = SourceCode(
        source_dir=DEFAULT_SOURCE_DIR,
        entry_script="custom_script.py",
    )

    tmp_dir = TemporaryDirectory()
    try:
        with patch.object(builtins, "open", _windows_open(builtins.open)):
            trainer._prepare_train_script(tmp_dir=tmp_dir, source_code=source_code)

        script_path = os.path.join(tmp_dir.name, TRAIN_SCRIPT)
        with io.open(script_path, "rb") as f:
            raw = f.read()

        assert b"\r" not in raw, (
            "sm_train.sh was written with CRLF line endings; the Linux training "
            "container cannot execute it (issue #5904)."
        )
    finally:
        tmp_dir.cleanup()
