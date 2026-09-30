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
"""Reproduces issue #6278: LineIterator hangs forever if the streamed response
ends without a trailing newline."""
from __future__ import absolute_import

import pytest

from sagemaker.core.iterators import LineIterator


class _GuardedIterator:
    """Wraps an iterator, raising after being exhausted too many times.

    A well-behaved LineIterator will call next() on an exhausted upstream
    iterator at most a small, bounded number of times. The buggy version
    calls it forever (infinite loop), which this guard converts into a
    deterministic failure instead of a hang.
    """

    def __init__(self, chunks, max_exhausted_calls=5):
        self._it = iter(chunks)
        self._exhausted = False
        self._exhausted_calls = 0
        self._max_exhausted_calls = max_exhausted_calls

    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            self._exhausted_calls += 1
            if self._exhausted_calls > self._max_exhausted_calls:
                raise AssertionError(
                    "LineIterator kept polling an exhausted stream -- infinite loop"
                )
            raise StopIteration
        try:
            return next(self._it)
        except StopIteration:
            self._exhausted = True
            raise


def test_line_iterator_final_line_without_trailing_newline():
    chunks = [
        {"PayloadPart": {"Bytes": b'{"outputs": [" a"]}\n'}},
        {"PayloadPart": {"Bytes": b'{"outputs": [" b"]}'}},
    ]

    lines = list(LineIterator(_GuardedIterator(chunks)))

    assert lines == [b'{"outputs": [" a"]}', b'{"outputs": [" b"]}']


if __name__ == "__main__":
    pytest.main([__file__])
