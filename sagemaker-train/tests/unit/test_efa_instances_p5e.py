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
"""Reproduction test for issue #5491: ml.p5e.48xlarge missing from EFA lists."""
from __future__ import absolute_import

from sagemaker.train.container_drivers.common.utils import (
    SM_EFA_NCCL_INSTANCES,
    SM_EFA_RDMA_INSTANCES,
)


def test_p5e_in_nccl_instances():
    assert "ml.p5e.48xlarge" in SM_EFA_NCCL_INSTANCES


def test_p5e_in_rdma_instances():
    assert "ml.p5e.48xlarge" in SM_EFA_RDMA_INSTANCES


def test_p5_in_rdma_instances():
    assert "ml.p5.48xlarge" in SM_EFA_RDMA_INSTANCES
