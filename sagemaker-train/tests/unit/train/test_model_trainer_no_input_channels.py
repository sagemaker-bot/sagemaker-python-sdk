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
"""Reproduction for GH#6156.

A ModelTrainer with no input channels must NOT emit ``InputDataConfig: []``
into the serialized (boto3/pipeline) training request. The SageMaker API
rejects an empty ``InputDataConfig`` (its botocore shape has min=1) but
accepts an absent one, so the key must be omitted when there are no channels.
"""
from __future__ import absolute_import

from unittest.mock import patch, MagicMock

import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import Compute, StoppingCondition, OutputDataConfig
from sagemaker.train.defaults import DEFAULT_INSTANCE_TYPE


DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_BUCKET_PREFIX = "sample-prefix"
DEFAULT_REGION = "us-west-2"
DEFAULT_COMPUTE = Compute(instance_type=DEFAULT_INSTANCE_TYPE, instance_count=1)
DEFAULT_STOPPING = StoppingCondition(max_runtime_in_seconds=3600)
DEFAULT_OUTPUT = OutputDataConfig(
    s3_output_path=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BUCKET_PREFIX}/test-job",
)


@pytest.fixture(autouse=True)
def modules_session():
    with patch("sagemaker.train.Session", spec=Session) as session_mock, patch(
        "sagemaker.train.defaults.resolve_and_validate_role",
        side_effect=lambda provided_role=None, **kwargs: provided_role or DEFAULT_ROLE,
    ):
        session_instance = session_mock.return_value
        session_instance.default_bucket.return_value = DEFAULT_BUCKET
        session_instance.get_caller_identity_arn.return_value = DEFAULT_ROLE
        session_instance.default_bucket_prefix = DEFAULT_BUCKET_PREFIX
        session_instance.boto_session = MagicMock(spec="boto3.session.Session")
        session_instance.boto_region_name = DEFAULT_REGION
        yield session_instance


def test_no_input_channels_omits_input_data_config():
    """With no input channels, InputDataConfig must be omitted (GH#6156)."""
    trainer = ModelTrainer(
        training_image=DEFAULT_IMAGE,
        base_job_name="repro",
        role=DEFAULT_ROLE,
        compute=DEFAULT_COMPUTE,
        stopping_condition=DEFAULT_STOPPING,
        output_data_config=DEFAULT_OUTPUT,
        # no input_data_config: none needed
    )

    request = trainer._create_training_job_args(boto3=True)

    # An empty InputDataConfig is rejected by CreatePipeline/CreateTrainingJob
    # (botocore shape has min=1). It must be absent, not present-and-empty.
    assert request.get("InputDataConfig") != []
    assert "InputDataConfig" not in request or request["InputDataConfig"]
