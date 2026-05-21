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
"""Fixtures for ModelTrainer integration tests."""
from __future__ import annotations

import os

import boto3
import pytest

import sagemaker


@pytest.fixture(scope="session")
def region():
    """Return the AWS region for testing."""
    return os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


@pytest.fixture(scope="session")
def sagemaker_session(region):
    """Create a SageMaker session for integration tests."""
    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


@pytest.fixture(scope="session")
def instance_type():
    """Return the instance type for testing."""
    return os.environ.get("SM_TEST_INSTANCE_TYPE", "ml.m5.xlarge")
