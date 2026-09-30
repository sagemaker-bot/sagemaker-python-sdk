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
"""Reproduction test for issue #5956.

ModelTrainer (V3) does not respect a user-provided KMS key when uploading
source code / driver files to S3. In secure environments where S3 bucket
policies enforce server-side encryption with KMS, these uploads (which use
``Session.upload_data`` with no encryption ExtraArgs) are rejected, so
ModelTrainer cannot be used with custom training scripts.

This test configures an ``output_data_config`` KMS key and asserts that the
source-code uploads are performed with that KMS key. On the unfixed code no
KMS key is ever passed to ``upload_data``, so the test fails.
"""
from __future__ import absolute_import

from unittest.mock import patch, MagicMock

from sagemaker.core.helper.session_helper import Session
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.defaults import DEFAULT_INSTANCE_TYPE
from sagemaker.train.configs import (
    Compute,
    StoppingCondition,
    OutputDataConfig,
    SourceCode,
)
from tests.unit import DATA_DIR

DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_BUCKET_PREFIX = "sample-prefix"
DEFAULT_REGION = "us-west-2"
DEFAULT_SOURCE_DIR = f"{DATA_DIR}/script_mode"
KMS_KEY_ID = "arn:aws:kms:us-west-2:000000000000:key/my-kms-key"


def _make_session():
    session = MagicMock(spec=Session)
    session.default_bucket.return_value = DEFAULT_BUCKET
    session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    session.default_bucket_prefix = DEFAULT_BUCKET_PREFIX
    session.boto_session = MagicMock(spec="boto3.session.Session")
    session.boto_region_name = DEFAULT_REGION
    session.upload_data.return_value = f"s3://{DEFAULT_BUCKET}/code"
    return session


@patch("sagemaker.train.defaults.resolve_and_validate_role", return_value=DEFAULT_ROLE)
@patch("sagemaker.train.model_trainer.TrainingJob")
def test_source_code_upload_uses_output_kms_key(mock_training_job, mock_resolve_role):
    """Source-code / driver uploads must use the configured output KMS key.

    Regression for issue #5956: ModelTrainer uploads source code and driver
    files with ``Session.upload_data`` without any KMS encryption, so it cannot
    be used in environments that enforce SSE-KMS on the staging bucket.
    """
    session = _make_session()

    trainer = ModelTrainer(
        training_image=DEFAULT_IMAGE,
        role=DEFAULT_ROLE,
        sagemaker_session=session,
        compute=Compute(instance_type=DEFAULT_INSTANCE_TYPE, instance_count=1),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=3600),
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BUCKET_PREFIX}/output",
            kms_key_id=KMS_KEY_ID,
        ),
        source_code=SourceCode(
            source_dir=DEFAULT_SOURCE_DIR,
            entry_script="custom_script.py",
        ),
    )

    # Avoid the IAM SimulatePrincipalPolicy round-trip in staging bucket resolution.
    with patch.object(trainer, "_resolve_staging_bucket", return_value=(DEFAULT_BUCKET, None)):
        trainer._create_training_job_args()

    # The source code / sm_drivers channels trigger local -> S3 uploads.
    assert session.upload_data.call_count >= 1, "expected source code to be uploaded to S3"

    # Every upload of source artifacts must carry the KMS key so that buckets
    # enforcing SSE-KMS accept the object.
    for call in session.upload_data.call_args_list:
        kwargs = call.kwargs
        extra_args = kwargs.get("extra_args") or (
            call.args[4] if len(call.args) > 4 else None
        )
        assert extra_args is not None, (
            "ModelTrainer uploaded source code without encryption ExtraArgs; "
            "the configured output KMS key is ignored (issue #5956)."
        )
        assert extra_args.get("SSEKMSKeyId") == KMS_KEY_ID
        assert extra_args.get("ServerSideEncryption") == "aws:kms"
