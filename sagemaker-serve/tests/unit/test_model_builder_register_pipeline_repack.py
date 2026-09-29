# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Reproduction for issue #5828.

ModelBuilder.register() under a PipelineSession, with custom inference
source_code (entry_script + source_dir) and an S3 model artifact, must flag a
runtime repack so that ``ModelStep`` inserts a ``_RepackModelStep`` (as V2 did).
The repack step bundles the inference code into the model artifact; without it
the registered model package cannot be repacked and is unusable.

The pipeline-context return value of register() is a ``_ModelStepArguments``
whose ``need_runtime_repack`` set is consumed by ``ModelStep`` to decide whether
to append the repack step. In V3 this set is left empty, so no repack step is
generated -- this test asserts it is populated (parity with V2).
"""
from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core.training.configs import SourceCode
from sagemaker.core.workflow.pipeline_context import PipelineSession


class _FakePipelineSession(PipelineSession):
    """A PipelineSession usable without any AWS calls.

    Bypasses the base ``Session.__init__`` (which reaches out to boto) while
    keeping the real ``context`` / ``init_model_step_arguments`` /
    ``_intercept_create_request`` behavior that ``runnable_by_pipeline`` relies
    on. It IS an instance of ``PipelineSession`` so the pipeline code path is
    taken.
    """

    def __init__(self):
        self._context = None
        self._region_name = "us-west-2"
        self.config = {}
        self.sagemaker_config = {}
        self.local_mode = False
        self.default_bucket_prefix = "prefix"
        self.settings = MagicMock()
        self.boto_session = MagicMock()
        self.boto_session.region_name = "us-west-2"
        self.sagemaker_client = MagicMock()

    def default_bucket(self):
        return "test-bucket"


@pytest.fixture(autouse=True)
def _ambient_execution_role():
    with patch(
        "sagemaker.serve.model_builder.resolve_and_validate_role",
        return_value="arn:aws:iam::123456789012:role/ambient-execution-role",
    ):
        yield


def test_register_generates_repack_step_with_pipeline_session():
    from sagemaker.serve import ModelBuilder

    pipeline_session = _FakePipelineSession()

    model_builder = ModelBuilder(
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/img:latest",
        s3_model_data_url="s3://bucket/model.tar.gz",
        source_code=SourceCode(
            entry_script="infer.py",
            source_dir="s3://bucket/source/sourcedir.tar.gz",
        ),
        role_arn="arn:aws:iam::123456789012:role/ambient-execution-role",
        sagemaker_session=pipeline_session,
        env_vars={"SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": "application/json"},
    )
    model_builder.framework = None
    model_builder.framework_version = None

    with patch("sagemaker.core.s3.determine_bucket_and_prefix", return_value=("bucket", "prefix")), \
        patch("sagemaker.serve.model_builder.repack_model") as mock_repack, \
        patch(
            "sagemaker.serve.model_builder.get_model_package_args",
            return_value={},
        ), \
        patch(
            "sagemaker.serve.model_builder.create_model_package_from_containers",
            return_value={
                "ModelPackageArn": "arn:aws:sagemaker:us-west-2:1:model-package/g/1"
            },
        ), \
        patch(
            "sagemaker.serve.model_builder.update_container_with_inference_params",
            side_effect=lambda **kw: kw.get("container_def"),
        ):
        step_args = model_builder.register(
            content_types=["application/jsonlines"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="my-group",
            approval_status="Approved",
        )

    # register() under a PipelineSession returns the _ModelStepArguments context.
    assert step_args is not None
    assert hasattr(step_args, "need_runtime_repack")

    # A repack step MUST be flagged (V2 parity). The bug leaves this empty and
    # instead repacks eagerly, so ModelStep never inserts a _RepackModelStep.
    assert step_args.need_runtime_repack, (
        "register() under a PipelineSession with source_code did not flag a "
        "runtime repack; ModelStep will not generate a repack step (issue #5828)."
    )
