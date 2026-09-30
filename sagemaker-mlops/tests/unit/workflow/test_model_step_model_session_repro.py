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
"""Reproduction for issue #5829: Model does not have a sagemaker_session property.

ModelStep expects the ``model`` object referenced in the step arguments to expose a
``sagemaker_session`` attribute (it validates that it is a ``PipelineSession``), but
the ``sagemaker.core.resources.Model`` spec (a pydantic model with ``extra="forbid"``)
does not define such an attribute. Passing a ``Model`` instance as the ``model`` in the
register/create model step arguments therefore fails.
"""
from __future__ import absolute_import

from sagemaker.core.resources import Model
from sagemaker.core.workflow.pipeline_context import _ModelStepArguments
from sagemaker.mlops.workflow.model_step import ModelStep


def test_model_step_accepts_model_instance():
    """ModelStep should be constructible when the step args reference a Model instance."""
    model = Model(model_name="test-model")

    step_args = _ModelStepArguments(model=model)
    step_args.caller_name = "create_model"
    step_args.create_model_request = {"ModelName": "test-model"}

    # On the buggy code this raises AttributeError because the core Model resource
    # has no ``sagemaker_session`` attribute.
    step = ModelStep(name="model-step", step_args=step_args)
    assert step.name == "model-step"
