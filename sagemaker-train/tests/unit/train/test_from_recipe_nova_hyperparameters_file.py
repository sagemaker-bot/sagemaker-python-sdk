# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License.
"""Repro for issue #5770: ModelTrainer.from_recipe fails to load a Nova
hyperparameters file because it calls a non-existent method
``_validate_and_load_hyperparameters_file`` instead of
``_validate_and_fetch_hyperparameters_file``.
"""
import json
import tempfile

from unittest.mock import patch, MagicMock

import yaml


def test_from_recipe_nova_hyperparameters_file(tmp_path):
    """Passing a hyperparameters file path (str) to a Nova recipe must not raise."""
    from omegaconf import OmegaConf
    from sagemaker.train.model_trainer import ModelTrainer
    from sagemaker.train.configs import Compute
    from sagemaker.core.helper.session_helper import Session

    # A hyperparameters file (JSON) that the user passes as a string path.
    hp_file = tmp_path / "hyperparameters.json"
    hp_file.write_text(json.dumps({"epochs": 3}))

    base_recipe_cfg = OmegaConf.create(
        {
            "run": {
                "model_type": "amazon.nova.lite",
                "model_name_or_path": "nova-lite",
                "replicas": 1,
            }
        }
    )

    recipe_tmp_dir = tempfile.TemporaryDirectory(prefix="test_recipe_")

    mock_session = MagicMock(spec=Session)
    mock_session.boto_region_name = "us-east-1"
    mock_session.default_bucket_prefix = None
    mock_session.default_bucket.return_value = "sagemaker-us-east-1-123456789012"

    compute = Compute(instance_type="ml.m5.xlarge", instance_count=1)

    with patch("sagemaker.train.model_trainer._determine_device_type", return_value="cpu"), \
         patch("sagemaker.train.model_trainer._load_base_recipe", return_value=base_recipe_cfg), \
         patch("sagemaker.train.model_trainer._is_nova_recipe", return_value=True), \
         patch("sagemaker.train.model_trainer._is_llmft_recipe", return_value=False), \
         patch("sagemaker.train.model_trainer._get_args_from_recipe", return_value=(
             {"hyperparameters": {}},
             recipe_tmp_dir,
         )), \
         patch("sagemaker.train.model_trainer.TrainDefaults.get_sagemaker_session", return_value=mock_session), \
         patch("sagemaker.train.model_trainer.TrainDefaults.get_role", return_value="arn:aws:iam::123456789012:role/SageMakerRole"):

        model_trainer = ModelTrainer.from_recipe(
            training_image="327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes:verl-v1.0.0-smtj",
            training_recipe="training/nova/nova_1_0/nova_lite/CPT/nova_lite_1_0_p5x16_gpu_pretrain",
            compute=compute,
            hyperparameters=str(hp_file),
        )

    # The hyperparameters loaded from the file must be present.
    assert model_trainer.hyperparameters.get("epochs") == 3

    recipe_tmp_dir.cleanup()
