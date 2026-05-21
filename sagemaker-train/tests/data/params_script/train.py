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
"""Simple training script for MPI distributed training integration tests."""
from __future__ import annotations

import argparse
import os
import json


def main():
    """Run a simple training loop to validate MPI setup and hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    args, _ = parser.parse_known_args()

    print(f"Training with lr={args.lr}, epochs={args.epochs}")
    print(f"Current host: {os.environ.get('SM_CURRENT_HOST', 'unknown')}")
    print(f"Hosts: {os.environ.get('SM_HOSTS', '[]')}")

    # Write a dummy model artifact to indicate successful training
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.json"), "w") as f:
        json.dump({"lr": args.lr, "epochs": args.epochs, "status": "completed"}, f)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
