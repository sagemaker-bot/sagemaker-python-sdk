"""Regression test for issue #5795.

Both ``sagemaker-train`` and ``sagemaker-core`` ship a physical
``sagemaker/__init__.py`` file in their source tree, and both configure
setuptools to include the top-level ``sagemaker`` package (``include =
["sagemaker*"]`` with ``namespaces = true``).

Because ``sagemaker`` is meant to be a shared *namespace* package, only one
distribution may own the ``sagemaker/__init__.py`` file (or, for a PEP 420
namespace, neither should). When both distributions install a file at
``site-packages/sagemaker/__init__.py`` -- and the file contents differ -- the
two wheels conflict on installation:

    pkg sagemaker-train-1.9.0 conflicts with sagemaker-core-2.9.0
    (installs files into the same place)
    Problematic file: .../site-packages/sagemaker/__init__.py

This test detects the conflict by comparing the ``sagemaker/__init__.py`` that
each distribution would install.
"""

import os


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

TRAIN_INIT = os.path.join(
    REPO_ROOT, "sagemaker-train", "src", "sagemaker", "__init__.py"
)
CORE_INIT = os.path.join(
    REPO_ROOT, "sagemaker-core", "src", "sagemaker", "__init__.py"
)


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def test_no_conflicting_sagemaker_namespace_init():
    """sagemaker-train and sagemaker-core must not both install the same file
    ``sagemaker/__init__.py`` with differing contents."""

    train_exists = os.path.exists(TRAIN_INIT)
    core_exists = os.path.exists(CORE_INIT)

    # sagemaker-core owns the shared namespace package init. sagemaker-train
    # must NOT also ship its own 'sagemaker/__init__.py', otherwise both wheels
    # install a file at the identical path and conflict on installation.
    assert core_exists, "expected sagemaker-core to own sagemaker/__init__.py"
    assert not (train_exists and core_exists), (
        "sagemaker-train and sagemaker-core both ship 'sagemaker/__init__.py' "
        "at the same install path, causing a file conflict when both are "
        "installed together (issue #5795). Only one distribution may own this "
        "namespace package init file."
    )
