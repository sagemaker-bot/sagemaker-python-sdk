"""Reproduction for issue #6019.

Under an AWS Organization that uses condition-based SCPs, the IAM policy
simulator cannot faithfully evaluate the SCP layer, so
``iam:SimulatePrincipalPolicy`` returns ``EvalDecision: implicitDeny`` with an
empty ``MatchedStatements`` list and ``OrganizationsDecisionDetail.
AllowedByOrganizations == "false"`` — even for actions the role is actually
allowed to perform at run time.

The client-side pre-check in ``resolve_and_validate_role`` treats this org-layer
verdict as a definitive denial and raises ``RoleValidationError`` — a false
positive. It should instead treat an org-layer-only denial (no matched identity
statement) as *unverifiable* (warn + proceed), the same as the "caller can't
simulate" path.
"""
from unittest.mock import MagicMock

import pytest

from sagemaker.core.helper.iam_role_resolver import resolve_and_validate_role


def _make_session(caller_arn, account="123456789012"):
    mock_session = MagicMock()
    mock_iam = MagicMock()
    mock_sts = MagicMock()

    def client_factory(service, **kwargs):
        return mock_iam if service == "iam" else mock_sts

    mock_session.boto_session.client.side_effect = client_factory
    mock_sts.get_caller_identity.return_value = {"Arn": caller_arn, "Account": account}
    return mock_session, mock_iam, mock_sts


def _trusted_doc():
    return {
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ]
    }


def test_condition_based_scp_denial_is_not_a_false_positive():
    """An org-layer-only ``implicitDeny`` (SCP condition unresolvable) must not raise.

    The role's own identity policies grant the smoke-test actions (no matched
    identity Deny statement), but the simulator reports ``implicitDeny`` because
    ``AllowedByOrganizations`` is ``false`` — an artifact of the simulator being
    unable to evaluate condition-based SCPs. This is unverifiable, not a real
    denial, so the resolver should proceed (returning the ARN) rather than raise.
    """
    arn = "arn:aws:iam::123456789012:role/TrainingExecRole"
    mock_session, mock_iam, _ = _make_session(
        "arn:aws:sts::123456789012:assumed-role/Other/sess"
    )
    mock_iam.get_role.return_value = {
        "Role": {"AssumeRolePolicyDocument": _trusted_doc()}
    }

    def paginate(**kwargs):
        # Every simulated smoke-test action comes back as an org-layer denial:
        # implicitDeny, NO matched identity statements, AllowedByOrganizations=false.
        return [
            {
                "EvaluationResults": [
                    {
                        "EvalActionName": action,
                        "EvalDecision": "implicitDeny",
                        "MatchedStatements": [],
                        "OrganizationsDecisionDetail": {
                            "AllowedByOrganizations": False
                        },
                    }
                    for action in kwargs["ActionNames"]
                ]
            }
        ]

    paginator = MagicMock()
    paginator.paginate.side_effect = paginate
    mock_iam.get_paginator.return_value = paginator

    # The real CreateTrainingJob would succeed; the pre-check must not block it.
    result = resolve_and_validate_role(
        provided_role=arn, role_type="training", sagemaker_session=mock_session
    )
    assert result == arn
