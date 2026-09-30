"""Regression test for issue #5930.

Server Side Encryption using a KMS key fails for _validate_s3_path_exists.

When the target S3 bucket enforces SSE-KMS (bucket policy denies PutObject that
does not carry the correct server-side-encryption headers), the empty-prefix
put_object performed by _validate_s3_path_exists is rejected with AccessDenied.

Since SFTTrainer (and friends) accept a kms_key_id, _create_output_config must
forward that key to _validate_s3_path_exists so the prefix-creation put_object
includes the SSE-KMS headers and succeeds.
"""

from unittest.mock import Mock, patch

import pytest

from sagemaker.train.common_utils.finetune_utils import _create_output_config


class _AccessDenied(Exception):
    """Stand-in for the S3 AccessDenied error raised on SSE-enforced buckets."""


@patch("boto3.client")
def test_create_output_config_forwards_kms_key_to_put_object(mock_boto_client):
    """On an SSE-KMS-enforced bucket, prefix creation must send the KMS key.

    A bucket that enforces SSE-KMS denies any put_object that omits the
    ServerSideEncryption / SSEKMSKeyId headers. We simulate that: put_object
    raises AccessDenied unless it is called with the KMS headers. The current
    code calls put_object without those headers, so validation fails.
    """
    kms_key_id = "arn:aws:kms:us-west-2:123456789012:key/abcd-1234"

    mock_session = Mock()
    mock_session.boto_region_name = "us-west-2"
    mock_s3_client = Mock()
    mock_session.boto_session.client.return_value = mock_s3_client

    # Prefix does not exist yet -> triggers put_object.
    mock_s3_client.list_objects_v2.return_value = {}

    def put_object(**kwargs):
        # SSE-enforced bucket: reject writes lacking the KMS encryption headers.
        if kwargs.get("SSEKMSKeyId") != kms_key_id or kwargs.get(
            "ServerSideEncryption"
        ) != "aws:kms":
            raise _AccessDenied("Access Denied")
        return {}

    mock_s3_client.put_object.side_effect = put_object

    # Should NOT raise: the KMS key must be threaded through to put_object.
    _create_output_config(
        mock_session,
        s3_output_path="s3://sse-bucket/output",
        kms_key_id=kms_key_id,
    )

    _, kwargs = mock_s3_client.put_object.call_args
    assert kwargs.get("SSEKMSKeyId") == kms_key_id
    assert kwargs.get("ServerSideEncryption") == "aws:kms"
