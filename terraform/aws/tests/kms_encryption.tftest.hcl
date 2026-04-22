# Tests for encryption_kms_key_arn variable validation and KMS key propagation.
# Requires Terraform >= 1.7 (mock_provider support).
# Run with: terraform test -filter=tests/kms_encryption.tftest.hcl

mock_provider "aws" {
  mock_data "aws_availability_zones" {
    defaults = {
      names = ["us-east-1a", "us-east-1b", "us-east-1c"]
    }
  }

  mock_data "aws_ami" {
    defaults = {
      id = "ami-0123456789abcdef0"
    }
  }
}
mock_provider "local" {}

# Shared base variables for all runs (required variables with mock values)
variables {
  aws_region         = "us-east-1"
  environment        = "test"
  instance_type      = "t3.micro"
  data_volume_size   = 100
  data_volume_type   = "gp3"
  root_volume_type   = "gp3"
  ssh_key_name       = "test-key"
  allowed_ssh_cidr   = ["10.0.0.0/8"]
  allowed_cidr_blocks = []
  use_elastic_ip     = false
  grafana_password   = "test-password"
}

# --- Validation: valid inputs must pass ---

run "empty_string_passes_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = ""
  }

  assert {
    condition     = aws_ebs_volume.data.encrypted == true
    error_message = "EBS data volume must be encrypted at rest"
  }

  # kms_key_id is Computed+Optional in the AWS provider schema, so its planned
  # value is "known after apply" even when we set it to null in the config.
  # The null-path cannot be asserted at plan time with a mock provider.
  # Non-null propagation (ARN → kms_key_id) is verified in
  # standard_key_arn_passes_validation and kms_key_propagates_to_* runs.

  assert {
    condition     = one(aws_instance.main.root_block_device).encrypted == true
    error_message = "EC2 root block device must be encrypted at rest"
  }
}

run "standard_key_arn_passes_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
  }

  assert {
    condition     = aws_ebs_volume.data.kms_key_id == "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
    error_message = "EBS data volume kms_key_id must equal encryption_kms_key_arn"
  }

  assert {
    condition     = one(aws_instance.main.root_block_device).kms_key_id == "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
    error_message = "Root block device kms_key_id must equal encryption_kms_key_arn"
  }
}

run "uppercase_hex_key_arn_passes_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:key/12345678-ABCD-1234-1234-123456789012"
  }
}

run "multi_region_key_arn_passes_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:key/mrk-1234abcd12ab34cd56ef1234567890ab"
  }
}

run "alias_arn_passes_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:alias/my-ebs-key"
  }
}

run "govcloud_key_arn_passes_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws-us-gov:kms:us-gov-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
  }
}

# --- Validation: invalid inputs must fail ---

run "plain_string_fails_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "not-an-arn"
  }

  expect_failures = [var.encryption_kms_key_arn]
}

run "wrong_service_fails_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:s3:::my-bucket"
  }

  expect_failures = [var.encryption_kms_key_arn]
}

run "truncated_arn_fails_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:key/..."
  }

  expect_failures = [var.encryption_kms_key_arn]
}

run "short_account_id_fails_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:12345678901:key/12345678-1234-1234-1234-123456789012"
  }

  expect_failures = [var.encryption_kms_key_arn]
}

run "malformed_key_id_fails_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234"
  }

  expect_failures = [var.encryption_kms_key_arn]
}

run "long_account_id_fails_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:1234567890123:key/12345678-1234-1234-1234-123456789012"
  }

  expect_failures = [var.encryption_kms_key_arn]
}

run "alias_arn_with_dots_passes_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:alias/my.project.ebs-key"
  }
}

run "short_key_id_fails_validation" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:key/abc"
  }

  expect_failures = [var.encryption_kms_key_arn]
}

# --- Propagation: KMS key ARN must be forwarded to EBS resources ---

run "kms_key_propagates_to_ebs_data_volume" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
  }

  assert {
    condition     = aws_ebs_volume.data.kms_key_id == "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
    error_message = "EBS data volume kms_key_id must equal encryption_kms_key_arn when non-empty"
  }

  assert {
    condition     = aws_ebs_volume.data.encrypted == true
    error_message = "EBS data volume must be encrypted at rest"
  }
}

run "kms_key_propagates_to_root_block_device" {
  command = plan

  variables {
    encryption_kms_key_arn = "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
  }

  assert {
    condition     = one(aws_instance.main.root_block_device).kms_key_id == "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
    error_message = "EC2 root block device kms_key_id must equal encryption_kms_key_arn when non-empty"
  }

  assert {
    condition     = one(aws_instance.main.root_block_device).encrypted == true
    error_message = "EC2 root block device must be encrypted at rest"
  }
}
