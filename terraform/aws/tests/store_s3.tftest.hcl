# pgai-mon's store: S3 by default, with an instance role limited to the store
# prefix; store_s3 = false keeps it on the instance (no bucket, no role).
# Run with: terraform test -filter=tests/store_s3.tftest.hcl

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

variables {
  aws_region          = "us-east-1"
  environment         = "Test_Env"
  instance_type       = "t3.micro"
  data_volume_size    = 100
  data_volume_type    = "gp3"
  root_volume_type    = "gp3"
  ssh_key_name        = "test-key"
  allowed_ssh_cidr    = ["10.0.0.0/8"]
  allowed_cidr_blocks = []
  use_elastic_ip      = false
  grafana_password    = "test-password" # gitleaks:allow (mock value for plan-only tests)
}

run "default_creates_a_private_bucket_and_a_scoped_role" {
  command = plan

  assert {
    condition     = length(aws_s3_bucket.store) == 1 && aws_s3_bucket.store[0].bucket_prefix == "test-env-pgai-store-"
    error_message = "store_s3 (default) must create one bucket, named after the environment in lowercase"
  }
  assert {
    condition     = aws_s3_bucket_public_access_block.store[0].block_public_acls && aws_s3_bucket_public_access_block.store[0].restrict_public_buckets
    error_message = "the store bucket must block public access"
  }
  assert {
    condition     = length(aws_s3_bucket_policy.store_tls) == 1 && length(aws_s3_bucket_server_side_encryption_configuration.store) == 1
    error_message = "the created bucket must be encrypted and refuse plain HTTP"
  }
  assert {
    condition     = length(aws_iam_instance_profile.instance) == 1 && length(aws_iam_role_policy.store) == 1
    error_message = "store_s3 must give the instance a role with the store policy"
  }
  assert {
    condition     = aws_instance.main.metadata_options[0].http_put_response_hop_limit == 2 && aws_instance.main.metadata_options[0].http_tokens == "required"
    error_message = "containers reach the role through IMDSv2 only with hop limit 2"
  }
  assert {
    condition     = local.store_prefix == "pgai-mon/Test_Env"
    error_message = "the default prefix is pgai-mon/<environment>"
  }
}

run "given_bucket_policy_is_limited_to_the_prefix" {
  command = plan

  variables {
    store_s3_bucket = "acme-metrics"
    store_s3_prefix = "pgai/prod-1"
  }

  assert {
    condition     = length(aws_s3_bucket.store) == 0
    error_message = "a given bucket must not be created"
  }
  assert {
    condition     = strcontains(aws_instance.main.user_data, "PGAI_MON_STORE=s3://acme-metrics/pgai/prod-1")
    error_message = "user_data must point pgai-mon at s3://bucket/prefix"
  }
  assert {
    condition     = strcontains(aws_instance.main.user_data, "AWS_REGION=us-east-1")
    error_message = "user_data must give pgai-mon the bucket's region"
  }
  assert {
    condition     = jsondecode(local.store_policy).Statement[1].Resource == ["arn:aws:s3:::acme-metrics/pgai/prod-1/*"] && toset(jsondecode(local.store_policy).Statement[0].Condition.StringLike["s3:prefix"]) == toset(["pgai/prod-1", "pgai/prod-1/*"]) && jsondecode(local.store_policy).Statement[0].Resource == ["arn:aws:s3:::acme-metrics"]
    error_message = "object access must be limited to bucket/prefix/*, listing to the prefix"
  }
  assert {
    condition     = length([for st in jsondecode(local.store_policy).Statement : st if contains(st.Action, "s3:*") || contains(st.Resource, "*") || contains(st.Resource, "arn:aws:s3:::acme-metrics/*")]) == 0
    error_message = "the store policy must not grant wildcard actions or resources beyond the prefix"
  }
  assert {
    condition     = toset(jsondecode(local.store_policy).Statement[1].Action) == toset(["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"])
    error_message = "the store needs exactly get, put, delete and multipart abort on objects"
  }
}

run "store_s3_off_keeps_the_store_local" {
  command = plan

  variables {
    store_s3 = false
  }

  assert {
    condition     = length(aws_s3_bucket.store) == 0 && length(aws_iam_role.instance) == 0 && length(aws_iam_instance_profile.instance) == 0
    error_message = "store_s3 = false must create no bucket and no role"
  }
  assert {
    condition     = aws_instance.main.metadata_options[0].http_put_response_hop_limit == 1
    error_message = "without a role, IMDS stays at hop limit 1"
  }
  assert {
    condition     = strcontains(aws_instance.main.user_data, "PGAI_MON_STORE=\n")
    error_message = "without the S3 store PGAI_MON_STORE is empty (local volume)"
  }
}

run "bad_prefix_is_rejected" {
  command = plan

  variables {
    store_s3_prefix = "pgai/*"
  }

  expect_failures = [var.store_s3_prefix]
}
