# docker-compose.yml refuses to start without PGAI_TAG and PGAI_MON_TOKEN, and
# the demo DB's replication needs REPLICATOR_PASSWORD: user_data must write all
# of them to .env (postgresai#292).
# Run with: terraform test -filter=tests/env_required.tftest.hcl

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
  environment         = "test"
  instance_type       = "t3.micro"
  data_volume_size    = 100
  data_volume_type    = "gp3"
  root_volume_type    = "gp3"
  ssh_key_name        = "test-key"
  allowed_ssh_cidr    = ["10.0.0.0/8"]
  allowed_cidr_blocks = []
  use_elastic_ip      = false
  grafana_password    = "test-password" # gitleaks:allow (mock value for plan-only tests)
  postgres_ai_version = "0.18.0-rc.1"
  # a named bucket: the name of a created one is unknown at plan time, and user_data carries it
  store_s3_bucket = "test-store-bucket"
}

run "tag_and_generated_secrets" {
  command = plan

  assert {
    condition     = strcontains(aws_instance.main.user_data, "PGAI_TAG=0.18.0-rc.1")
    error_message = "user_data must set PGAI_TAG to the release tag"
  }
  assert {
    condition     = strcontains(aws_instance.main.user_data, "for secret_name in PGAI_MON_TOKEN PGAI_MON_READ_TOKEN REPLICATOR_PASSWORD; do")
    error_message = "the instance must generate PGAI_MON_TOKEN, PGAI_MON_READ_TOKEN and REPLICATOR_PASSWORD"
  }
  assert {
    condition     = !strcontains(aws_instance.main.user_data, "VM_AUTH")
    error_message = "the pgai-mon stack has no VictoriaMetrics credentials"
  }
}
