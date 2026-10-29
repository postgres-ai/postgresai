# docker-compose.yml refuses to start without PGAI_TAG and VM auth, and the
# demo DB's replication needs REPLICATOR_PASSWORD: user_data must write all of
# them to .env (postgresai#292).
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
}

run "generated_vm_password_and_replicator_password" {
  command = plan

  variables {
    vm_auth_password = ""
  }

  assert {
    condition     = strcontains(aws_instance.main.user_data, "PGAI_TAG=0.18.0-rc.1")
    error_message = "user_data must set PGAI_TAG to the release tag"
  }
  assert {
    condition     = strcontains(aws_instance.main.user_data, "VM_AUTH_USERNAME=vmauth")
    error_message = "user_data must always set VM_AUTH_USERNAME"
  }
  assert {
    condition     = strcontains(aws_instance.main.user_data, "for secret_name in VM_AUTH_PASSWORD REPLICATOR_PASSWORD; do")
    error_message = "without vm_auth_password the instance must generate VM_AUTH_PASSWORD and REPLICATOR_PASSWORD"
  }
}

run "given_vm_password_is_used" {
  command = plan

  variables {
    vm_auth_password = "given-vm-password" # gitleaks:allow (mock value)
  }

  assert {
    condition     = strcontains(aws_instance.main.user_data, "VM_AUTH_PASSWORD=given-vm-password")
    error_message = "a given vm_auth_password must be written to .env"
  }
  assert {
    condition     = strcontains(aws_instance.main.user_data, "for secret_name in REPLICATOR_PASSWORD; do")
    error_message = "with vm_auth_password set only REPLICATOR_PASSWORD is generated"
  }
}
