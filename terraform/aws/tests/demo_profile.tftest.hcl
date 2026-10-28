# The demo DB containers (target-db, target-standby) start only with the "demo"
# compose profile, so enable_demo_db must turn it on in the box's .env.
# Run with: terraform test -filter=tests/demo_profile.tftest.hcl

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
}

run "demo_db_turns_on_the_demo_profile" {
  command = plan

  variables {
    enable_demo_db = true
  }

  assert {
    condition     = strcontains(aws_instance.main.user_data, "COMPOSE_PROFILES=demo")
    error_message = "enable_demo_db must write COMPOSE_PROFILES=demo to .env, or target-db never starts"
  }
}

run "no_demo_db_no_demo_profile" {
  command = plan

  variables {
    enable_demo_db = false
  }

  assert {
    condition     = !strcontains(aws_instance.main.user_data, "COMPOSE_PROFILES=demo")
    error_message = "without enable_demo_db the demo profile must stay off"
  }
}
