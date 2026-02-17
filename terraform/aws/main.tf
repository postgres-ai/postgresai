terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "postgres-ai-monitoring"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.environment}-postgres-ai-vpc"
  }
}

resource "aws_subnet" "main" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.environment}-postgres-ai-subnet"
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.environment}-postgres-ai-igw"
  }
}

resource "aws_route_table" "main" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.environment}-postgres-ai-rt"
  }
}

resource "aws_route_table_association" "main" {
  subnet_id      = aws_subnet.main.id
  route_table_id = aws_route_table.main.id
}

# Security Group
resource "aws_security_group" "main" {
  name        = "${var.environment}-postgres-ai-sg"
  description = "Security group for postgres_ai monitoring EC2"
  vpc_id      = aws_vpc.main.id

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidr
  }

  # Grafana (optional, only if allowed_cidr_blocks is not empty)
  # If empty, use SSH tunnel: ssh -i ~/.ssh/key.pem -NL 3000:localhost:3000 ubuntu@<instance-ip>
  dynamic "ingress" {
    for_each = length(var.allowed_cidr_blocks) > 0 ? [1] : []
    content {
      description = "Grafana"
      from_port   = 3000
      to_port     = 3000
      protocol    = "tcp"
      cidr_blocks = var.allowed_cidr_blocks
    }
  }

  # Allow all outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-postgres-ai-sg"
  }
}

# EBS Volume for data persistence
resource "aws_ebs_volume" "data" {
  availability_zone = aws_subnet.main.availability_zone
  size              = var.data_volume_size
  type              = var.data_volume_type
  encrypted         = true

  tags = {
    Name = "${var.environment}-postgres-ai-data"
  }
}

# EC2 Instance
resource "aws_instance" "main" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  subnet_id     = aws_subnet.main.id

  vpc_security_group_ids = [aws_security_group.main.id]

  key_name = var.ssh_key_name

  root_block_device {
    volume_size = 30
    volume_type = var.root_volume_type
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    grafana_password     = var.grafana_password
    postgres_ai_api_key  = var.postgres_ai_api_key
    enable_demo_db       = var.enable_demo_db
    postgres_ai_version  = var.postgres_ai_version
    bind_host            = var.bind_host
    grafana_bind_host    = var.grafana_bind_host
    vm_auth_username     = var.vm_auth_username
    vm_auth_password     = var.vm_auth_password
    instances_yml        = templatefile("${path.module}/instances.yml.tpl", {
      monitoring_instances = var.monitoring_instances
      enable_demo_db       = var.enable_demo_db
    })
  })

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  tags = {
    Name = "${var.environment}-postgres-ai-monitoring"
  }

  lifecycle {
    ignore_changes = [user_data]
  }
}

# Attach EBS volume
resource "aws_volume_attachment" "data" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.data.id
  instance_id = aws_instance.main.id
}

# Elastic IP (optional, for stable IP)
resource "aws_eip" "main" {
  count    = var.use_elastic_ip ? 1 : 0
  instance = aws_instance.main.id
  domain   = "vpc"

  tags = {
    Name = "${var.environment}-postgres-ai-eip"
  }
}

# Generate instances.yml from template
resource "local_sensitive_file" "instances_config" {
  content = templatefile("${path.module}/instances.yml.tpl", {
    monitoring_instances = var.monitoring_instances
    enable_demo_db       = var.enable_demo_db
  })
  filename = "${path.module}/.terraform/instances.yml"
}

# Deploy instances.yml to EC2 when config changes
resource "terraform_data" "deploy_config" {
  triggers_replace = {
    config_hash          = local_sensitive_file.instances_config.content_md5
    monitoring_instances = jsonencode(var.monitoring_instances)
    enable_demo_db       = var.enable_demo_db
  }

  depends_on = [aws_instance.main, aws_volume_attachment.data]

  provisioner "remote-exec" {
    inline = [
      "if ! sudo test -f /home/postgres_ai/postgres_ai/postgres_ai; then echo 'Skipping - installation not complete'; exit 0; fi",
      "cat > /tmp/instances.yml << 'EOF'",
      local_sensitive_file.instances_config.content,
      "EOF",
      "sudo mv /tmp/instances.yml /home/postgres_ai/postgres_ai/instances.yml",
      "sudo chown postgres_ai:postgres_ai /home/postgres_ai/postgres_ai/instances.yml",
      "sudo -u postgres_ai /home/postgres_ai/postgres_ai/postgres_ai update-config",
      "echo 'Config updated successfully'"
    ]
    
    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file("~/.ssh/${var.ssh_key_name}.pem")
      host        = var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip
    }
  }
}

