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
  kms_key_id        = var.encryption_kms_key_arn != "" ? var.encryption_kms_key_arn : null

  tags = {
    Name = "${var.environment}-postgres-ai-data"
  }
}

# pgai-mon store in S3 (store_s3, default): a bucket (or a given one) and an
# instance role that can list, read, write and delete under the store prefix
# only. The role reaches the containers through IMDSv2, which needs one more
# hop than the instance itself (the Docker bridge): hop limit 2 when it exists.
locals {
  store_bucket_name = var.store_s3_bucket != "" ? var.store_s3_bucket : (
    var.store_s3 ? aws_s3_bucket.store[0].bucket : ""
  )
  store_prefix   = var.store_s3_prefix != "" ? var.store_s3_prefix : "pgai-mon/${var.environment}"
  pgai_mon_store = var.store_s3 ? "s3://${local.store_bucket_name}/${local.store_prefix}" : ""

  store_bucket_arn = "arn:aws:s3:::${local.store_bucket_name}"
  store_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListStorePrefix"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [local.store_bucket_arn]
        Condition = {
          StringLike = { "s3:prefix" = [local.store_prefix, "${local.store_prefix}/*"] }
        }
      },
      {
        Sid      = "StoreObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
        Resource = ["${local.store_bucket_arn}/${local.store_prefix}/*"]
      },
    ]
  })
}

resource "aws_s3_bucket" "store" {
  count = var.store_s3 && var.store_s3_bucket == "" ? 1 : 0
  # bucket_prefix is at most 37 characters; bucket names are lowercase.
  bucket_prefix = substr("${replace(lower(var.environment), "/[^a-z0-9-]/", "-")}-pgai-store-", 0, 37)

  tags = {
    Name = "${var.environment}-postgres-ai-store"
  }
}

resource "aws_s3_bucket_public_access_block" "store" {
  count                   = length(aws_s3_bucket.store)
  bucket                  = aws_s3_bucket.store[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "store" {
  count  = length(aws_s3_bucket.store)
  bucket = aws_s3_bucket.store[0].id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "store" {
  count  = length(aws_s3_bucket.store)
  bucket = aws_s3_bucket.store[0].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Refuse plain-HTTP access to the store bucket.
resource "aws_s3_bucket_policy" "store_tls" {
  count  = length(aws_s3_bucket.store)
  bucket = aws_s3_bucket.store[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.store[0].arn, "${aws_s3_bucket.store[0].arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.store]
}

resource "aws_iam_role" "instance" {
  count = var.store_s3 ? 1 : 0
  name  = "${var.environment}-postgres-ai-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "store" {
  count  = var.store_s3 ? 1 : 0
  name   = "pgai-mon-store"
  role   = aws_iam_role.instance[0].id
  policy = local.store_policy
}

resource "aws_iam_instance_profile" "instance" {
  count = var.store_s3 ? 1 : 0
  name  = "${var.environment}-postgres-ai-instance"
  role  = aws_iam_role.instance[0].name
}

# EC2 Instance
resource "aws_instance" "main" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  subnet_id     = aws_subnet.main.id

  vpc_security_group_ids = [aws_security_group.main.id]
  iam_instance_profile   = var.store_s3 ? aws_iam_instance_profile.instance[0].name : null

  key_name = var.ssh_key_name

  root_block_device {
    volume_size = 30
    volume_type = var.root_volume_type
    encrypted   = true
    kms_key_id  = var.encryption_kms_key_arn != "" ? var.encryption_kms_key_arn : null
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    grafana_password    = var.grafana_password
    postgres_ai_api_key = var.postgres_ai_api_key
    enable_demo_db      = var.enable_demo_db
    postgres_ai_version = var.postgres_ai_version
    bind_host           = var.bind_host
    grafana_bind_host   = var.grafana_bind_host
    aws_region          = var.aws_region
    pgai_mon_store      = local.pgai_mon_store
    instances_yml = templatefile("${path.module}/instances.yml.tpl", {
      monitoring_instances = var.monitoring_instances
      enable_demo_db       = var.enable_demo_db
    })
  })

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = var.store_s3 ? 2 : 1
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

