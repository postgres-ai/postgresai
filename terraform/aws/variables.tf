variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "data_volume_size" {
  description = "Size of EBS data volume in GiB"
  type        = number
}

variable "data_volume_type" {
  description = "EBS volume type for data disk (gp3 for SSD, st1 for HDD throughput optimized, sc1 for HDD cold)"
  type        = string
}

variable "root_volume_type" {
  description = "EBS volume type for root disk (gp3 for SSD, gp2 for older SSD)"
  type        = string
}

variable "ssh_key_name" {
  description = "Name of SSH key pair for EC2 access"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed for Grafana access"
  type        = list(string)
}

variable "use_elastic_ip" {
  description = "Allocate Elastic IP for stable address"
  type        = bool
}

variable "grafana_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
}

variable "postgres_ai_api_key" {
  description = "PostgresAI API key (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "monitoring_instances" {
  description = "PostgreSQL instances to monitor"
  type = list(object({
    name        = string
    conn_str    = string
    environment = string
    cluster     = string
    node_name   = string
  }))
  default = []
}

variable "enable_demo_db" {
  description = "Enable demo database"
  type        = bool
  default     = false
}

variable "postgres_ai_version" {
  description = "postgres_ai release tag (also the container image tag, PGAI_TAG)"
  type        = string
  default     = "0.10"
}

variable "encryption_kms_key_arn" {
  description = "KMS key ARN or alias ARN for EBS encryption. Leave empty to use the default AWS-managed aws/ebs key (AES-256). Accepts key ARNs (arn:aws:kms:REGION:ACCOUNT:key/UUID) and alias ARNs (arn:aws:kms:REGION:ACCOUNT:alias/NAME). WARNING: changing this value in any direction (empty to ARN, ARN to empty, or ARN to different ARN) on existing infrastructure will destroy and recreate EBS volumes and the EC2 instance, causing data loss. Snapshot both volumes before making this change on a live deployment."
  type        = string
  default     = ""

  validation {
    condition     = var.encryption_kms_key_arn == "" || can(regex("^arn:aws(-[a-z-]+)?:kms:[a-z0-9-]+:[0-9]{12}:(key/((mrk-[a-fA-F0-9]{32})|([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}))|alias/[a-zA-Z0-9/._-]+)$", var.encryption_kms_key_arn))
    error_message = "Must be a valid KMS key ARN (e.g., 'arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012') or alias ARN (e.g., 'arn:aws:kms:us-east-1:123456789012:alias/my-ebs-key'), or empty string to use the AWS-managed aws/ebs key."
  }
}

variable "bind_host" {
  description = "Written to .env as BIND_HOST for older tooling; the pgai-mon stack publishes pgai-mon on 127.0.0.1 regardless"
  type        = string
  default     = "127.0.0.1:"
}

variable "grafana_bind_host" {
  description = "Bind host for Grafana port (127.0.0.1: for localhost only, empty for all interfaces)"
  type        = string
  default     = "127.0.0.1:"
}

variable "store_s3" {
  description = "Keep pgai-mon's Parquet store in S3 (default), so the instance is disposable: the instance role may read and write only the store prefix. false: the store stays in a Docker volume on the instance."
  type        = bool
  default     = true
}

variable "store_s3_bucket" {
  description = "Existing S3 bucket for the store (in aws_region). Empty: Terraform creates a private, SSE-S3 encrypted bucket named <environment>-pgai-store-<suffix>."
  type        = string
  default     = ""

  validation {
    condition     = var.store_s3_bucket == "" || can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.store_s3_bucket))
    error_message = "store_s3_bucket must be a valid S3 bucket name (lowercase letters, digits, dots, hyphens; 3-63 characters) or empty."
  }
}

variable "store_s3_prefix" {
  description = "Key prefix of the store in the bucket; the instance role's permissions are limited to it. Empty: pgai-mon/<environment>."
  type        = string
  default     = ""

  validation {
    condition     = var.store_s3_prefix == "" || can(regex("^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$", var.store_s3_prefix))
    error_message = "store_s3_prefix must be path segments of letters, digits, '.', '_' or '-' separated by single '/', with no leading or trailing '/' and no wildcards."
  }
}
