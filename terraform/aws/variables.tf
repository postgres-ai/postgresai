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
  description = "postgres_ai version (git tag or branch)"
  type        = string
  default     = "0.10"
}

variable "bind_host" {
  description = "Bind host for internal service ports (127.0.0.1: for localhost only, empty for all interfaces)"
  type        = string
  default     = "127.0.0.1:"
}

variable "grafana_bind_host" {
  description = "Bind host for Grafana port (127.0.0.1: for localhost only, empty for all interfaces)"
  type        = string
  default     = "127.0.0.1:"
}

variable "vm_auth_username" {
  description = "VictoriaMetrics HTTP Basic Auth username"
  type        = string
  default     = "vmauth"
}

variable "vm_auth_password" {
  description = "VictoriaMetrics HTTP Basic Auth password. Leave empty to disable VM auth."
  type        = string
  default     = ""
  sensitive   = true
}

