output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.main.id
}

output "data_volume_id" {
  description = "EBS data volume ID (for snapshots)"
  value       = aws_ebs_volume.data.id
}

output "root_volume_id" {
  description = "EBS root volume ID (for snapshots)"
  value       = one(aws_instance.main.root_block_device).volume_id
}

output "public_ip" {
  description = "Public IP address"
  value       = var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip
}

output "grafana_url" {
  description = "Grafana dashboard URL"
  value       = "http://${var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip}:3000"
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/${var.ssh_key_name}.pem ubuntu@${var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip}"
}

output "grafana_credentials" {
  description = "Grafana credentials"
  value = {
    username = "monitor"
    password = var.grafana_password
  }
  sensitive = true
}

output "grafana_access_hint" {
  description = "How to access Grafana based on your configuration"
  value = var.grafana_bind_host == "127.0.0.1:" || length(var.allowed_cidr_blocks) == 0 ? (
    <<-EOT
  
  Grafana Access: SSH Tunnel Required
  
  Your configuration disables direct access (allowed_cidr_blocks is empty).
  
  Step 1: Create SSH tunnel
    ssh -i ~/.ssh/${var.ssh_key_name}.pem -NL 3000:localhost:3000 ubuntu@${var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip}
  
  Step 2: Open browser
    http://localhost:3000
  
  Login:
    Username: monitor
    Password: (see terraform.tfvars)
  
  EOT
  ) : (
    <<-EOT
  
  Grafana Access: Direct URL
  
  Your configuration allows direct access.
  
  URL: http://${var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip}:3000
  
  Login:
    Username: monitor
    Password: (see terraform.tfvars)
  
  Allowed from: ${join(", ", var.allowed_cidr_blocks)}
  
  EOT
  )
}

output "deployment_info" {
  description = "Deployment information"
  value = {
    instance_type        = var.instance_type
    region               = var.aws_region
    public_ip            = var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip
    data_volume          = "${var.data_volume_size} GiB"
    api_key_configured   = var.postgres_ai_api_key != ""
    monitoring_instances = length(var.monitoring_instances)
    demo_mode            = var.enable_demo_db
  }
  sensitive = true
}

output "next_steps" {
  description = "Next steps after deployment"
  value = var.grafana_bind_host == "127.0.0.1:" || length(var.allowed_cidr_blocks) == 0 ? (
    <<-EOT

Deployment complete

Grafana Access: SSH Tunnel Required
  Step 1: ssh -i ~/.ssh/${var.ssh_key_name}.pem -NL 3000:localhost:3000 ubuntu@${var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip}
  Step 2: Open http://localhost:3000
  Login: monitor / (see terraform.tfvars)

Monitoring: ${length(var.monitoring_instances)} instance(s) configured

SSH: ssh -i ~/.ssh/${var.ssh_key_name}.pem ubuntu@${var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip}

For detailed access instructions: terraform output grafana_access_hint
For deployment info: terraform output deployment_info

EOT
  ) : (
    <<-EOT

Deployment complete

Grafana URL: http://${var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip}:3000
  Username: monitor
  Password: see terraform.tfvars
  Allowed from: ${join(", ", var.allowed_cidr_blocks)}

Monitoring: ${length(var.monitoring_instances)} instance(s) configured

SSH: ssh -i ~/.ssh/${var.ssh_key_name}.pem ubuntu@${var.use_elastic_ip ? aws_eip.main[0].public_ip : aws_instance.main.public_ip}

For detailed access instructions: terraform output grafana_access_hint
For deployment info: terraform output deployment_info

EOT
  )
}

