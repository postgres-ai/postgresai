# Terraform deployment modules

Infrastructure as Code modules for deploying postgres_ai monitoring to cloud providers.

## Available modules

### AWS (EC2)
Single EC2 instance deployment with Docker Compose.

- **Path**: `aws/`
- **Architecture**: Single EC2 instance with Docker Compose
- **Best for**: Small to medium deployments (1-10 databases)
- **Documentation**: [aws/README.md](aws/README.md)

### GCP (coming soon)
Deploy to Google Cloud Platform using Compute Engine or Cloud Run.

### Azure (coming soon)
Deploy to Microsoft Azure using Virtual Machines or Container Instances.

## Quick start

### AWS deployment

```bash
cd terraform/aws

# Copy example variables
cp terraform.tfvars.example terraform.tfvars

# Edit variables with your settings
vim terraform.tfvars

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Deploy infrastructure (takes 5-10 minutes)
terraform apply
```

## Architecture overview

The AWS deployment creates:

1. **Compute**
   - Single EC2 instance (t3.medium default)
   - Ubuntu 22.04 LTS (Jammy) with Docker and Docker Compose
   - Systemd service for automatic startup

2. **Storage**
   - EBS volume for persistent data
   - Automated snapshots available via AWS Backup

3. **Networking**
   - VPC with public subnet
   - Security Group with restricted access
   - Optional Elastic IP for stable addressing

4. **Monitoring stack**
   - Runs docker-compose from cloned repository
   - Grafana accessible on port 3000

## Security considerations

- EC2 instance in public subnet (can be changed to private with bastion)
- Security groups restrict access to SSH and Grafana only
- All data encrypted at rest (EBS encryption)
- Recommended: Use AWS Systems Manager Session Manager instead of SSH
- Recommended: Restrict `allowed_cidr_blocks` to your office/VPN IP

## Instance types

Recommended instance types based on workload:

- **t3.medium**: 2 vCPU, 4 GiB RAM - suitable for 1-3 databases (default)
- **t3.large**: 2 vCPU, 8 GiB RAM - suitable for 3-10 databases
- **t3.xlarge**: 4 vCPU, 16 GiB RAM - suitable for 10+ databases

Additional options:
- Use Spot Instances for non-critical workloads (subject to interruption)
- Disable Elastic IP if stable address not required

## Support

For issues or questions:
- Open an issue on GitLab
- Contact PostgresAI support
- Check documentation at https://postgres.ai

