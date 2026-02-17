#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "Starting postgres_ai monitoring installation..."

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl enable docker
systemctl start docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create postgres_ai user
useradd -m -s /bin/bash postgres_ai
usermod -aG docker postgres_ai

# Mount and prepare data volume
if [ ! -d /data ]; then
    mkdir -p /data
    
    # Wait for volume to be attached with proper polling
    echo "Waiting for EBS volume to attach..."
    MAX_RETRIES=60  # 5 minutes total (60 * 5 seconds)
    RETRY=0
    DEVICE=""
    
    while [ $RETRY -lt $MAX_RETRIES ]; do
        if [ -e /dev/nvme1n1 ]; then
            DEVICE=/dev/nvme1n1
            echo "Volume found at $DEVICE"
            break
        elif [ -e /dev/xvdf ]; then
            DEVICE=/dev/xvdf
            echo "Volume found at $DEVICE"
            break
        fi
        
        RETRY=$((RETRY + 1))
        # Log progress every 10 attempts (50 seconds)
        if [ $((RETRY % 10)) -eq 0 ]; then
            echo "Still waiting for volume... (attempt $RETRY/$MAX_RETRIES)"
        fi
        sleep 5
    done
    
    if [ -z "$DEVICE" ]; then
        echo "WARNING: EBS volume not attached after 5 minutes, using root volume"
    fi
    
    if [ -n "$DEVICE" ]; then
        # Check if filesystem exists
        if ! blkid $DEVICE; then
            mkfs.ext4 $DEVICE
        fi
        
        # Mount volume
        mount $DEVICE /data
        
        # Add to fstab for persistence
        UUID=$(blkid -s UUID -o value $DEVICE)
        echo "UUID=$UUID /data ext4 defaults,nofail 0 2" >> /etc/fstab
    fi
fi

# Set permissions
chown -R postgres_ai:postgres_ai /data

# Clone postgres_ai repository
cd /home/postgres_ai
sudo -u postgres_ai git clone --branch ${postgres_ai_version} https://gitlab.com/postgres-ai/postgres_ai.git

# Configure postgres_ai
cd postgres_ai

# Create configuration with secure permissions
umask 077  # Ensure files are created with 600 permissions
cat > .pgwatch-config <<EOF
grafana_password=${grafana_password}
%{ if postgres_ai_api_key != "" }api_key=${postgres_ai_api_key}%{ endif }
%{ if enable_demo_db }demo_mode=true%{ else }demo_mode=false%{ endif }
EOF

# Create .env file for docker-compose
cat > .env <<ENV_EOF
GF_SECURITY_ADMIN_PASSWORD=${grafana_password}
BIND_HOST=${bind_host}
GRAFANA_BIND_HOST=${grafana_bind_host}
ENV_EOF

%{ if vm_auth_password != "" ~}
cat >> .env <<ENV_EOF
VM_AUTH_USERNAME=${vm_auth_username}
VM_AUTH_PASSWORD=${vm_auth_password}
ENV_EOF
%{ endif ~}

# Ensure secure permissions
chmod 600 .pgwatch-config .env

# Configure monitoring instances from template
cat > instances.yml <<'INSTANCES_EOF'
${instances_yml}INSTANCES_EOF

# Set ownership
chown -R postgres_ai:postgres_ai /home/postgres_ai/postgres_ai

# Create systemd service
cat > /etc/systemd/system/postgres-ai.service <<'SERVICE_EOF'
[Unit]
Description=Postgres AI Monitoring
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/postgres_ai/postgres_ai
User=postgres_ai
Group=postgres_ai

# Start services
ExecStart=/usr/local/bin/docker-compose up -d

# Stop services
ExecStop=/usr/local/bin/docker-compose down

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Enable and start service
systemctl daemon-reload
systemctl enable postgres-ai
systemctl start postgres-ai

# Wait for services to be healthy
sleep 30

# Reset Grafana admin password to match terraform config
echo "Setting Grafana admin password..."
cd /home/postgres_ai/postgres_ai
docker exec grafana-with-datasources grafana-cli admin reset-admin-password "${grafana_password}"

echo "Installation complete!"
echo "Access Grafana at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000"
echo "Username: monitor"
echo "Password: ${grafana_password}"
if [ -n "${vm_auth_username}" ] && [ -n "${vm_auth_password}" ]; then
  echo ""
  echo "VictoriaMetrics Auth: enabled (username: ${vm_auth_username})"
fi

