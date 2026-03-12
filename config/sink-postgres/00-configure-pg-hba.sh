#!/bin/bash
# Configure pg_hba.conf to allow trust authentication from Docker networks
#
# SECURITY NOTE: This configuration uses trust authentication, which is appropriate
# for this use case because:
# 1. The sink-postgres container runs in an isolated Docker network
# 2. No ports are exposed to the host or external networks
# 3. Only other containers in the same Docker Compose network can connect
# 4. This is an internal data collection service, not a production database
# 5. The container network provides the security boundary
#
# This approach simplifies container-to-container communication while maintaining
# appropriate security isolation from external access.

cat > "${PGDATA}/pg_hba.conf" <<EOF
# PostgreSQL Client Authentication Configuration File
# Custom configuration for sink-postgres container
# 
# SECURITY CONTEXT:
# This configuration uses trust authentication for connections within Docker networks.
# This is safe because:
#   - The container is NOT exposed to external networks (no published ports)
#   - Only containers within the same Docker Compose network can connect
#   - The Docker network itself provides the security boundary
#   - This simplifies internal service communication without compromising security
#
# If you expose this container's ports to the host or internet, you MUST change
# the authentication method to 'scram-sha-256' or 'md5' and use strong passwords.

# TYPE  DATABASE        USER            ADDRESS                 METHOD

# "local" is for Unix domain socket connections only
# Safe: only processes within the same container can connect via socket
local   all             all                                     trust

# IPv4 local connections:
# Safe: only connections from within the container itself
host    all             all             127.0.0.1/32            trust

# IPv6 local connections:
# Safe: only connections from within the container itself
host    all             all             ::1/128                 trust

# Allow replication connections from localhost
# Safe: only for internal container operations
local   replication     all                                     trust
host    replication     all             127.0.0.1/32            trust
host    replication     all             ::1/128                 trust

# Allow all connections from Docker networks without password
# Safe: these are private Docker network ranges used by Docker Compose
# External networks cannot reach these addresses
# 172.16.0.0/12   - Default Docker bridge networks
# 192.168.0.0/16  - User-defined bridge networks
# 10.0.0.0/8      - Additional private network range
host    all             all             172.16.0.0/12           trust
host    all             all             192.168.0.0/16          trust
host    all             all             10.0.0.0/8              trust
EOF

# Reload PostgreSQL configuration
pg_ctl reload -D "${PGDATA}"

