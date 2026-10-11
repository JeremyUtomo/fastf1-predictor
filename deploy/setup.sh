#!/usr/bin/env bash
# One-time bootstrap for a fresh Ubuntu 24.04 server (run as root).
# Installs Docker, starts the app and enables the auto-deploy timer.
set -euxo pipefail

DIR=/opt/fastf1-predictor

# Swap: loading a FastF1 session can briefly need more than 2 GB
if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y git jq curl docker.io docker-compose-v2
systemctl enable --now docker

mkdir -p /var/lib/fastf1
chown 1000:1000 /var/lib/fastf1  # the container's f1 user

[ -d "$DIR" ] || git clone https://github.com/JeremyUtomo/fastf1-predictor.git "$DIR"
cd "$DIR"
docker compose up -d --build

cat > /etc/systemd/system/fastf1-deploy.service <<EOF
[Unit]
Description=Deploy fastf1-predictor when main changes and CI passed
After=docker.service network-online.target

[Service]
Type=oneshot
ExecStart=/bin/bash $DIR/deploy/auto-deploy.sh
EOF

cat > /etc/systemd/system/fastf1-deploy.timer <<EOF
[Unit]
Description=Check for fastf1-predictor updates every 2 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=2min

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now fastf1-deploy.timer
