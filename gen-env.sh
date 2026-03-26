#!/bin/bash
# Gera o .env com valores dinâmicos do sistema para o docker compose

cat > "$(dirname "$0")/.env" <<EOF
UID=$(id -u)
GID=$(id -g)
MEMBERS_GID=$(getent group members | cut -d: -f3)
CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
EOF

echo ".env gerado:"
cat "$(dirname "$0")/.env"
