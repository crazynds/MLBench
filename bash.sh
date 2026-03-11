#/bin/bash


docker compose -f docker-compose-arm64.yaml up -d

docker compose -f docker-compose-arm64.yaml exec pytorch bash