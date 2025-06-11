#!/bin/bash

set -x

endpoint_name='mma-server'
proxystore-endpoint list
proxystore-endpoint configure ${endpoint_name}
proxystore-endpoint start ${endpoint_name}

log_file="$HOME/.local/share/proxystore/${endpoint_name}/log.txt"

while true; do
    if cat "$log_file" | grep -q "Uvicorn running on http://"; then
        echo "Detected 'Uvicorn running on http://'. Proceed to mini app."
        break
    else
        echo "$(date): Waiting for Uvicorn to start. Retrying in 5 seconds..."
        tail -n 1 "$log_file"
        sleep 5
    fi
done

uuid=$(proxystore-endpoint list | grep ${endpoint_name} | awk '{print $NF}')
echo ${uuid} > /app/.proxystore/server

python /app/examples/octopus/run_server.py --config examples/configs/FLserver.yaml