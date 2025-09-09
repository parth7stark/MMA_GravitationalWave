#!/bin/bash

set -x

endpoint_name='mma-server'
proxystore-endpoint list
proxystore-endpoint configure ${endpoint_name}
proxystore-endpoint --log-level DEBUG start ${endpoint_name}

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

export PROXYSTORE_SERVER_ENDPOINT=${uuid}

for ((i=0;i<${NDETECTORS};i++))
do
    detector_file="/app/.proxystore/detector${i}"

    # Get server endpoint UUID
    while true; do
        if [ -s "${detector_file}" ]
        then
            export PROXYSTORE_DETECTOR${i}_ENDPOINT=$(cat "$detector_file")
            break
        fi
        ls ${detector_file}
    done

done

python /app/examples/octopus/run_server.py --config examples/configs/FLserver.yaml