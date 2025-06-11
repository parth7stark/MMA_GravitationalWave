#!/bin/bash

set -x

# cat /root/.local/share/proxystore

server_file=/app/.proxystore/server

# Get server endpoint UUID
while true; do
    if [ -s "${server_file}" ]
    then
        export PROXYSTORE_SERVER_ENDPOINT=$(cat "$server_file")
        break
    fi
    ls ${server_file}
done

proxystore-endpoint list
proxystore-endpoint configure ${PROXYSTORE_ENDPOINT_NAME}
proxystore-endpoint start ${PROXYSTORE_ENDPOINT_NAME}

log_file="$HOME/.local/share/proxystore/${PROXYSTORE_ENDPOINT_NAME}/log.txt"

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

export PROXYSTORE_DETECTOR_ENDPOINT=$(proxystore-endpoint list | grep ${PROXYSTORE_ENDPOINT_NAME} | awk '{print $NF}')

python /app/examples/octopus/run_detector.py --config ${CONFIGURATION_FILE}