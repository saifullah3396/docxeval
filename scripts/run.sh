#!/bin/bash

# take the input config file as an argument and source it to get the CONFIGS array
CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

source "$CONFIG_FILE"

for cfg in "${CONFIGS[@]}"; do
    job_name="${cfg%% *}"
    script="${cfg#* }"

    echo "JOB NAME : $job_name"
    echo "SCRIPT   : $script"
    echo "----------------------------------------"
    $script
done