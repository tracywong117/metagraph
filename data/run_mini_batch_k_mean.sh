#!/bin/bash

# Name of the Python script
PYTHON_SCRIPT="mini_batch_k_mean.py"

# Maximum number of retries if the script fails
MAX_RETRIES=100

# Retry counter
RETRY_COUNT=0

# Sleep time between retries (in seconds)
SLEEP_TIME=60

# Start with resume mode set to 0 (start from scratch)
RESUME_MODE=1

# Function to run the Python script
run_script() {
    echo "Running Mini-Batch K-means with resume mode: $RESUME_MODE"
    python $PYTHON_SCRIPT $RESUME_MODE
}

# Main loop
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    run_script
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Script completed successfully!"
        exit 0
    else
        echo "Script failed with exit code $EXIT_CODE. Retrying in $SLEEP_TIME seconds..."
        RETRY_COUNT=$((RETRY_COUNT + 1))
        RESUME_MODE=1  # Switch to resume mode after the first failure
        sleep $SLEEP_TIME
    fi
done

echo "Script failed after $MAX_RETRIES retries."
exit 1