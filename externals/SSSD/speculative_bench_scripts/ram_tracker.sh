#!/bin/bash

# This script tracks the maximum RAM usage in gigabytes since it started.

# Initialize the maximum RAM usage to 0.00 GB.
max_ram_gb=0.00

# Set the time interval in seconds for checking RAM usage.
check_interval=2

# Function to be executed when the script is terminated (e.g., with Ctrl+C).
cleanup() {
    echo -e "\n\nScript terminated."
    echo "Maximum RAM usage recorded: ${max_ram_gb} GB"
    exit 0
}

# Trap the interrupt signal (Ctrl+C) and call the cleanup function.
trap cleanup SIGINT

echo "Starting RAM usage monitoring..."
echo "Press Ctrl+C to stop the script."
echo "----------------------------------------"

# Loop indefinitely to check RAM usage.
while true; do
    # Get the current used RAM in megabytes and convert it to gigabytes.
    # 'free -m' displays memory in megabytes.
    # 'awk' processes the output:
    # 'NR==2' selects the second line (the memory line).
    # '{printf "%.2f", $3/1024}' prints the third column (used RAM) divided by 1024 to convert to GB, formatted to two decimal places.
    current_ram_gb=$(free -m | awk 'NR==2{printf "%.2f", $3/1024}')

    # Compare the current RAM usage with the recorded maximum.
    # 'bc -l' is used for floating-point comparison.
    if (( $(echo "$current_ram_gb > $max_ram_gb" | bc -l) )); then
        # If the current usage is higher, update the maximum.
        max_ram_gb=$current_ram_gb
    fi

    # Print the current and maximum RAM usage.
    # '\r' at the beginning of the line moves the cursor to the start,
    # so the output overwrites the previous line, creating a live update effect.
    echo -ne "Current RAM Usage: ${current_ram_gb} GB | Maximum RAM Usage: ${max_ram_gb} GB\r"

    # Wait for the specified interval before the next check.
    sleep $check_interval
done
