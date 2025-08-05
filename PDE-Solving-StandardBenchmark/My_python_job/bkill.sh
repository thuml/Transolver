#!/usr/bin/env bash

# List your jobs (assumes default user)
# Using "-o id -u <user> -noheader" prints only job IDs, newest at bottom
jobids=( $(bjobs -u $(whoami) -o id -noheader) )

# Check count
total=${#jobids[@]}
echo "Total jobs found: $total"

if (( total <= 1 )); then
  echo "Nothing to kill, only last job present."
  exit 0
fi

# Compute which jobs to kill: all except the last
to_kill=( "${jobids[@]:0:total-1}" )

echo "Killing jobs: ${to_kill[*]}"
bkill "${to_kill[@]}"

echo "Done. Kept job ID: ${jobids[-1]}"

