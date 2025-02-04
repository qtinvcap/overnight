#!/bin/bash

#THIS SCRIPT IS MEANT TO RUN IN THE BACKGROUND WITH (LOGS LOCATED IN => ibgateway.log)
# nohup ./restart_ibgateway.sh > /home/ubuntu/trading_project/ibgateway.log 2>&1 &

# Locate the process ID of the script
# ps aux | grep ibgateway
# kill [process_id]

echo "Stopping IB Gateway..."
my_pid=$$
# Check if the process is running before attempting to kill it
if pgrep -f ibgateway | grep -v $my_pid; then
    pgrep -f ibgateway | grep -v $my_pid | xargs -r kill
    echo "IB Gateway processes stopped."
else
    echo "No IB Gateway processes were running."
fi

echo "Stopping Xvfb..."
# Check if the process is running before attempting to kill it
if pgrep -f Xvfb | grep -v $my_pid; then
    pgrep -f Xvfb | grep -v $my_pid | xargs -r kill
    echo "Xvfb processes stopped."
else
    echo "No Xvfb processes were running."
fi

echo "Waiting for processes to terminate..."
sleep 5

echo "Starting Xvfb..."
Xvfb :1 -screen 0 1024x768x16 &
sleep 2
if ! pgrep -f "Xvfb :1"; then
    echo "Failed to start Xvfb, stopping script."
    exit 1
fi

echo "Setting DISPLAY variable..."
export DISPLAY=:1.0

echo "Starting IB Gateway..."
/opt/ibc/gatewaystart.sh

echo "Script execution completed: IB Gateway in now Running."
