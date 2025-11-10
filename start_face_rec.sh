#!/bin/bash

# Start Face Recognition Backend
echo "Starting Face Recognition Backend on port 5006..."
echo "========================================"

# Kill any existing processes on port 5006
lsof -ti:5006 | xargs kill -9 2>/dev/null

# Check if we're in the right directory
cd "$(dirname "$0")"

# Try to use the system Python that has all dependencies
# Based on the error output, it seems to be using /Users/kabirmathur/Documents/a_s/venv
if [ -d "/Users/kabirmathur/Documents/a_s/venv" ]; then
    echo "Using venv from /Users/kabirmathur/Documents/a_s/venv"
    source /Users/kabirmathur/Documents/a_s/venv/bin/activate
    python app.py
elif [ -d "venv" ]; then
    echo "Using local venv"
    source venv/bin/activate
    python app.py
else
    echo "Using system Python"
    python3 app.py
fi
