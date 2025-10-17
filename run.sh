#!/bin/bash
source .venv/bin/activate

# Run main.py in the background
python main.py &

# Run the adk command
adk api_server --host 0.0.0.0 --port 12000 --a2a &

# Wait for both to finish (optional, keeps script alive)
wait
