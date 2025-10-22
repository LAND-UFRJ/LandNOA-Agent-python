#!/bin/bash
source .venv/bin/activate

# Run main.py in the background
python main.py &

# Run the adk command
python3 agente/root_agent.py
# Wait for both to finish (optional, keeps script alive)
wait
