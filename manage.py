#!/usr/bin/env python3
# filepath: /home/theo/Desktop/LAND/LandNOA-Agent-python/manage.py
import subprocess
import sys
import os
import time
from pathlib import Path

# Paths to scripts
MAIN_PY = Path("main.py")
AGENT_PY = Path("agente/root_agent.py")
RETRIEVAL_PY = Path("mcps/retrieval.py")

processes = []

def start_services():
  """Start all services in the background."""
  print("Starting services...")
  project_root = Path(__file__).parent
  os.chdir(project_root)  # Ensure we're in the project root
  
  # Set PYTHONPATH to include the project root for module imports
  env = os.environ.copy()
  env['PYTHONPATH'] = str(project_root)
  
  # Note: Venv is already activated in your shell, so no need for os.system here

  # Start main.py
  proc1 = subprocess.Popen(["python", str(MAIN_PY)], env=env)
  processes.append(proc1)
  time.sleep(3)
  # Start mcps/retrieval.py
  proc3 = subprocess.Popen(["python3", str(RETRIEVAL_PY)], env=env)
  processes.append(proc3)
  time.sleep(3)
  # Start agente/root_agent.py
  proc2 = subprocess.Popen(["python3", str(AGENT_PY)], env=env)
  processes.append(proc2)

  print("All services started. Press Ctrl+C to stop.")

  try:
    # Wait for all processes
    for proc in processes:
      proc.wait()
  except KeyboardInterrupt:
    stop_services()

def stop_services():
  """Stop all running services."""
  print("Stopping services...")
  for proc in processes:
    if proc.poll() is None:  # If still running
      proc.terminate()
      proc.wait()
  print("All services stopped.")

if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "stop":
    stop_services()
  else:
    start_services()