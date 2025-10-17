# LandNOA-Agent-Python

A Python-based AI agent application built with Flask, integrating ChromaDB for vector collections, MCP tools, and registry-based agent management. It supports health checks, collection manipulation, tool management, and periodic heartbeats for agent registration.

## Features
- **Flask API**: Endpoints for health checks, Chroma collection CRUD operations, and MCP tool management.
- **ChromaDB Integration**: Manage vector collections for RAG (Retrieval-Augmented Generation) workflows.
- **Authentication**: Bearer token-based auth for secure endpoints.
- **Registry Integration**: Automatic registration, deregistration, and heartbeat with a central registry.
- **MCP Tools**: Add, remove, and list tools dynamically (triggers hot reload).
- **Concurrency**: Runs with threading for background tasks like heartbeats.
- **Planned Enhancements**: Support for plain text splitting (currently PDF-only), Prometheus metrics (tokens, uptime, requests), conversation monitoring (MongoDB), RAG validation, and a management interface.

## Requirements
- Python 3.x
- Dependencies: `flask`, `requests`, `chromadb`, `python-dotenv`, `pathlib` install via `pip install -r requirements.txt` if available
- External services: ChromaDB server, Registry server, ADK (for `adk api_server`)
- Environment variables (via `.env`): `AGENT_ID`, `AGENT_SECRET_TOKEN`, `AGENT_BASE_URL`, `REGISTRY_BASE_URL`, `FLASK_RUN_PORT`, `CHROMADB_URL`, `CHROMADB_PORT`

## Installation
1. Clone or navigate to the project directory.
2. Install dependencies: `pip install flask requests chromadb python-dotenv`.
3. Set up your `.env` file with required variables.

## Running
- Run the agent: `python main.py` (starts Flask server and heartbeat thread).
- For full setup, run concurrently with ADK: Use the provided `run.sh` script (e.g., `./run.sh`) to start both `python main.py` and `adk api_server --a2a`.

## API Endpoints
- `GET /api/v1/health`: Health check.
- `GET /api/v1/collections/list`: List Chroma collections (requires auth).
- `POST /api/v1/collections/create`: Create a collection (requires auth).
- `POST /api/v1/collections/delete`: Delete a collection (requires auth).
- `POST /api/v1/mcp/add`: Add an MCP tool.
- `POST /api/v1/mcp/remove`: Remove an MCP tool.
- `GET /api/v1/mcp/list`: List MCP tools.

# THIS IS NOT FINISHED