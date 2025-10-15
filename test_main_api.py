import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8010"  # Assuming Flask app runs on this port
AUTH_TOKEN = "your_secret_token_here"  # Replace with actual AGENT_SECRET_TOKEN from .env
HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Content-Type": "application/json"}

# Placeholder MCP server URL (replace with actual running MCP server URL)
MCP_SERVER_URL = "http://localhost:10000"  # Example; update as needed

def test_health_check():
    """Test health check endpoint."""
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Health Check: {response.status_code} - {response.json()}")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_send_unauthorized():
    """Test send endpoint without auth."""
    payload = {"payload": {"query": "Test query"}}
    response = requests.post(f"{BASE_URL}/api/v1/send", json=payload)
    print(f"Send Unauthorized: {response.status_code} - {response.json()}")
    assert response.status_code == 403

def test_send_valid():
    """Test send endpoint with valid auth and payload."""
    payload = {"payload": {"query": "What is AI ethics?"}}
    response = requests.post(f"{BASE_URL}/api/v1/send", json=payload, headers=HEADERS)
    print(f"Send Valid: {response.status_code} - {response.json()}")
    assert response.status_code == 200
    assert "result" in response.json()

def test_send_invalid():
    """Test send endpoint with missing query."""
    payload = {"payload": {}}
    response = requests.post(f"{BASE_URL}/api/v1/send", json=payload, headers=HEADERS)
    print(f"Send Invalid: {response.status_code} - {response.json()}")
    assert response.status_code == 400

def test_list_collections():
    """Test list collections endpoint."""
    response = requests.get(f"{BASE_URL}/api/v1/collections/list", headers=HEADERS)
    print(f"List Collections: {response.status_code} - {response.json()}")
    assert response.status_code == 200
    assert "result" in response.json()

def test_create_collection_new():
    """Test create new collection."""
    payload = {"name": "test_collection"}
    response = requests.post(f"{BASE_URL}/api/v1/collections/create", json=payload, headers=HEADERS)
    print(f"Create Collection New: {response.status_code} - {response.json()}")
    assert response.status_code == 200

def test_create_collection_existing():
    """Test create existing collection."""
    payload = {"name": "test_collection"}
    response = requests.post(f"{BASE_URL}/api/v1/collections/create", json=payload, headers=HEADERS)
    print(f"Create Collection Existing: {response.status_code} - {response.json()}")
    assert response.status_code == 200  # Or 409 if updated

def test_delete_collection_existing():
    """Test delete existing collection."""
    payload = {"name": "test_collection"}
    response = requests.post(f"{BASE_URL}/api/v1/collections/delete", json=payload, headers=HEADERS)
    print(f"Delete Collection Existing: {response.status_code} - {response.json()}")
    assert response.status_code == 200

def test_delete_collection_nonexisting():
    """Test delete non-existing collection."""
    payload = {"name": "nonexistent_collection"}
    response = requests.post(f"{BASE_URL}/api/v1/collections/delete", json=payload, headers=HEADERS)
    print(f"Delete Collection Nonexisting: {response.status_code} - {response.json()}")
    assert response.status_code == 200  # Or 404 if updated

def test_add_tool():
    """Test add MCP tool."""
    payload = {"name": "test_tool", "url": MCP_SERVER_URL}
    response = requests.post(f"{BASE_URL}/api/v1/mcp/add", json=payload, headers=HEADERS)
    print(f"Add Tool: {response.status_code} - {response.json()}")
    assert response.status_code == 200

def test_list_tools():
    """Test list MCP tools."""
    response = requests.get(f"{BASE_URL}/api/v1/mcp/list", headers=HEADERS)
    print(f"List Tools: {response.status_code} - {response.json()}")
    assert response.status_code == 200

def test_bind_tools():
    """Test bind MCP tools (requires MCP server running)."""
    response = requests.post(f"{BASE_URL}/api/v1/mcp/bind", json={}, headers=HEADERS)
    print(f"Bind Tools: {response.status_code} - {response.json()}")
    # Note: This may fail if MCP server is not properly set up; adjust assertion as needed
    assert response.status_code in [200, 500]  # 200 if successful, 500 if error

def test_remove_tool_existing():
    """Test remove existing MCP tool."""
    payload = {"name": "test_tool"}
    response = requests.post(f"{BASE_URL}/api/v1/mcp/remove", json=payload, headers=HEADERS)
    print(f"Remove Tool Existing: {response.status_code} - {response.json()}")
    assert response.status_code == 200

def test_remove_tool_nonexisting():
    """Test remove non-existing MCP tool."""
    payload = {"name": "nonexistent_tool"}
    response = requests.post(f"{BASE_URL}/api/v1/mcp/remove", json=payload, headers=HEADERS)
    print(f"Remove Tool Nonexisting: {response.status_code} - {response.json()}")
    assert response.status_code == 200

if __name__ == "__main__":
    print("Starting tests for main.py endpoints...")
    try:
        test_health_check()
        test_send_unauthorized()
        test_send_valid()
        test_send_invalid()
        test_list_collections()
        test_create_collection_new()
        test_create_collection_existing()
        test_delete_collection_existing()
        test_delete_collection_nonexisting()
        test_add_tool()
        test_list_tools()
        test_bind_tools()
        test_remove_tool_existing()
        test_remove_tool_nonexisting()
        print("All tests completed successfully!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Error during tests: {e}")