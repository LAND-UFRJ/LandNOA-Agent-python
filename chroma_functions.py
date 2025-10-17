import chromadb
from dotenv import load_dotenv
import os
load_dotenv()

# --- Functions to manage ChromaDB collections 

def connect_chroma():
    """Connects to ChromaDB using credentials from environment variables."""
    host = os.getenv('CHROMADB_HOST')
    port = os.getenv('CHROMADB_PORT')
    if not host or not port:
        raise ValueError("CHROMADB_HOST and CHROMADB_PORT must be set in your .env file")
    client = chromadb.HttpClient(host=host, port=port)
    return client

def list_all_collections(client):
    """Lists all collection names."""
    collection_names = [coll.name for coll in client.list_collections()]
    return collection_names

def get_collection(client, collection_name):
    """Gets a collection object by its name."""
    collection = client.get_collection(name=collection_name)
    return collection

def create_collection(client, collection_name):
    """Creates a new collection."""
    collection = client.create_collection(name=collection_name)
    return collection

def add_documents(collection, documents, source_name):
    """Adds processed documents to a ChromaDB collection."""
    current_count = collection.count()
    ids = [f"id_{source_name}_{current_count + i}" for i, _ in enumerate(documents)]
    metadatas = [{"source": source_name} for _ in documents]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )