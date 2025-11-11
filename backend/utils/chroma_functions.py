import chromadb
from dotenv import load_dotenv
import os
import json
load_dotenv()

def connect_chroma():
    """Connects to ChromaDB using credentials from environment variables."""
    host = os.getenv('CHROMA_HOST')
    port = os.getenv('CHROMA_PORT')
    if not host or not port:
        raise ValueError("CHROMA_HOST and CHROMA_PORT must be set in your .env file")
    client = chromadb.HttpClient(host=host, port=port)
    return client

def get_collection(client:chromadb.api.client.Client,
                  collection_name:str) -> chromadb.api.client.Collection:
    """Gets a collection object by its name."""
    collection = client.get_collection(name=collection_name)
    return collection

def create_collection(client:chromadb.api.client.Client,
                      collection_name:str)->chromadb.api.client.Collection:
  """Creates a new collection in ChromaDB."""
  collection = client.create_collection(
    name=collection_name,
    configuration={
      "hnsw": {
        "space": "cosine"
      }
    }
  )
  return collection

def delete_collection(client:chromadb.api.client.Client, collection_name: str):
    """Deletes a collection from ChromaDB."""
    client.delete_collection(name=collection_name)

def add_documents(collection:chromadb.api.client.Collection,
                  documents:list,
                  source_name:str):
    """Adds processed documents to a ChromaDB collection."""
    current_count = collection.count()
    ids = [f"id_{source_name}_{current_count + i}" for i, _ in enumerate(documents)]
    metadatas = [{"source": source_name} for _ in documents]
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    return True