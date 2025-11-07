import os
import json
import chromadb
from dotenv import load_dotenv

load_dotenv()

def connect_chroma():
  """Connects to ChromaDB using credentials from environment variables."""
  host = os.getenv("CHROMA_HOST")
  port = int(os.getenv("CHROMA_PORT"))
  if not host or not port:
    raise ValueError("chroma_url and chroma_port must be set in the interface")
  client = chromadb.HttpClient(host=host, port=port)
  return client

def list_all_collections(client:chromadb.api.client.Client) ->list:
  """Lists all collection names."""
  collection_names = [coll.name for coll in client.list_collections()]
  return collection_names

def get_collection(client:chromadb.api.client.Client,
                  collection_name:str) -> chromadb.api.client.Collection:
  """Gets a collection object by its name."""
  collection = client.get_collection(name=collection_name)
  return collection

def create_collection_with_info(client:chromadb.api.client.Client,
                                collection_name:str,
                                index_method:str,
                                parameters:dict):
  """Creates a new collection. And add the indexing methods informations"""
  collection = client.create_collection(
    name=collection_name,
    metadata={
      "index_method":index_method,
      "parameters":json.dumps(parameters)
    },
    configuration={
    "hnsw": {
        "space": "cosine"
    }
  }
  )
  return collection

def create_collection(client:chromadb.api.client.Client,
                      collection_name:str)->chromadb.api.client.Collection:
  """Creates a new collection.
      This functions will not be used in app.py"""
  collection = client.create_collection(
  	name=collection_name,
    configuration={
      "hnsw": {
        "space": "cosine"
      }
    }
  )
  return collection


def add_documents(collection:chromadb.api.client.Collection,
                  documents:list,
                  source_name:str):
  """Adds processed documents to a ChromaDB collection."""
  current_count = collection.count()
  ids = [f"id_{source_name}_{current_count + i}" for i, _ in enumerate(documents)]
  metadatas = [{"source": source_name} for _ in documents]

  existing = collection.get(where={"source": source_name})
  if existing['ids']:
    print(f"The pdf {source_name} already exist in the {collection}")
    return None

  collection.add(
      ids=ids,
      documents=documents,
      metadatas=metadatas
  )

def show_collection_info(client:chromadb.api.client.Collection,
                         collection_name:str,):
  """Takes information from the collections"""
  collection = client.get_collection(name=collection_name)
  lista = collection.get()['metadatas']
  pdfs_names = [tuple(d.values()) for d in lista]
  unique_pdfs = set(pdfs_names)

  infos = collection.metadata

  return unique_pdfs, infos
