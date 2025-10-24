import os
import chromadb
import time
import json
from dotenv import load_dotenv
from pathlib import Path
from FlagEmbedding import FlagReranker
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from fastmcp import FastMCP
from typing import Dict, List, Any

mcp = FastMCP("RAG server")

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

OPENAI_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_KEY = os.getenv('OPENAI_KEY')
CHROMA_URL = os.getenv('CHROMADB_URL')
CHROMA_PORT = os.getenv('CHROMADB_PORT',8000)
MODEL = os.getenv('OPENAI_MODEL')
AGENT_NAME = os.getenv('AGENT_NAME')

client = chromadb.HttpClient(host=CHROMA_URL,port=CHROMA_PORT)

llm = ChatOpenAI(base_url=OPENAI_URL,model="qwen3:14b",api_key=OPENAI_KEY)

# Cache reranker instance for performance
RE_RANKER = None

def get_reranker():
  """gets the reranker"""
  global RE_RANKER
  if RE_RANKER is None:
    RE_RANKER = FlagReranker('BAAI/bge-reranker-base',
                            use_fp16=True,
                            normalize=True)
  return RE_RANKER
def get_collections():
  """Get the names of the collections"""
  return [x.name for x in client.list_collections()]
def rerank_documents(query: str,
                    documents: List[str],
                    top_r: int =None) -> tuple[List[str], List[float]]:
  """Helper function to rerank documents using the reranker model."""
  reranker = get_reranker()
  tuples = [[query, d] for d in documents]
  scores = reranker.compute_score(tuples)
  if top_r:
    indices = sorted(range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True)[:top_r]
    return [documents[i] for i in indices], [scores[i] for i in indices]
  else:
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [documents[i] for i in indices], [scores[i] for i in indices]
def get_metadatas():
  """Returns the metadatas of chroma collections"""
  collections = client.list_collections()
  metadatas = []
  for collection in collections:
    x = client.get_collection(collection.name).metadata
    metadatas.append(x)
  return metadatas

def save_result(result: Dict[str, Any]):
  """Save the result to retrieval_results.json"""
  file_path = Path(__file__).parent / "retrieval_results.json"
  try:
    if file_path.exists():
      with open(file_path, 'r',encoding="UTF-8") as f:
        data = json.load(f)
    else:
      data = []
    data.append(result)
    with open(file_path, 'w',encoding="UTF-8") as f:
      json.dump(data, f, indent=2)
  except Exception as e:
    print(f"Error saving result: {e}")


@mcp.prompt(name='Initial Considerations',enabled=True)
def initial_instruction():
  """Initial instructions"""
  return f"""This is the RAG mcp server for you, the agent {AGENT_NAME}. At
every prompt the user sends you, you should use one of these functions in order to
retrieve knowledge from the vector base and enhance your answer. The
collections available are {get_collections()}
and their respective descriptions are {get_metadatas()}.
You should choose the collection according to the user prompt, but the method of
retrieval is up to you. Choose wisely, according to each method's description"""

@mcp.tool(
    name="sentence_window_retrieval",
    description="Implements the SWR (Sentence Window Retrieval) strategy. "
                "This function finds the most relevant document and also "
                "retrieves the documents that were physically stored next "
                "to it (before and after), assuming they might contain "
                "relevant context. "
                f"The collections available are {get_collections()} "
                f"and their respective descriptions are {get_metadatas()}.",
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "collection": {"type": "string"},
            "content": {"type": "array", "items": {"type": "string"}},
            "distances": {
                "type": "array",
                "items": {"oneOf": [{"type": "number"}, {"type": "null"}]}
            },
            "parameters": {"type": "object"},
            "time": {"type": "number"}
        }
    }
)
def sentence_window_retrieval(query: str,
                              collection_name: str,
                              n_main: int = 1,
                              n_around: int = 3) -> Dict[str, Any]:
  """
  Implements the SWR (Sentence Window Retrieval) strategy.
  This function finds the most relevant document and also retrieves the
  documents that were physically stored next to it (before and after),
  assuming they might contain relevant context.
  Args:
    query (str): The user's query.
    collection_name (str): The name of an existing ChromaDB collection.
    n_main (int): The number of central documents to find.
    n_around (int): The number of neighboring documents to retrieve.
  Returns:
    Dict[str, Any]: A dictionary with 'collection', 'content', 'distances',
      'parameters', and 'query'.
  """
  try:
    collection = client.get_collection(name=collection_name)
  except Exception as e:
    raise ValueError(f"Collection '{collection_name}' not found: {e}")

  ids = collection.get()['ids']
  all_ids = set()
  distances_map = {}

  results = collection.query(query_texts=query,
                             n_results=n_main,
                             include=['distances'])
  for i, dist in zip(results['ids'][0], results['distances'][0]):
    index = ids.index(i)
    indexes = ids[max(index-n_around,0):min(index+n_around+1,len(ids))]
    for x in indexes:
      all_ids.add(x)
      if x == i:
        distances_map[x] = dist

  all_docs = collection.get(ids=list(all_ids))['documents']
  doc_map = {id_: doc for id_, doc in zip(list(all_ids), all_docs)}

  final_docs = []
  distances_list = []
  for x in sorted(all_ids, key=lambda x: ids.index(x)):
    final_docs.append(doc_map[x])
    distances_list.append(distances_map.get(x, None))
  result = {
    'query': query,
    'collection': collection_name,
    'content': final_docs,
    'distances': distances_list,
    'parameters': {'n_main': n_main,
                   'n_around': n_around},
    'time':time.time()
  }
  save_result(result)
  return result

@mcp.tool(
    name="multi_query",
    description="Generates multiple variations of a query using an LLM to "
                "broaden the search. This technique helps find documents "
                "that the original query might have missed, improving the "
                "chance of finding relevant information (recall). "
                f"The collections available are {get_collections()} "
                f"and their respective descriptions are {get_metadatas()}.",
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "collection": {"type": "string"},
            "content": {"type": "array", "items": {"type": "string"}},
            "distances": {"type": "array", "items": {"type": "number"}},
            "parameters": {"type": "object"},
            "time": {"type": "number"}
        }
    }
)
def multi_query(query: str,
                collection_name: str,
                n_results: int,
                n_queries: int = 5) -> Dict[str, Any]:
  """
  Generates multiple variations of a query using an LLM to broaden the search.
  This technique helps find documents that the original query might have missed,
  improving the chance of finding relevant information (recall).
  Args:
    query (str): The user's original query.
    collection_name (str): The name of an existing ChromaDB collection.
    n_results (int): The number of results per query variation.
    n_queries (int): The number of query variations to generate.
  Returns:
    Dict[str, Any]: A dictionary with 'collection', 'content', 'distances',
      'parameters', and 'query'.
  """
  try:
    collection = client.get_collection(name=collection_name)
  except Exception as e:
    raise ValueError(f"Collection '{collection_name}' not found: {e}")

  questions = {}
  for n in range(n_queries):
    questions[f'question_{n}'] = ''
  messages = [
    SystemMessage(content=f'''You are a helpful assistant that will enhance my
RAG application. I'll give you one question and you will rewrite it in
{n_queries} different ways. You cannot use under any circumstances your
knowledge, just rewriting. You should follow strictly the following schema
{questions}, nothing less nothing more than that'''),
    HumanMessage(content=f'The question is: {query}')
  ]
  try:
    rewrite = eval(llm.invoke(messages).content)
  except Exception as e:
    raise ValueError(f"Failed to parse LLM response: {e}")

  final_docs = []
  distances_list = []
  for question in rewrite.keys():
    answer = collection.query(query_texts=[rewrite[question]],
                              n_results=n_results,
                              include=['documents', 'distances'])
    distances = answer.get('distances', [[]])[0]
    documents = answer.get('documents', [[]])[0]
    final_docs.extend(documents)
    distances_list.extend(distances)
  result = {
    'query': query,
    'collection': collection_name,
    'content': final_docs,
    'distances': distances_list,
    'parameters': {'n_queries': n_queries,
                   'n_results': n_results},
    'time':time.time()
  }
  save_result(result)
  return result

@mcp.tool(
    name="top_k",
    description="Performs a simple vector search and returns the top 'k' "
                "most similar docs. This is the most basic form of "
                "Retrieval-Augmented Generation (RAG). "
                f"The collections available are {get_collections()} "
                f"and their respective descriptions are {get_metadatas()}.",
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "collection": {"type": "string"},
            "content": {"type": "array", "items": {"type": "string"}},
            "distances": {"type": "array", "items": {"type": "number"}},
            "parameters": {"type": "object"},
            "time": {"type": "number"}
        }
    }
)
def top_k(query: str,
          collection_name: str,
          k: int = 5) -> Dict[str, Any]:
  """
  Performs a simple vector search and returns the top 'k' most similar docs.
  This is the most basic form of Retrieval-Augmented Generation (RAG).
  Args:
    query (str): The user's query.
    collection_name (str): The name of an existing ChromaDB collection.
    k (int): The number of documents to return.
  Returns:
    Dict[str, Any]: A dictionary with 'collection', 'content', 'distances',
      'parameters', and 'query'.
  """
  try:
    collection = client.get_collection(name=collection_name)
  except Exception as e:
    raise ValueError(f"Collection '{collection_name}' not found: {e}")

  results = collection.query(
    query_texts=query,
    n_results=k,
    include=["documents", "distances"]
  )
  content = results["documents"][0]
  distances = results["distances"][0]
  result = {
    'query': query,
    'collection': collection_name,
    'content': content,
    'distances': distances,
    'parameters': {'k': k},
    'time':time.time()
  }
  save_result(result)
  return result

@mcp.tool(
    name="top_k_reranker",
    description="Implements the 'retrieve-then-rerank' strategy. First, it "
                "retrieves a larger number of documents (high_k), and then "
                "uses a more accurate reranker MODEL to find the top 5 "
                "best matches among them. "
                f"The collections available are {get_collections()} "
                f"and their respective descriptions are {get_metadatas()}.",
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "collection": {"type": "string"},
            "content": {"type": "array", "items": {"type": "string"}},
            "distances": {"type": "array", "items": {"type": "number"}},
            "parameters": {"type": "object"},
            "time": {"type": "number"}
        }
    }
)
def top_k_reranker(query: str,
              collection_name: str,
              high_k: int = 20) -> Dict[str, Any]:
  """
  Implements the 'retrieve-then-rerank' strategy.
  First, it retrieves a larger number of documents (high_k), and then uses a
  more accurate reranker MODEL to find the top 5 best matches among them.
  Args:
    query (str): The user's query.
    collection_name (str): The name of an existing ChromaDB collection.
    high_k (int): The initial number of documents to retrieve for reranking.
  Returns:
    Dict[str, Any]: A dictionary with 'collection', 'content', 'distances',
      'parameters', and 'query'.
  """
  try:
    collection = client.get_collection(name=collection_name)
  except Exception as e:
    raise ValueError(f"Collection '{collection_name}' not found: {e}")

  results = collection.query(
    query_texts=query,
    n_results=high_k,
    include=["documents"]
  )
  documents = results["documents"][0]
  docs, scores = rerank_documents(query, documents)
  result = {
    'query': query,
    'collection': collection_name,
    'content': docs,
    'distances': scores,  # Actually scores from reranker
    'parameters': {'high_k': high_k},
    'time':time.time()
  }
  save_result(result)
  return result

@mcp.tool(
    name="sentence_window_retriever_reranker",
    description="Implements the SWR (Sentence Window Retrieval) strategy, "
                "with ReRanker. This function finds the most relevant "
                "document and also retrieves the documents that were "
                "physically stored next to it (before and after), "
                "assuming they might contain relevant context. Its "
                "necessary to use higher variables with ReRanker. "
                f"The collections available are {get_collections()} "
                f"and their respective descriptions are {get_metadatas()}.",
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "collection": {"type": "string"},
            "content": {"type": "array", "items": {"type": "string"}},
            "distances": {"type": "array", "items": {"type": "number"}},
            "parameters": {"type": "object"},
            "time": {"type": "number"}
        }
    }
)
def sentence_window_retriever_reranker(query: str,
                                       collection_name: str,
                                       n_main: int = 3,
                                       n_around: int = 4) -> Dict[str, Any]:
  """
  Implements the SWR (Sentence Window Retrieval) strategy, with ReRanker.
  This function finds the most relevant document and also retrieves the
  documents that were physically stored next to it (before and after),
  assuming they might contain relevant context. Its necessary to use higher
  variables with ReRanker.
  Args:
    query (str): The user's query.
    collection_name (str): The name of an existing ChromaDB collection.
    n_main (int): The number of central documents to find.
    n_around (int): The number of neighboring documents to retrieve.
  Returns:
    Dict[str, Any]: A dictionary with 'collection', 'content', 'distances',
      'parameters', and 'query'.
  """
  try:
    collection = client.get_collection(name=collection_name)
  except Exception as e:
    raise ValueError(f"Collection '{collection_name}' not found: {e}")

  ids = collection.get()['ids']
  results = collection.query(query_texts=query, n_results=n_main)['ids']
  all_ids = []
  for i in results[0]:
    index = ids.index(i)
    indexes = ids[max(index-n_around,0):min(index+n_around+1,len(ids))]
    all_ids.extend(indexes)
  single_ids = list(set(all_ids))
  documents = collection.get(ids=single_ids)['documents'][0]
  docs, scores = rerank_documents(query, documents)
  result = {
    'query': query,
    'collection': collection_name,
    'content': docs,
    'distances': scores,
    'parameters': {'n_main': n_main, 'n_around': n_around},
    'time':time.time()
  }
  save_result(result)
  return result

@mcp.tool(
    name="multi_query_reranker",
    description="Generates multiple variations of a query using an LLM to "
                "broaden the search. This technique helps find documents "
                "that the original query might have missed, improving the "
                "chance of finding relevant information (recall). Now the "
                "final docs are passing through a ReRanker MODEL, its "
                "recommend higher returns to work better with ReRanker "
                "MODELs. "
                f"The collections available are {get_collections()} "
                f"and their respective descriptions are {get_metadatas()}.",
    output_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "collection": {"type": "string"},
            "content": {"type": "array", "items": {"type": "string"}},
            "distances": {"type": "array", "items": {"type": "number"}},
            "parameters": {"type": "object"},
            "time": {"type": "number"}
        }
    }
)
def multi_query_reranker(query: str,
                         collection_name: str,
                         n_results: int,
                         n_queries: int) -> Dict[str, Any]:
  """
  Generates multiple variations of a query using an LLM to broaden the search.
  This technique helps find documents that the original query might have missed,
  improving the chance of finding relevant information (recall). Now the final
  docs are passing through a ReRanker MODEL, its recommend higher returns to
  work better with ReRanker MODELs.
  Args:
    query (str): The user's original query.
    collection_name (str): The name of an existing ChromaDB collection.
    n_results (int): The number of results per query variation.
    n_queries (int): The number of query variations to generate.
  Returns:
    Dict[str, Any]: A dictionary with 'collection', 'content', 'distances',
      'parameters', and 'query'.
  """
  try:
    collection = client.get_collection(name=collection_name)
  except Exception as e:
    raise ValueError(f"Collection '{collection_name}' not found: {e}")

  questions = {}
  for n in range(n_queries):
    questions[f'question_{n}'] = ''
  messages = [
    SystemMessage(content=f'''You are a helpful assistant that will enhance my
RAG application. I'll give you one question and you will rewrite it in
{n_queries} different ways. You cannot use under any circumstances your
knowledge, just rewriting. You should follow strictly the following schema
{questions}, nothing less nothing more than that'''),
    HumanMessage(content=f'The question is: {query}')
  ]
  try:
    rewrite = eval(llm.invoke(messages).content)
  except Exception as e:
    raise ValueError(f"Failed to parse LLM response: {e}")

  all_ids = []
  for question in rewrite.keys():
    answer = collection.query(query_texts=[rewrite[question]],
                              n_results=n_results)
    ids = answer.get('ids',[[]])[0]
    all_ids.extend(ids)
  single_ids = list(set(all_ids))
  documents = collection.get(ids=single_ids)['documents'][0]
  docs, scores = rerank_documents(query, documents)
  result = {
    'query': query,
    'collection': collection_name,
    'content': docs,
    'distances': scores,
    'parameters': {'n_queries': n_queries, 'n_results': n_results},
    'time':time.time()
  }
  save_result(result)
  return result

if __name__ == "__main__":
  mcp.run(transport='sse',host = '0.0.0.0',port = 11000)
