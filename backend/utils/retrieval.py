import os
import time
from typing import Dict, List, Any
from pathlib import Path
from .sqlite_functions import get_config
import chromadb
from FlagEmbedding import FlagReranker
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

OPENAI_URL = get_config('openai_baseurl')
OPENAI_KEY = get_config('openai_api_key')
CHROMA_URL = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT"))
MODEL = get_config('model')

client = chromadb.HttpClient(host=CHROMA_URL,port=CHROMA_PORT)

llm = ChatOpenAI(base_url=OPENAI_URL,MODEL=MODEL,api_key=OPENAI_KEY)

class Retriever():
  """Class that has the rertieval functions"""
  def __init__(self):
    self.re_ranker = None

  def get_reranker(self):
    """gets the reranker"""
    if self.re_ranker is None:
      self.re_ranker = FlagReranker('BAAI/bge-reranker-base',
                              use_fp16=True,
                              normalize=True)
    return self.re_ranker

  def rerank_documents(self,
                      query: str,
                      documents: List[str],
                      top_r: int =None) -> tuple[List[str], List[float]]:
    """Helper function to rerank documents using the reranker model."""
    tuples = [[query, d] for d in documents]
    scores = self.re_ranker.compute_score(tuples)
    if top_r:
      indices = sorted(range(len(scores)),
                      key=lambda i: scores[i],
                      reverse=True)[:top_r]
      return [documents[i] for i in indices], [scores[i] for i in indices]
    else:
      indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
      return [documents[i] for i in indices], [scores[i] for i in indices]

  def sentence_window_retrieval(self,
                                query: str,
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
    return result

  def multi_query(self,
                  query: str,
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
    return result


  def top_k(self,
            query: str,
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
    return result


  def top_k_reranker(self,
                     query: str,
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
    docs, scores = self.rerank_documents(query, documents)
    result = {
      'query': query,
      'collection': collection_name,
      'content': docs,
      'distances': scores,  # Actually scores from reranker
      'parameters': {'high_k': high_k},
      'time':time.time()
    }
    return result


  def sentence_window_retriever_reranker(self,
                                         query: str,
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
    docs, scores = self.rerank_documents(query, documents)
    result = {
      'query': query,
      'collection': collection_name,
      'content': docs,
      'distances': scores,
      'parameters': {'n_main': n_main, 'n_around': n_around},
      'time':time.time()
    }
    return result


  def multi_query_reranker(self,
                           query: str,
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
    docs, scores = self.rerank_documents(query, documents)
    result = {
      'query': query,
      'collection': collection_name,
      'content': docs,
      'distances': scores,
      'parameters': {'n_queries': n_queries, 'n_results': n_results},
      'time':time.time()
    }
    return result
