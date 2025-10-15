import os
import chromadb
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

openai_url = os.getenv('OPENAI_BASE_URL')
openai_key = os.getenv('OPENAI_KEY')
chroma_url = os.getenv('CHROMA_URL')
chorma_port = int(os.getenv('CHROMADB_PORT'))
model = os.getenv('OPENAI_MODEL')

client = chromadb.HttpClient(host=chroma_url,port=chorma_port)

llm = ChatOpenAI(base_url=openai_url,model=model,api_key='')

class Retriever():
  """Class that has the rertieval functions"""
  def __init__(self):
    pass
  def sentence_window_retrieval(self,
                                query:str,
                                collection_name:str,
                                n_main:int=1,
                                n_around:int =3):

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
      prompt (str)
        """

    collection = client.get_collection(name = collection_name)
    ids = collection.get()['ids']
    final = []
    results = collection.query(query_texts=query, n_results= n_main )['ids']
    for i in results[0]:
      index = ids.index(i)
      indexes = ids[max(index-n_around,0):min(index+n_around,len(ids))]
      for x in indexes:
        final.append(collection.get(ids=[x])['documents'])
      prompt = 'Here are the RAG results:\n\n'
      for string in final:
        prompt += string[0] +'\n'
        prompt += '----------------------------------\n'
    return prompt

  def multi_query(self,
                  query:str,
                  collection_name:str,
                  n_queries:int):
    """
    Generates multiple variations of a query using an LLM to broaden the search.

    This technique helps find documents that
    the original query might have missed,
    improving the chance of finding relevant information (recall).

    Args:
      query (str): The user's original query.
      collection_name (str): The name of an existing ChromaDB collection.
      n_queries (int): The number of query variations to generate.
    """
    collection = client.get_collection(name=collection_name)
    questions = {}
    for n in range(n_queries):
      questions[f'question_{n}'] = ''
    messages = [SystemMessage(content=f'''You are a helpfull asistant that will
    enhance my RAG application. I'll give you one question and you will
    rewrite it in {n_queries} diferent ways. You cannot use under any
    cisrcunstances your knowladge, just rewriteing. You should follow
    strictlly the following schema {questions}, nothing less nothing more than
    that'''), HumanMessage(content=f'The question is: {query}')]
    prompt = '''The RAG results are the following.
    The lower the distance, the best the match is:\n'''
    rewrite = eval(llm.invoke(messages).content)
    for question in rewrite.keys():
      answer = collection.query(query_texts=[rewrite[question]], n_results=3)
      distances = answer.get('distances', [[]])[0]
      documents = answer.get('documents', [[]])[0]
      for dist, doc in zip(distances, documents):
        prompt += f'Score: {dist}\nDocuments: {doc}\n------------------------\n'
    return prompt

  def top_k(self,
            query:str,
            collection_name:str,
            k:int=5):
    """
    Performs a simple vector search and returns the top 'k' most similar docs.

    This is the most basic form of Retrieval-Augmented Generation (RAG).

    Args:
      query (str): The user's query.
      collection_name (str): The name of an existing ChromaDB collection.
      k (int): The number of documents to return.
    """
    collection = client.get_collection(name=collection_name)
    results = collection.query(
      query_texts=query,
      n_results= k,
      include = ["documents"]
      )
    rag = results["documents"][0]
    prompt  = f"Here are the RAG results in order to best to worst: {rag}"
    return prompt
  #Functions With Re_Ranker

  def re_ranker(self,
                query:str,
                collection_name:str,
                high_k:int=20):
    """
    Implements the 'retrieve-then-rerank' strategy.

    First, it retrieves a larger number of documents (high_k), and then uses a
    more accurate reranker model to find the top 5 best matches among them.

    Args:
      query (str): The user's query.
      collection_name (str): The name of an existing ChromaDB collection.
      high_k (int): The initial number of documents to retrieve for reranking.
    """
    collection = client.get_collection(name=collection_name)
    reranker = FlagReranker('BAAI/bge-reranker-base',
                          use_fp16=True,
                          normalize=True)
    results = collection.query(
      query_texts=query,
      n_results= high_k,
      include = ["documents"]
      )
    rag = results["documents"][0]
    touples  = [[query,d]for d in rag]
    scores = reranker.compute_score(touples)
    indices_maiores = sorted(range(len(scores)),
                              key=lambda i: scores[i],
                              reverse=True)[:5]
    docs = []
    for i in indices_maiores:
      docs.append(rag[i])
    prompt  = f"Here are the RAG results in order to best to worst: {docs}"
    return prompt

  def sentence_window_retriever_reranker(self,
                                         query:str,
                                         collection_name:str,
                                         n_main:int=3,
                                         n_around:int =4):
    """
    Implements the SWR (Sentence Window Retrieval) strategy, with ReRanker
    This function finds the most relevant document and also retrieves the
    documents that were physically stored next to it (before and after),
    assuming they might contain relevant context.
    Its necessary to use higher variabes with ReRanker.
    Args:
      query (str): The user's query.
      collection_name (str): The name of an existing ChromaDB collection.
      n_main (int): The number of central documents to find.
      n_around (int): The number of neighboring documents to retrieve.
    """
    collection = client.get_collection(name = collection_name)
    reranker = FlagReranker('BAAI/bge-reranker-base',
                          use_fp16=True,
                          normalize=True)
    ids = collection.get()['ids']
    final = []
    results = collection.query(query_texts=query, n_results= n_main )['ids']
    all_ids = []
    for i in results[0]:
      index = ids.index(i)
      indexes = ids[max(index-n_around,0):min(index+n_around,len(ids))]
      all_ids.extend(indexes)
    print(f"ALL_IDS HERE:{all_ids}")
    single_ids = list(set(all_ids))
    print(f"ALL_SINGLE IDS HERE:{single_ids}")
    final.append(collection.get(ids=single_ids)['documents'])
    print(final[0])
    touples = [[query,d]for d in final[0]]
    scores = reranker.compute_score(touples)
    indices_maiores = sorted(range(len(scores)),
                              key=lambda i: scores[i],
                              reverse=True)[:5]
    print(indices_maiores)
    print(scores)
    rag = []
    for i in indices_maiores:
      rag.append(final[0][i])
    prompt = f'Here are the RAG results:{rag}'
    return prompt

  def multi_query_reranker(self,
                           query:str,
                           collection_name:str,
                           n_queries:int):
    """
    Generates multiple variations of a query using an LLM to broaden the search.
    This technique helps find documents that 
    the original query might have missed,
    improving the chance of finding relevant information (recall).
    Now the final docs are passing trough a ReRanker model, its recommend higher
    returns to work better with ReRanker models
    Args:
      query (str): The user's original query.
      collection_name (str): The name of an existing ChromaDB collection.
      n_queries (int): The number of query variations to generate.
    """
    reranker = FlagReranker('BAAI/bge-reranker-base',
                          use_fp16=True,
                          normalize=True)
    collection = client.get_collection(name=collection_name)
    questions = {}
    for n in range(n_queries):
      questions[f'question_{n}'] = ''
    messages = [SystemMessage(content=f'''You are a helpfull asistant that will
    enhance my RAG application. I'll give you one question and you will
    rewrite it in {n_queries} diferent ways. You cannot use under any
    cisrcunstances your knowladge, just rewriteing. You should follow
    strictlly the following schema {questions}, nothing less nothing more than
    that'''), HumanMessage(content=f'The question is: {query}')]
    prompt = '''The RAG results are the following.
    The lower the distance, the best the match is:\n'''
    rewrite = eval(llm.invoke(messages).content)
    all_ids = []
    for question in rewrite.keys():
      answer = collection.query(query_texts=[rewrite[question]], n_results=10)
      ids = answer.get('ids',[[]])[0]
      all_ids.extend(ids)
    single_ids = set(all_ids)
    documents = collection.get(ids=single_ids)['documents']
    touples = [[query,d]for d in documents[0]]
    scores = reranker.compute_score(touples)
    indices_maiores = sorted(range(len(scores)),
                              key=lambda i: scores[i],
                              reverse=True)[:5]
    rag = []
    for i in indices_maiores:
      rag.append(documents[0][i])
    prompt = f"Here are the RAG documents:{rag}"
    return prompt
