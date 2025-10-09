import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
import nltk
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")

class Splitter():
  """A class that has text splitting functions"""
  def __init__(self):
    pass
  def equal_chunks(self,file_path:str,
                   chunck_size:int,
                   chunk_overlap:int) -> list[str]:
    """Extract text from a PDF file and split it into equal-sized chunks.
    Args:
        file_path (str): The path to the PDF file to process.
    Returns:
        list[str]: A list of text chunks.
    """
    text = ""
    with open(file_path, "rb") as file:
      reader = PyPDF2.PdfReader(file)
      for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
          text += page_text
      text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = chunck_size,
      chunk_overlap = chunk_overlap,
      length_function = len,
      is_separator_regex= False,
      )
    chunk = text_splitter.create_documents([text])
    documents = []
    for c in chunk:
      documents.append(c.page_content)
    return documents

  # Function to return chunks usign usntructured library
  def unstructured_chunks(self,file_path:str)->list[str]:
    """Extract and chunk text from a PDF file using the unstructured library.
    Args:
        file_path (str): The path to the PDF file to process.
    Returns:
        list[str]: A list of text chunks extracted and chunked by title.
    """
    raw_chunks = partition_pdf(
    filename= file_path,
    strategy="hi_res",
    extract_images_in_pdf= False,
    chunking_strategy = "by_title",
    )
    documents = []
    for c in raw_chunks:
      documents.append(c.text)
      return documents
  # Functions to return semantic chunks
  def simple_decision(self,
                      start_limit:float,
                      y:float,
                      sentences:list[str],
                      all_embeddings)->list[str]:
    """Groups sentences using a linearly increasing similarity threshold.
      Args:
          start_limit (float): The initial similarity threshold (e.g., 0.7).
          y (float): Amount to increase the threshold after each addition.
          sentences (list[str]): The list of sentences to group.
          all_embeddings: The embedding of each senetence
      Returns:
          list[str]: A list of cohesive text chunks.
      """
    chunks = []
    i = 0
    while i < (len(sentences)):
      chunk_raw = sentences[i]
      embedding_frase = all_embeddings[i]
      current_limit = start_limit
      counter = 1
      while True:
        following = i + counter
        if following > len(sentences)-1:
          break
        embedding_next = all_embeddings[following]
        similaridade = cosine_similarity([embedding_frase], [embedding_next])
        if similaridade >= current_limit :
          chunk_raw += " " + sentences[following]
          counter += 1
          current_limit = current_limit + y
        else:
          break
      chunks.append(chunk_raw)
      i = i + counter
    return chunks

  def changing_decision(self,
                        start_limit:float,
                        y:float,
                        sentences:list[str],
                        all_embeddings) -> list[str]:
    """Groups sentences using an exponentially increasing similarity threshold.
      Args:
          start_limit (float): The initial similarity threshold (e.g., 0.7).
          y (float): Growth rate (0-1). A smaller value means faster
              growth; a larger value means slower growth.
          sentences (list[str]): The list of sentences to group.
          all_embeddings: embedding for each sentece
      Returns:
          list[str]: A list of cohesive text chunks.
      """
    chunks = []
    i = 0
    while i < (len(sentences)):
      chunk_raw = sentences[i]
      embedding_frase = all_embeddings[i]
      counter = 1
      current_limit = start_limit
      limit = 1 - start_limit
      while True:
        following = i + counter
        if following > len(sentences)-1:
          break
        embedding_next = all_embeddings[following]
        similaridade = cosine_similarity([embedding_frase], [embedding_next])
        x = counter
        if similaridade >= current_limit :
          chunk_raw += " " + sentences[following]
          counter += 1
          current_limit = start_limit + (limit*(1-y**x))
        else:
          break
      chunks.append(chunk_raw)
      i = i + counter
    return chunks
