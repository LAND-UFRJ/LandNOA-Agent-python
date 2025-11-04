import streamlit as st
import utils.chroma_functions as cf
from utils.indexing import Splitter
from utils.retrieval import Retriever
import os
import inspect
import nltk
import json 

nltk.download("punkt")

# --- 1. Page Configuration ---
st.set_page_config(page_title="Chroma Manager", layout="wide")
st.title("🤖 Document Manager for ChromaDB")
st.write("Upload, process, and add PDF documents to your ChromaDB collections.")

# --- 2. Starting Services ---
@st.cache_resource
def start_services():
  """Connects to ChromaDB and loads the Splitter and Retriever classes."""
  try:
    client = cf.connect_chroma()
    splitter = Splitter()
    retriever = Retriever()
    return client, splitter, retriever
  except Exception as e:
    st.error(f"Could not initialize services. Please check ChromaDB connection. Error: {e}")
    return None, None, None

client, splitter, retriever = start_services() 
if client is None:
  st.stop()

# --- 3. Method Dictionaries ---

# Index Methods
METHOD_NAMES = {
  "equal_chunks": "Fixed Size (Langchain)",
  "unstructured_chunks": "Structured (Unstructured)",
  "simple_decision": "Semantic (Linear)",
  "changing_decision": "Semantic (Exponential)"
}
available_methods = [
  name for name, func in inspect.getmembers(splitter, predicate=inspect.ismethod)
  if name in METHOD_NAMES and not name.startswith('_')
]

# Retrieval Methods
RETRIEVAL_METHOD_NAMES = {
  "top_k": "Top-K",
  "re_ranker": "Top-K (with Re-Ranker)",
  "multi_query": "Multi-Query",
  "multi_query_self.re_ranker": "Multi-Query (with Re-Ranker)",
  "sentence_window_retrieval": "Sentence Window",
  "sentence_window_retriever_self.re_ranker": "Sentence Window (with Re-Ranker)"
}
available_retrievers = [
  name for name, func in inspect.getmembers(retriever, predicate=inspect.ismethod)
  if name in RETRIEVAL_METHOD_NAMES and not name.startswith('_')
]

# --- 4. Sidebar for Collection Management ---
st.sidebar.header("Collection Management")
try:
  collection_list = cf.list_all_collections(client)
except Exception as e:
  st.sidebar.error("Error listing collections.")
  collection_list = []

st.sidebar.subheader("Select or Create")
option = st.sidebar.radio("What would you like to do?",("Use existing collection", "Create a new collection"))

active_collection_name = None
if option == "Use existing collection":
  if collection_list:
    active_collection_name = st.sidebar.selectbox("Available collections:", options=collection_list)
  else:
    st.sidebar.warning("No collections found. Please create one.")
else:
  new_collection_name = st.sidebar.text_input("New collection name:")
  
  friendly_names = [METHOD_NAMES.get(method, method) for method in available_methods]
  selected_method_friendly = st.sidebar.radio(
    "Choose a processing method:",
    options=friendly_names
  )
  technical_method = next(key for key, value in METHOD_NAMES.items() if value == selected_method_friendly)

  params = {}
  st.sidebar.caption("Method Parameters")
  
  if technical_method == "equal_chunks":
    params['chunck_size'] = st.sidebar.slider("Chunk Size:", 200, 2000, 750)
    params['chunk_overlap'] = st.sidebar.slider("Chunk Overlap:", 0, 500, 50)
  
  elif technical_method in ["simple_decision", "changing_decision"]:
    params['start_limit'] = st.sidebar.slider("Initial Similarity:", 0.0, 1.0, 0.7, 0.01)
    params['y'] = st.sidebar.slider("Growth Factor (y):", 0.0, 1.0, 0.5, 0.01)

  elif technical_method == "unstructured_chunks":
    st.sidebar.text("No parameters for this method.")

  if st.sidebar.button("Create Collection"):
    if new_collection_name and new_collection_name not in collection_list:
      try:
        cf.create_collection_with_info(
          client=client,
          collection_name=new_collection_name,
          index_method=technical_method,
          parameters=params 
        )
        st.sidebar.success(f"Collection '{new_collection_name}' created!")
        st.rerun() 
      except Exception as e:
        st.sidebar.error(f"Creation failed: {e}")
        st.exception(e) 
    elif new_collection_name in collection_list:
      st.sidebar.error("A collection with this name already exists.")
    else:
      st.sidebar.error("Collection name cannot be empty.")
  
  if new_collection_name:
    active_collection_name = new_collection_name

st.sidebar.subheader("Delete Collection")
if collection_list:
  collection_to_delete = st.sidebar.selectbox("Select to delete:", options=[""] + collection_list)
  if collection_to_delete and st.sidebar.button(f"Delete '{collection_to_delete}'"):
    try:
      client.delete_collection(name=collection_to_delete)
      st.sidebar.success(f"Collection '{collection_to_delete}' deleted.")
      st.rerun()
    except Exception as e:
      st.sidebar.error(f"Deletion failed: {e}")

# --- 5. Main Area---
if not active_collection_name:
  st.info("Please select or create a collection in the sidebar to continue.")
else:
  st.success(f"Working on collection: **{active_collection_name}**")

  try:
    unique_pdfs, infos = cf.show_collection_info(client, active_collection_name)
    saved_method = infos.get("index_method")
    saved_params_str = infos.get("parameters", "{}") 
    saved_params = json.loads(saved_params_str)

    st.subheader("Collection Info")
    col1, col2 = st.columns(2)
    with col1:
      st.markdown(f"**Processing Method:**")
      st.code(METHOD_NAMES.get(saved_method, "Unknown"))
      st.markdown(f"**Saved Parameters:**")
      if saved_params:
        st.json(saved_params)
      else:
        st.text("None")
    with col2:
      st.markdown(f"**Documents in Collection ({len(unique_pdfs)}):**")
      if unique_pdfs:
        pdf_names = [data[0] for data in unique_pdfs if isinstance(data, tuple) and data]
        st.dataframe(pdf_names, use_container_width=True, hide_index=True, column_config={"value":"File Name"})
      else:
        st.text("No documents added yet.")
  except Exception as e:
    st.error(f"Could not load collection info. Error: {e}")
    st.stop()

  st.divider()

  # --- Tabs Management ---
  tab1, tab2 = st.tabs(["🗂️ Manage Documents", "🔍 Query Collection"])

  # Tab 1
  with tab1:
    st.header(f"➕ Add New Documents to '{active_collection_name}'")
    uploaded_files = st.file_uploader("Drag and drop PDF files here", type="pdf", accept_multiple_files=True)

    if st.button("Process and Add to Collection", type="primary"):
      if uploaded_files and active_collection_name:
        try:
          collection = cf.get_collection(client, active_collection_name)
          progress_bar = st.progress(0, text="Starting process...")

          for i, file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress, text=f"Processing: {file.name}")
            temp_path = os.path.join("/tmp", file.name)
            with open(temp_path, "wb") as f:
              f.write(file.getbuffer())

            processing_function = getattr(splitter, saved_method)
            all_params = saved_params.copy()
            all_params['file_path'] = temp_path
            documents = processing_function(**all_params)

            if documents:
              result = cf.add_documents(collection, documents, file.name)
              if result is None:
                st.warning(f"  > File '{file.name}' already exists in this collection. Skipped.")
              else:
                st.write(f"  > File '{file.name}' processed and added successfully.")
            else:
              st.warning(f"No text extracted from '{file.name}'.")
            os.remove(temp_path)

          progress_bar.progress(1.0, text="Process complete!")
          st.success("All files were processed!")
          st.rerun()

        except Exception as e:
          st.error(f"An error occurred during processing: {e}")
          st.exception(e)
      elif not uploaded_files:
        st.warning("Please upload at least one file.")

  # Tab 2
  with tab2:
    st.header(f"❓ Query '{active_collection_name}'")

    retrieval_friendly_names = [RETRIEVAL_METHOD_NAMES.get(method, method) for method in available_retrievers]
    selected_retrieval_friendly = st.radio(
      "Choose a retrieval method:",
      options=retrieval_friendly_names
    )
    technical_retrieval_method = next(key for key, value in RETRIEVAL_METHOD_NAMES.items() if value == selected_retrieval_friendly)

    retrieval_params = {}
    st.caption("Retrieval Parameters")

    if technical_retrieval_method == "top_k":
      retrieval_params['k'] = st.number_input("K (documents to return)", min_value=1, max_value=50, value=5, step=1)
    
    elif technical_retrieval_method == "re_ranker":
      retrieval_params['high_k'] = st.number_input("High K (docs to retrieve for reranking)", min_value=5, max_value=100, value=20, step=1)

    elif technical_retrieval_method == "multi_query":
      retrieval_params['n_queries'] = st.number_input("Number of Queries (variations)", min_value=2, max_value=10, value=3, step=1)

    elif technical_retrieval_method == "multi_query_self.re_ranker":
      retrieval_params['n_queries'] = st.number_input("Number of Queries (variations)", min_value=2, max_value=10, value=5, step=1) # Default maior p/ self.re_ranker

    elif technical_retrieval_method == "sentence_window_retrieval":
      retrieval_params['n_main'] = st.number_input("N Main (central docs)", min_value=1, max_value=10, value=1, step=1)
      retrieval_params['n_around'] = st.number_input("N Around (neighbor docs)", min_value=1, max_value=10, value=3, step=1)

    elif technical_retrieval_method == "sentence_window_retriever_self.re_ranker":
      retrieval_params['n_main'] = st.number_input("N Main (central docs)", min_value=1, max_value=10, value=3, step=1)
      retrieval_params['n_around'] = st.number_input("N Around (neighbor docs)", min_value=1, max_value=10, value=4, step=1)

 

    query = st.text_input("Your question:", key="query_input")

    if st.button("Run Query", type="primary"):
      if query and active_collection_name:
        try:

          retrieval_function = getattr(retriever, technical_retrieval_method)
          
          all_retrieval_params = retrieval_params.copy()
          all_retrieval_params['query'] = query
          all_retrieval_params['collection_name'] = active_collection_name
          
      
          with st.spinner(f"Running '{selected_retrieval_friendly}'..."):
            result_prompt = retrieval_function(**all_retrieval_params)
          
          st.subheader("Retrieval Results:")
          st.markdown(result_prompt)

        except Exception as e:
          st.error(f"An error occurred during retrieval: {e}")
          st.exception(e)
      else:
        st.warning("Please enter a question.")
