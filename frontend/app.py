import streamlit as st
import backend.utils.chroma_functions as cf
import os
import inspect
import nltk
import backend.utils.sqlite_functions as sq 
from backend.utils.indexing import Splitter

try:
    db_conn = sq.connect()
    sq.create_tables(db_conn) 
except Exception as e:
    st.error(f"Falha ao conectar/criar tabelas no SQLite: {e}")
    st.stop()


nltk.download("punkt")

# --- 1. Page Configuration ---
st.set_page_config(page_title="Chroma Manager", layout="wide")
st.title("🤖 Document Manager for ChromaDB")
st.write("Upload, process, and add PDF documents to your ChromaDB collections.")

# --- 2. Starting Services ---
@st.cache_resource
def start_services():
  """Connects to ChromaDB and loads the Splitter class."""
  try:
    client = cf.connect_chroma()
    splitter = Splitter()
    return client, splitter, db_conn
  except Exception as e:
    st.error(f"Could not initialize services. Please check ChromaDB connection. Error: {e}")
    return None, None, None

client, splitter, sqlite = start_services()
if client is None:
  st.stop()

# --- 3. Method Dictionaries ---
# (Nenhuma mudança aqui, está perfeito)
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


# --- 4. Sidebar for Collection Management ---
st.sidebar.header("Collection Management")
try:
  # MUDANÇA: Lista de coleções vem do SQLite
  collection_list = sq.get_all_collections_names(sqlite) 
except Exception as e:
  st.sidebar.error(f"Error listing collections: {e}")
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
        # MUDANÇA: Orquestração da criação
        # 1. Adiciona no SQLite
        sq.add_collection(
            conn=sqlite,
            name=new_collection_name,
            index_method=technical_method,
            index_params=params
        )
        # 2. Cria no Chroma
        cf.create_collection(
            client=client,
            collection_name=new_collection_name
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
      # MUDANÇA: Orquestração da deleção
      # 1. Deleta do Chroma
      cf.delete_collection(client, collection_to_delete)
      # 2. Deleta do SQLite
      sq.delete_collection(sqlite, collection_to_delete)
      
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
    # MUDANÇA: Pega informações do SQLite
    collection_details = sq.get_collection_details(sqlite, active_collection_name)
    pdf_names = sq.get_pdfs_for_collection(sqlite, active_collection_name)

    if not collection_details:
        st.error(f"Detalhes da coleção '{active_collection_name}' não encontrados no SQLite. Pode haver uma dessincronização.")
        st.stop()

    saved_method = collection_details.get("index_method")
    saved_params = collection_details.get("index_params", {}) # Já vem como dict

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
      st.markdown(f"**Documents in Collection ({len(pdf_names)}):**")
      if pdf_names:
        st.dataframe(pdf_names, use_container_width=True, hide_index=True, column_config={"value":"File Name"})
      else:
        st.text("No documents added yet.")
  except Exception as e:
    st.error(f"Could not load collection info. Error: {e}")
    st.stop()

  st.divider()

  # --- Adicionar Documentos ---
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
          
          # MUDANÇA: Verificação de duplicata no SQLite
          if sq.check_pdf_exists(sqlite, active_collection_name, file.name):
              st.warning(f"  > File '{file.name}' already exists in this collection (SQLite). Skipped.")
              continue

          temp_path = os.path.join("/tmp", file.name)
          with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

          processing_function = getattr(splitter, saved_method)
          all_params = saved_params.copy()
          all_params['file_path'] = temp_path
          documents = processing_function(**all_params)

          if documents:
            # 1. Adiciona no Chroma
            cf.add_documents(collection, documents, file.name)
            # 2. Registra no SQLite
            sq.add_pdf_to_collection(sqlite, active_collection_name, file.name)
            
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