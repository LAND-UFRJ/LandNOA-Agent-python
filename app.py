import streamlit as st
import chroma_functions as cf
from indexing import Splitter
import os
import inspect
import nltk
from sentence_transformers import SentenceTransformer
import PyPDF2

# --- 1. Page Configuration ---
st.set_page_config(page_title="Chroma Manager", layout="wide")

st.title("🤖 Document Manager for ChromaDB")
st.write("Upload, process, and add PDF documents to your ChromaDB collections.")
nltk.download("punkt")
   

# --- 2. Starting Services ---
@st.cache_resource
def start_services():
    """Connects to ChromaDB and loads the necessary models and classes."""
    try:
        client = cf.connect_chroma()
        splitter = Splitter()
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return client, splitter, embedding_model
    except Exception as e:
        st.error(f"Could not initialize services. Please check ChromaDB connection. Error: {e}")
        return None, None, None

client, splitter, embedding_model = start_services()

if client is None:
    st.stop()

# --- 3. Dynamic Method Discovery ---
# Maps technical function names to user-friendly names for the UI
METHOD_NAMES = {
    "equal_chunks": "Fixed Size (Langchain)",
    "unstructured_chunks": "Structured (Unstructured)",
    "simple_decision": "Semantic (Linear)",
    "changing_decision": "Semantic (Exponential)"
}

# Discovers available methods from the Splitter class
available_methods = [
    name for name, func in inspect.getmembers(splitter, predicate=inspect.ismethod)
    if name in METHOD_NAMES and not name.startswith('_')
]

# --- 4. Sidebar for Collection Management ---
st.sidebar.header("🗂️ Collection Management")
try:
    # Corrigido para usar 'cf.list_all_collections'
    collection_list = cf.list_all_collections(client)
except Exception as e:
    st.sidebar.error("Error listing collections.")
    collection_list = []

st.sidebar.subheader("Select or Create")
option = st.sidebar.radio("What would you like to do?", ("Use existing collection", "Create a new collection"))

active_collection_name = None
if option == "Use existing collection":
    if collection_list:
        active_collection_name = st.sidebar.selectbox("Available collections:", options=collection_list)
    else:
        st.sidebar.warning("No collections found. Please create one.")
else:
    new_collection_name = st.sidebar.text_input("New collection name:")
    if st.sidebar.button("Create Collection"):
        if new_collection_name and new_collection_name not in collection_list:
            try:
                # Corrigido para usar 'cf.create_collection'
                cf.create_collection(client, new_collection_name)
                st.sidebar.success(f"Collection '{new_collection_name}' created!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Creation failed: {e}")
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

# --- 5. Main Area for Upload and Processing ---
st.header("📄 Upload & Processing")

if not active_collection_name:
    st.info("Please select or create a collection in the sidebar to continue.")
else:
    st.success(f"Working on collection: **{active_collection_name}**")

    friendly_names = [METHOD_NAMES[method] for method in available_methods]
    selected_method_friendly = st.radio(
        "Choose a processing method:",
        options=friendly_names,
        horizontal=True
    )
    technical_method = next(key for key, value in METHOD_NAMES.items() if value == selected_method_friendly)

    # --- Dynamic parameters based on the chosen method ---
    params = {}
    st.subheader("Method Parameters")
    if technical_method == "equal_chunks":
        params['chunck_size'] = st.slider("Chunk Size:", 200, 2000, 750)
        params['chunk_overlap'] = st.slider("Chunk Overlap:", 0, 500, 50)
    elif technical_method in ["simple_decision", "changing_decision"]:
        params['start_limit'] = st.slider("Initial Similarity:", 0.5, 1.0, 0.7, 0.01)
        params['y'] = st.slider("Growth Factor (y):", 0.0, 1.0, 0.5, 0.01)

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

                    processing_function = getattr(splitter, technical_method)
                    documents = []

                    if technical_method in ["simple_decision", "changing_decision"]:
                        text = ""
                        with open(temp_path, "rb") as pdf_file:
                            reader = PyPDF2.PdfReader(pdf_file)
                            for page in reader.pages:
                                page_text = page.extract_text()
                                if page_text: text += page_text

                        sentences = nltk.sent_tokenize(text)
                        if sentences:
                            params['sentences'] = sentences
                            params['all_embeddings'] = embedding_model.encode(sentences)
                            documents = processing_function(**params)
                    else:
                        params['file_path'] = temp_path
                        documents = processing_function(**params)

                    if documents:
                        # Corrigido para usar 'cf.add_documents'
                        cf.add_documents(collection, documents, file.name)
                        st.write(f"  > File '{file.name}' processed and added successfully.")
                    else:
                        st.warning(f"No text extracted from '{file.name}'.")

                    os.remove(temp_path)

                progress_bar.progress(1.0, text="Process complete!")
                st.success("All files were processed and added to the collection!")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)
        elif not uploaded_files:
            st.warning("Please upload at least one file.")