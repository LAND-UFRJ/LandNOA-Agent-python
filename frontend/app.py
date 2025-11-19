import streamlit as st
import backend.utils.chroma_functions as cf
from backend.utils.indexing import Splitter
from backend.utils.retrieval import Retriever
import os
import inspect
import nltk
import backend.utils.sqlite_functions as sq
from dotenv import load_dotenv
import requests
import json

nltk.download("punkt")

# --- Page Configuration ---
st.set_page_config(page_title="LandNOA Manager", layout="wide")

# Título e Menu Superior (Funciona como Abas, mas atualiza a sidebar)
st.title("🤖 LandNOA Agent Manager")

# --- MENU SUPERIOR (TIPO ABAS) ---
# horizontal=True faz parecer abas no topo
page_selection = st.radio(
    "Navegação", 
    ["🧠 RAG System", "⚙️ Agent Configuration"], 
    horizontal=True,
    label_visibility="collapsed" # Esconde o label para ficar mais limpo
)
st.divider()

# --- Helper Functions ---
def get_models(base_url: str):
    try:
        r = requests.get(f"{base_url}/models", timeout=2)
        return [m["id"] for m in r.json().get("data", [])]
    except:
        return []

# --- Starting Services ---
@st.cache_resource
def start_services():
    try:
        client = cf.connect_chroma()
        splitter = Splitter()
        retriever = Retriever()
        return client, splitter, retriever
    except Exception as e:
        st.error(f"Could not initialize services. Error: {e}")
        return None, None, None

client, splitter, retriever = start_services() 
if client is None:
    st.stop()

# --- Method Dictionaries ---
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

RETRIEVAL_METHOD_NAMES = {
    "top_k": "Top-K",
    "top_k_reranker": "Top-K (with Re-Ranker)",
    "multi_query": "Multi-Query",
    "multi_query_reranker": "Multi-Query (with Re-Ranker)",
    "sentence_window_retrieval": "Sentence Window",
    "sentence_window_retriever_reranker": "Sentence Window (with Re-Ranker)"
}
available_retrievers = [
    name for name, func in inspect.getmembers(retriever, predicate=inspect.ismethod)
    if name in RETRIEVAL_METHOD_NAMES and not name.startswith('_')
]

# ==============================================================================
# LOGIC FOR PAGE 1: RAG SYSTEM
# ==============================================================================
if page_selection == "🧠 RAG System":
    
    # --- Sidebar Specific to RAG (Só aparece nesta aba) ---
    st.sidebar.header("📂 Collection Management")
    try:
        collection_list = sq.list_collections_sqlite() 
    except Exception as e:
        st.sidebar.error(f"Error listing collections: {e}")
        collection_list = []

    st.sidebar.subheader("Select or Create")
    # Usando selectbox na sidebar (mais limpo que radio vertical)
    option = st.sidebar.selectbox("Action:",("Use existing collection", "Create a new collection"))

    active_collection_name = None
    
    if option == "Use existing collection":
        if collection_list:
            active_collection_name = st.sidebar.selectbox("Available collections:", options=collection_list)
        else:
            st.sidebar.warning("No collections found.")
    else:
        new_collection_name = st.sidebar.text_input("New collection name:")
        
        friendly_names = [METHOD_NAMES.get(method, method) for method in available_methods]
        selected_method_friendly = st.sidebar.selectbox("Processing method:", options=friendly_names)
        technical_method = next(key for key, value in METHOD_NAMES.items() if value == selected_method_friendly)

        params = {}
        st.sidebar.caption("Method Parameters")
        
        if technical_method == "equal_chunks":
            params['chunck_size'] = st.sidebar.slider("Chunk Size:", 200, 2000, 750)
            params['chunk_overlap'] = st.sidebar.slider("Chunk Overlap:", 0, 500, 50)
        elif technical_method in ["simple_decision", "changing_decision"]:
            params['start_limit'] = st.sidebar.slider("Initial Similarity:", 0.0, 1.0, 0.7, 0.01)
            params['y'] = st.sidebar.slider("Growth Factor (y):", 0.0, 1.0, 0.5, 0.01)

        if st.sidebar.button("Create Collection"):
            if new_collection_name and new_collection_name not in collection_list:
                try:
                    sq.create_collection_sqlite(new_collection_name, technical_method, params)
                    cf.create_collection(client, new_collection_name)
                    st.sidebar.success(f"Collection '{new_collection_name}' created!")
                    st.rerun() 
                except Exception as e:
                    st.sidebar.error(f"Creation failed: {e}")
            elif new_collection_name in collection_list:
                st.sidebar.error("Name already exists.")
            else:
                st.sidebar.error("Name cannot be empty.")
        
        if new_collection_name:
            active_collection_name = new_collection_name

    if collection_list:
        st.sidebar.markdown("---")
        collection_to_delete = st.sidebar.selectbox("Select to delete:", options=[""] + collection_list)
        if collection_to_delete and st.sidebar.button(f"Delete '{collection_to_delete}'"):
            try:
                cf.delete_collection(client, collection_to_delete)
                sq.delete_collection_sqlite(collection_to_delete)
                st.sidebar.success(f"Collection '{collection_to_delete}' deleted.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Deletion failed: {e}")

    # --- Main RAG Content ---
    if not active_collection_name:
        st.info("Please select or create a collection in the sidebar.")
    else:
        st.success(f"Working on collection: **{active_collection_name}**")
        
        try:
            collection_details = sq.get_collection_params_sqlite(active_collection_name)
            if not collection_details:
                st.error(f"Details for '{active_collection_name}' not found in SQLite.")
                st.stop()

            saved_method = collection_details.get("index_method")
            saved_params = collection_details.get("index_params", {}) 
            pdf_names = collection_details.get("pdfs", [])

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
                st.markdown(f"**Documents ({len(pdf_names)}):**")
                if pdf_names:
                    st.dataframe(pdf_names, use_container_width=True, hide_index=True, column_config={"value":"File Name"})
                else:
                    st.text("No documents added yet.")
        except Exception as e:
            st.error(f"Could not load collection info. Error: {e}")
            st.stop()

        st.divider()

        rag_subtab1, rag_subtab2 = st.tabs(["🗂️ Manage Documents", "🔍 Query Collection"])

        # --- Subtab 1: Add Docs ---
        with rag_subtab1:
            st.header(f"➕ Add New Documents")
            uploaded_files = st.file_uploader("Drag and drop PDF files here", type="pdf", accept_multiple_files=True)

            if st.button("Process and Add to Collection", type="primary"):
                if uploaded_files:
                    try:
                        collection = cf.get_collection(client, active_collection_name)
                        progress_bar = st.progress(0, text="Starting process...")
                        
                        for i, file in enumerate(uploaded_files):
                            progress = (i + 1) / len(uploaded_files)
                            progress_bar.progress(progress, text=f"Processing: {file.name}")
                            
                            if file.name in pdf_names:
                                st.warning(f"File '{file.name}' already exists. Skipped.")
                                continue

                            temp_path = os.path.join("/tmp", file.name)
                            with open(temp_path, "wb") as f:
                                f.write(file.getbuffer())

                            processing_function = getattr(splitter, saved_method)
                            all_params = saved_params.copy()
                            all_params['file_path'] = temp_path
                            documents = processing_function(**all_params)

                            if documents:
                                cf.add_documents(collection, documents, file.name)
                                sq.add_pdf_to_collection_sqlite(active_collection_name, file.name)
                                st.write(f"File '{file.name}' processed and added successfully.")
                            else:
                                st.warning(f"No text extracted from '{file.name}'.")
                            
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                        progress_bar.progress(1.0, text="Process complete!")
                        st.success("All files were processed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.warning("Please upload at least one file.")

        # --- Subtab 2: Query ---
        with rag_subtab2:
            st.header(f"❓ Query '{active_collection_name}'")
            
            friendly_retrievers = [RETRIEVAL_METHOD_NAMES.get(m, m) for m in available_retrievers]
            selected_retrieval_friendly = st.selectbox("Choose a retrieval method:", options=friendly_retrievers)
            technical_retrieval_method = next(k for k, v in RETRIEVAL_METHOD_NAMES.items() if v == selected_retrieval_friendly)

            retrieval_params = {}
            st.caption("Retrieval Parameters")

            if "top_k" in technical_retrieval_method:
                if "re_ranker" in technical_retrieval_method:
                     retrieval_params['high_k'] = st.number_input("High K", 5, 100, 20)
                else:
                     retrieval_params['k'] = st.number_input("K", 1, 50, 5)
            
            if "multi_query" in technical_retrieval_method:
                retrieval_params['n_queries'] = st.number_input("Number of Queries", 2, 10, 3)
                retrieval_params['n_results'] = st.number_input("Results per Query", 2, 20, 5)
            
            if "sentence_window" in technical_retrieval_method:
                default_main = 3 if "re_ranker" in technical_retrieval_method else 1
                default_around = 4 if "re_ranker" in technical_retrieval_method else 3
                retrieval_params['n_main'] = st.number_input("N Main", 1, 10, default_main)
                retrieval_params['n_around'] = st.number_input("N Around", 1, 10, default_around)

            query = st.text_input("Your question:", key="query_input")
            if st.button("Run Query", type="primary"):
                if query and active_collection_name:
                    try:
                        retrieval_function = getattr(retriever, technical_retrieval_method)
                        full_params = retrieval_params.copy()
                        full_params['query'] = query
                        full_params['collection_name'] = active_collection_name
                        
                        with st.spinner(f"Running '{selected_retrieval_friendly}'..."):
                            result_prompt = retrieval_function(**full_params)
                        
                        st.subheader("Retrieval Results:")
                        st.write(result_prompt)
                    except Exception as e:
                        st.error(f"Error during retrieval: {e}")
                else:
                    st.warning("Please enter a question.")

# ==============================================================================
# LOGIC FOR PAGE 2: AGENT CONFIG
# ==============================================================================
elif page_selection == "⚙️ Agent Configuration":
    
    st.header("🛠️ Global Agent Configuration")
    
    # --- Bloco 1: Conexão ---
    st.subheader("1️⃣ Connection Settings")
    with st.container(border=True):
        with st.form("connection_form"):
            try: curr_name = sq.get_config_sqlite("agent_name")
            except: curr_name = ""
            
            try: curr_url = sq.get_config_sqlite("openai_baseurl")
            except: curr_url = ""
            
            try: curr_key = sq.get_config_sqlite("openai_api_key")
            except: curr_key = ""

            new_name = st.text_input("Agent Name:", value=curr_name)
            new_url = st.text_input("Base URL:", value=curr_url)
            new_key = st.text_input("API Key:", value=curr_key, type="password")
            
            st.markdown("---")
            if st.form_submit_button("💾 Save Connection", type="primary"):
                sq.update_config_sqlite("agent_name", new_name)
                sq.update_config_sqlite("openai_baseurl", new_url)
                sq.update_config_sqlite("openai_api_key", new_key)
                st.success("Connection saved!")
                st.rerun()

    # --- Bloco 2: Cérebro ---
    st.subheader("2️⃣ Brain & Behavior")
    
    db_url = ""
    try: db_url = sq.get_config_sqlite("openai_baseurl")
    except: pass

    if not db_url:
        st.warning("Salve a URL acima primeiro para carregar os modelos.")
    else:
        with st.container(border=True):
            api_models = get_models(db_url)
            
            if not api_models:
                st.error(f"Não foi possível listar modelos em: `{db_url}`. Verifique se o serviço está rodando.")
            else:
                with st.form("behavior_form"):
                    # Model Select
                    try: curr_model = sq.get_config_sqlite("model")
                    except: curr_model = api_models[0]

                    if curr_model not in api_models:
                        idx_model = 0
                    else:
                        idx_model = api_models.index(curr_model)
                    
                    selected_model = st.selectbox("Select Model:", options=api_models, index=idx_model)

                    # Retrieval Select
                    try: curr_ret_tech = sq.get_config_sqlite("retrieval_function")
                    except: curr_ret_tech = list(RETRIEVAL_METHOD_NAMES.keys())[0]

                    friendly_opts = list(RETRIEVAL_METHOD_NAMES.values())
                    curr_friendly = RETRIEVAL_METHOD_NAMES.get(curr_ret_tech, friendly_opts[0])
                    
                    try: idx_ret = friendly_opts.index(curr_friendly)
                    except: idx_ret = 0
                    
                    selected_ret_friendly = st.selectbox("Retrieval Strategy:", options=friendly_opts, index=idx_ret)

                    st.markdown("---")
                    if st.form_submit_button("💾 Save Behavior", type="primary"):
                        sq.update_config_sqlite("model", selected_model)
                        
                        tech_name = {v: k for k, v in RETRIEVAL_METHOD_NAMES.items()}[selected_ret_friendly]
                        sq.update_config_sqlite("retrieval_function", tech_name)
                        
                        st.success("Behavior saved!")