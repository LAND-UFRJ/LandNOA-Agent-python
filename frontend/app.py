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
from streamlit_option_menu import option_menu
import json

nltk.download("punkt")

# ==============================================================================
# 1. Page configuration
# ==============================================================================
st.set_page_config(page_title="LandNOA Manager", layout="wide")


# ==============================================================================
# 2. Needed functions
# ==============================================================================

def get_models(base_url: str):
    """Busca modelos disponíveis na API do Ollama/OpenAI"""
    try:
        r = requests.get(f"{base_url}/models", timeout=2)
        data = r.json()
        if "data" in data:
            return [m["id"] for m in data["data"]]
        return []
    except:
        return []

@st.cache_resource
def start_services():
    try:
        client = cf.connect_chroma()
        splitter = Splitter()
        retriever = Retriever()
        return client, splitter, retriever
    except:
        return None, None, None

client, splitter, retriever = start_services() 
if client is None: st.stop()


METHOD_NAMES = {
    "equal_chunks": "Fixed Size (Langchain)",
    "unstructured_chunks": "Structured (Unstructured)",
    "simple_decision": "Semantic (Linear)",
    "changing_decision": "Semantic (Exponential)"
}
available_methods = [m for m, f in inspect.getmembers(splitter, inspect.ismethod) if m in METHOD_NAMES]

RETRIEVAL_METHOD_NAMES = {
    "top_k": "Top-K",
    "top_k_reranker": "Top-K (with Re-Ranker)",
    "multi_query": "Multi-Query",
    "multi_query_reranker": "Multi-Query (with Re-Ranker)",
    "sentence_window_retrieval": "Sentence Window",
    "sentence_window_retriever_reranker": "Sentence Window (with Re-Ranker)"
}
available_retrievers = [m for m, f in inspect.getmembers(retriever, inspect.ismethod) if m in RETRIEVAL_METHOD_NAMES]

# ==============================================================================
# 3. NAVBAR
# ==============================================================================

col_logo, col_menu = st.columns([1, 5])

with col_logo:
    st.title("🤖 LandNOA")

with col_menu:
    selected_page = option_menu(
        menu_title=None,
        options=["RAG System", "Agent Config", "System Prompts", "Tools Config"],
        icons=["database", "cpu", "card-text", "tools"],
        default_index=0,
        orientation="horizontal"
    )

st.divider()

# Tab 1: RAG SYS
if selected_page == "RAG System":
    
    # Sidebar
    st.sidebar.header("📂 Collection Management")
    try: collection_list = sq.list_collections_sqlite() 
    except: collection_list = []
    try:
        mode = st.sidebar.segmented_control("Mode", ["Select", "Create"], selection_mode="single", default="Select")
    except AttributeError:
        mode = st.sidebar.radio("Mode", ["Select", "Create"])

    active_collection_name = None
    
    if mode == "Select":
        if collection_list:
            active_collection_name = st.sidebar.selectbox("Choose Collection", collection_list)
        else:
            st.sidebar.info("No collections found.")
    else:
        with st.sidebar.container(border=True):
            new_name = st.text_input("Name", placeholder="e.g., Finance_Docs")
            friendly_names = [METHOD_NAMES.get(m, m) for m in available_methods]
            sel_method = st.selectbox("Strategy", friendly_names)
            tech_method = next(k for k, v in METHOD_NAMES.items() if v == sel_method)

            params = {}
            if tech_method == "equal_chunks":
                c1, c2 = st.columns(2)
                params['chunck_size'] = c1.number_input("Size", 100, 2000, 750)
                params['chunk_overlap'] = c2.number_input("Overlap", 0, 500, 50)
            elif "decision" in tech_method:
                params['start_limit'] = st.slider("Similarity", 0.0, 1.0, 0.7)
                params['y'] = st.slider("Growth", 0.0, 1.0, 0.5)

            if st.button("Create Collection", type="primary", use_container_width=True):
                if new_name and new_name not in collection_list:
                    sq.create_collection_sqlite(new_name, tech_method, params)
                    cf.create_collection(client, new_name)
                    st.rerun()
        
        if new_name: active_collection_name = new_name

    if collection_list and mode == "Select":
        st.sidebar.divider()
        with st.sidebar.expander("⚠️ Delete Collection"):
            del_col = st.selectbox("Select to Delete", [""] + collection_list)
            if del_col and st.button("Confirm Delete", type="primary"):
                cf.delete_collection(client, del_col)
                sq.delete_collection_sqlite(del_col)
                st.rerun()

    # Main Area
    if not active_collection_name:
        st.info("👈 Select a collection from the sidebar to manage documents.")
    else:
        c_head, c_meta = st.columns([3, 1])
        c_head.subheader(f"{active_collection_name}")
        
        try:
            details = sq.get_collection_params_sqlite(active_collection_name)
            pdf_names = details.get("pdfs", [])
            c_meta.caption(f"📚 {len(pdf_names)} Documents Indexed")
            
            with st.expander("View Configuration Details"):
                st.json(details)
        except: pass

        t1, t2 = st.tabs(["📥 Upload Documents", "🔎 Search & Test"])
        
        with t1:
            with st.container(border=True):
                files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
                if files:
                    if st.button(f"Process {len(files)} Files", type="primary"):
                        coll = cf.get_collection(client, active_collection_name)
                        bar = st.progress(0, "Starting...")
                        for i, f in enumerate(files):
                            if f.name in pdf_names: continue
                            tmp = os.path.join("/tmp", f.name)
                            with open(tmp, "wb") as file: file.write(f.getbuffer())
                            
                            func = getattr(splitter, details["index_method"])
                            p = details["index_params"].copy()
                            p['file_path'] = tmp
                            docs = func(**p)
                            
                            if docs:
                                cf.add_documents(coll, docs, f.name)
                                sq.add_pdf_to_collection_sqlite(active_collection_name, f.name)
                            os.remove(tmp)
                            bar.progress((i+1)/len(files))
                        st.success("Done!")
                        st.rerun()

        with t2:
            query = st.text_input("Search", placeholder="Ask a question about your documents...")
            c_opt, c_btn = st.columns([4, 1])
            with c_opt:
                friendly_rets = [RETRIEVAL_METHOD_NAMES.get(m, m) for m in available_retrievers]
                sel_ret = st.selectbox("Method", friendly_rets, label_visibility="collapsed")
                tech_ret = next(k for k, v in RETRIEVAL_METHOD_NAMES.items() if v == sel_ret)
            
            with c_btn:
                search_btn = st.button("Search", type="primary", use_container_width=True)
            
            if search_btn and query:
                func = getattr(retriever, tech_ret)
                p = {'query': query, 'collection_name': active_collection_name, 'k': 5}
                with st.spinner("Searching..."):
                    res = func(**p)
                st.markdown("### Results")
                with st.container(border=True):
                     st.write(res)

#Tab 2 Config
elif selected_page == "Agent Config":
    
    st.header("⚙️ Agent Configuration")
    c_main, c_side = st.columns([2, 1])
    
    with c_main:
        st.subheader("🔌 Connection")
        with st.container(border=True):
            try: c_name = sq.get_config_sqlite("agent_name")
            except: c_name = ""
            try: c_url = sq.get_config_sqlite("openai_baseurl")
            except: c_url = ""
            try: c_key = sq.get_config_sqlite("openai_api_key")
            except: c_key = ""

            n_name = st.text_input("Agent Name", c_name)
            n_url = st.text_input("Base URL", c_url, placeholder="http://localhost:11434/v1")
            n_key = st.text_input("API Key", c_key, type="password")
            
            if st.button("Save Connection", type="primary"):
                sq.update_config_sqlite("agent_name", n_name)
                sq.update_config_sqlite("openai_baseurl", n_url)
                sq.update_config_sqlite("openai_api_key", n_key)
                st.success("Connection Saved!")
                st.rerun()

    with c_side:
        st.subheader("🧠 Brain")
        with st.container(border=True):
            if not c_url:
                st.info("Set URL first.")
            else:
                models = get_models(c_url)
                if not models: 
                    st.warning("No models found.")
                    models = [""] 
                
                try: c_mod = sq.get_config_sqlite("model")
                except: c_mod = models[0]
                
                idx = models.index(c_mod) if c_mod in models else 0
                sel_mod = st.selectbox("Model", models, index=idx)

                try: c_ret = sq.get_config_sqlite("retrieval_function")
                except: c_ret = list(RETRIEVAL_METHOD_NAMES.keys())[0]
                
                f_opts = list(RETRIEVAL_METHOD_NAMES.values())
                cur_f = RETRIEVAL_METHOD_NAMES.get(c_ret, f_opts[0])
                idx_r = f_opts.index(cur_f)
                sel_ret_f = st.selectbox("Retrieval", f_opts, index=idx_r)

                if st.button("Update Brain", use_container_width=True):
                    sq.update_config_sqlite("model", sel_mod)
                    tech = {v: k for k, v in RETRIEVAL_METHOD_NAMES.items()}[sel_ret_f]
                    sq.update_config_sqlite("retrieval_function", tech)
                    st.success("Brain Updated!")

# Tab 3: Prompts
elif selected_page == "System Prompts":
    
    st.header("📝 System Prompts")
    
    col_list, col_edit = st.columns([1, 2])
    
    with col_list:
        st.subheader("Select Prompt")
        try: prompts = sq.list_prompts_sqlite()
        except: prompts = []
        
        
        mode = st.radio("Mode:", ["Edit Existing", "Create New"], horizontal=True)
        
        selected_prompt = None
        if mode == "Edit Existing":
            if not prompts:
                st.info("No prompts created yet.")
            else:
                prompt_options = {p['id']: f" {p.get('obs', 'Untitled')}" for p in prompts}
                sel_id = st.selectbox("Choose Prompt:", list(prompt_options.keys()), format_func=lambda x: prompt_options[x])
                selected_prompt = next((p for p in prompts if p['id'] == sel_id), None)

    with col_edit:
        st.subheader("Editor")
        with st.container(border=True):
            if mode == "Create New":
                with st.form("new_prompt"):
                    new_text = st.text_area("System Prompt Text", height=300, placeholder="You are a helpful assistant...")
                    new_obs = st.text_input("Observation")
                    
                    if st.form_submit_button("Create Prompt", type="primary"):
                        if new_obs and new_text:
                            sq.add_prompt_sqlite(new_obs, new_text)
                            st.success("Prompt Created!")
                            st.rerun()
                        else:
                            st.error("Please fill all fields.")
                            
            elif mode == "Edit Existing" and selected_prompt:
            
                with st.form("edit_prompt"):
                    st.caption(f"Editing ID: {selected_prompt['id']}")
                    

                    ed_text = st.text_area("System Prompt Text", value=selected_prompt['prompt'], height=300)
                    ed_obs = st.text_input("Observation", value=selected_prompt['obs'])
                    
                    
                    if st.form_submit_button("💾 Update Prompt", type="primary", use_container_width=True):
                        sq.update_prompt_sqlite(selected_prompt['id'], ed_obs, ed_text)
                        st.toast("Prompt updated successfully!", icon="✅")
                        st.rerun()

                
                st.markdown("---") 
                with st.expander("🗑️ Delete this prompt", expanded=False):
                    st.warning("This action cannot be undone.")
                    
                    if st.button("Confirm Deletion", type="secondary", use_container_width=True):
                        sq.delete_prompt_sqlite(selected_prompt['id'])
                        st.success("Prompt deleted!")
                        st.rerun()

# Tab 4 Tools
elif selected_page == "Tools Config":
    
    st.header("🔧 Tools Configuration")
    
    try: tools = sq.list_tools_sqlite()
    except: tools = []
    
    if tools:
        with st.expander("📚 View Tool Library", expanded=True):
            st.dataframe(tools, use_container_width=True, hide_index=True)
    else:
        st.info("No tools configured yet.")

    st.divider()
    c_add, c_edit = st.columns(2)
    
    
    with c_add:
        st.subheader("Add Tool")
        with st.container(border=True):
            with st.form("add_tool"):
                t_name = st.text_input("Tool Name (Unique ID)", placeholder="search_api")
                t_url = st.text_input("Endpoint URL", placeholder="http://localhost:8000/search")
                t_desc = st.text_area("Description", placeholder="Searches the web...")
                
                if st.form_submit_button("Add Tool", type="primary"):
                    if t_name and t_url:
                        try:
                            sq.add_tool_sqlite(t_name, t_url, t_desc)
                            st.success(f"Added '{t_name}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("Name and URL required.")

   
    with c_edit:
        st.subheader("Edit / Remove")
        with st.container(border=True):
            if not tools:
                st.info("Nothing to edit.")
            else:
                tool_names = [t['name'] for t in tools]
                sel_tool_name = st.selectbox("Select Tool", tool_names)
                sel_tool = next((t for t in tools if t['name'] == sel_tool_name), None)
                
                if sel_tool:
                    with st.form("edit_tool"):
                        st.caption(f"Editing: {sel_tool['name']}")
                        et_url = st.text_input("URL", value=sel_tool['url'])
                        et_desc = st.text_area("Description", value=sel_tool['description'])
                        
                        c1, c2 = st.columns(2)
                        if c1.form_submit_button("Update", type="primary"):
                            sq.update_tool_sqlite(sel_tool['name'], et_url, et_desc)
                            st.success("Updated!")
                            st.rerun()
                        
                        if c2.form_submit_button("Delete"):
                            sq.remove_tool_sqlite(sel_tool['name'])
                            st.rerun()