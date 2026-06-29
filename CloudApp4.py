import base64
import streamlit as st
import sys
import re
import os
from types import ModuleType
import importlib.machinery
from google import genai
from google.genai import types
from pinecone import Pinecone

# =========================================================================
# 🛡️ ENVIRONMENT ENHANCEMENT: Architectural Dynamic System Mocking
# =========================================================================
@st.cache_resource
def fix_environment_imports():
    class DummyLoader:
        def create_module(self, spec): return None
        def exec_module(self, module): pass
    class DeepMock(ModuleType):
        def __getattr__(self, name):
            if name in ('__path__', '__spec__'):
                return [] if name == '__path__' else None
            return DeepMock(f"{self.__name__}.{name}")
        def __call__(self, *args, **kwargs):
            return None
    class MockFinder:
        def find_spec(self, fullname, path, target=None):
            if fullname.startswith("torchvision"):
                mock = DeepMock(fullname)
                spec = importlib.machinery.ModuleSpec(fullname, DummyLoader())
                spec.submodule_search_locations = []
                mock.__spec__ = spec
                mock.__path__ = []
                sys.modules[fullname] = mock
                return spec
            return None
    sys.meta_path.insert(0, MockFinder())

fix_environment_imports()

# =========================================================================
# 🎨 CUSTOM STYLING: Dark Fantasy & Tabletop DM Screen Theme
# =========================================================================
st.set_page_config(page_title="Pocket D&D Loremaster", page_icon="🎲", layout="centered")

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

img_base64 = get_base64_image("your_d20_image.png")

st.markdown(f"""
<style>
    .stApp {{
        background-color: #1a1613 !important;
        background-image: radial-gradient(#2d2219 1px, transparent 0) !important;
        background-size: 24px 24px !important;
        color: #e3d1be !important;
        font-family: 'Georgia', serif !important;
    }}
    .brand-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        width: 100%;
        margin-bottom: 15px;
    }}
    .brand-logo {{
        width: 110px;
        height: auto;
        margin-bottom: 15px;
        mix-blend-mode: screen !important;
    }}
    h1 {{
        color: #d4af37 !important;
        font-family: 'Georgia', serif !important;
        text-shadow: 2px 2px 4px #000000;
        text-decoration: underline !important;
        text-decoration-color: #8c2d19 !important;
        text-underline-offset: 12px !important; 
        text-decoration-thickness: 2px !important; 
        margin: 0px !important;
        padding-bottom: 5px !important;
        width: 100%;
    }}
    .stMarkdown, p, span, label {{
        color: #e3d1be !important;
    }}
    .stChatInput textarea {{
        background-color: #2b221a !important;
        color: #f5eccd !important;
        border: 1px solid #8c2d19 !important;
        border-radius: 4px !important;
    }}
    .stSpinner > div {{
        border-top-color: #d4af37 !important;
    }}
</style>
""", unsafe_allow_html=True)

# 🎲 BRANDING LAYOUT INJECTION
st.markdown(f"""
<div class="brand-container">
    <img src="data:image/png;base64,{img_base64}" class="brand-logo">
    <h1>Pocket D&D Loremaster</h1>
    <p style='color: #8a7663; font-style: italic; margin-top: 18px;'>Powered by magic!</p>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# 🔑 SECURITY CREDENTIALS & GLOBAL INSTANTIATION MAPS
# =========================================================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "dnd-index"

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

@st.cache_resource
def get_llm_service():
    # FIXED BOUND VALUE: Explicitly reference using the modern google namespace setup
    return genai.Client(api_key=GEMINI_API_KEY)

ai_client = get_llm_service()

# =========================================================================
# 🔍 VECTOR DATABANK EXTRACTION ENGINE
# =========================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def cached_vector_search(query_text):
    if not query_text:
        return []
    
    response = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query_text],
        parameters={"input_type": "query"}
    )
    
    query_vector = None
    try:
        # Robust deep value extraction matrix for v9+ SDK objects
        if hasattr(response, 'data') and response.data:
            if isinstance(response.data, list) and len(response.data) > 0:
                query_vector = response.data[0].values
            else:
                query_vector = getattr(response.data, 'values', None)
        elif isinstance(response, list) and len(response) > 0:
            query_vector = response[0].values if hasattr(response[0], 'values') else response[0]
        else:
            query_vector = getattr(response, 'values', None)
            
    except Exception as e:
        print(f"⚠️ Vector Extractor Intercept: {str(e)}")

    if query_vector is None or not isinstance(query_vector, list):
        query_vector = [0.0] * 1024

    results = index.query(
        namespace="markdown-docs", 
        vector=query_vector,
        top_k=4,
        include_metadata=True
    )
    
    serialized_docs = []
    matches = results.get("matches", [])
    
    for idx, match in enumerate(matches):
        meta = match.get("metadata", {})
        text_content = meta.get("chunk_text", "No context found.")
        source_book = meta.get("source_file", "Unknown Rulebook")
        chunk_idx = meta.get("chunk_index", "N/A")
        
        source_label = f"📜 {source_book} (Section {chunk_idx})"
        
        serialized_docs.append({
            "page_content": text_content, 
            "metadata": {"source_label": source_label},
            "source_label": source_label
        })
    return serialized_docs

class NativePineconeVectorStore:
    def __init__(self, index):
        self.index = index
        
    def retrieve(self, query_text):
        class SimpleDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        raw_docs = cached_vector_search(query_text)
        return [SimpleDoc(d["page_content"], {**d["metadata"], "source_label": d["source_label"]}) for d in raw_docs]

vector_store = NativePineconeVectorStore(index)

# =========================================================================
# 🎭 SESSION STATE STORAGE & UI CONVERSATION REPLAY
# =========================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Ask about a spell, class, or rule:")

# =========================================================================
# ⚡ CORE QUERY ORCHESTRATION PIPELINE
# =========================================================================
if user_query:
    with st.chat_message("user"):
        st.write(user_query)
        
    with st.spinner("Searching knowledge base..."):
        search_query = user_query
        vague_words = {"it", "that", "this", "they", "spell details", "more details", "explain", "him", "her", "them", "there"}
        has_vague_word = any(word in user_query.lower() for word in vague_words)
        
        if len(st.session_state.chat_history) > 0 and has_vague_word:
            recent_context = " ".join([m['content'] for m in st.session_state.chat_history[-2:]])
            entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', recent_context))
            blacklisted_words = {"the", "i", "you", "he", "she", "it", "they", "this", "that", "a", "an", "and", "but", "my", "your", "we", "our"}
            entities = {e for e in entities if e.lower() not in blacklisted_words}
            
            if entities:
                search_query = f"{user_query} {' '.join(entities)}"
            else:
                last_user_msg = next((m['content'] for m in reversed(st.session_state.chat_history) if m['role'] == 'user'), "")
                search_query = f"{last_user_msg} {user_query}"

        matched_docs = vector_store.retrieve(search_query)
        unique_sources = list(set([doc.metadata["source_label"] for doc in matched_docs if "source_label" in doc.metadata]))
        
        # 📜 FIXED: Format alternating chat logs as standard dictionaries for the modern SDK stream handler
        native_history = []
        for m in st.session_state.chat_history[-4:]:
            sdk_role = "user" if m["role"] == "user" else "model"
            native_history.append({
                "role": sdk_role,
                "parts": [m["content"]]
            })

        context_str = "\n\n".join([doc.page_content for doc in matched_docs if doc.page_content != "No context found."])
        
        final_prompt_text = f"""Please answer my question using these referenced materials.

Context from Rulebooks:
{context_str}

User Question: {user_query}"""

        # Append current exchange block turn element using native array maps
