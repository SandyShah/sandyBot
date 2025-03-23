import streamlit as st
import os
import numpy as np
from llama_index.core import load_index_from_storage, Settings, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore

# **🔹 Load OpenAI API Key Securely**
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
if not OPENAI_API_KEY:
    st.error("OpenAI API Key is missing. Please set it in Hugging Face Spaces secrets.")
    st.stop()

# **🔹 Configure OpenAI Models (Cached)**
@st.cache_resource
def get_models():
    llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.2)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
    return llm, embed_model

Settings.llm, Settings.embed_model = get_models()

# **🔹 Load Vector Store (Cached)**
@st.cache_resource
def load_vector_store():
    storage_path = "./vector_store"
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=storage_path),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=storage_path),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=storage_path),
    )
    return load_index_from_storage(storage_context)

index = load_vector_store()
retriever = index.as_retriever(similarity_top_k=15)
retriever_small = index.as_retriever(similarity_top_k=5)
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=800)
reranker = LLMRerank(llm=Settings.llm, top_n=5)

# **🔹 Function to Analyze Top-k Relevance Scores**
def analyze_scores(nodes, k=5):
    scores = np.array([node.score for node in nodes[:k]])
    avg_score = np.mean(scores)
    std_dev = np.std(scores)
    return avg_score, std_dev

# **🔹 Small LLM Decides if Answer is Scattered**
def needs_reranking(query, nodes):
    prompt = f"""
    You are deciding whether the answer to a user's query is found in a single location or spread across many blog posts.
    If the answer is mostly in 1-2 passages, reply: "localized".
    If it requires multiple passages, reply: "general".
    
    Examples:
    Query: "Fetch me names of all cars mentioned in blogpost."
    Reply: "general"
    
    Query: "Which college did Sandy study in?"
    Reply: "localized"
    
    Query: "Which all cities Sandy has been to?"
    Reply: "general"
    
    User's Query: "{query}"
    """
    response = Settings.llm.complete(prompt).text.lower()
    return response == "general"

# **🔹 Smart Query Handling**
def smart_query(query):
    initial_results = retriever_small.retrieve(query)
    avg_score, std_dev = analyze_scores(initial_results)

    if needs_reranking(query, initial_results):
        query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[reranker], memory=chat_memory)
    else:
        query_engine = RetrieverQueryEngine.from_args(retriever, memory=chat_memory)
    
    return query_engine.query(query).response

# **🔹 Streamlit UI**
st.title("📚 Let us Scan through Sandy's writings !")
st.write("This chatbot retrieves information from Sandy's blog articles.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages in a WhatsApp-style interface
for role, text in st.session_state.chat_history:
    with st.chat_message("user" if role == "You" else "assistant"):
        st.markdown(f"**{role}:** {text}")

# User input with 'Enter' to send
user_query = st.chat_input("Type your message...")
if user_query:
    with st.spinner("Retrieving answer..."):
        response = smart_query(user_query)
        st.session_state.chat_history.append((user_query))
        st.session_state.chat_history.append((response))
        st.rerun()

# Reset chat button
if st.button("Reset Chat"):
    chat_memory.reset()
    st.session_state.chat_history = []
    st.success("Chat memory cleared!")
    st.rerun()