import streamlit as st
import pypdf
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Page Configuration
st.set_page_config(page_title="Free RAG Chatbot", page_icon="🤖", layout="wide")

# App Header
st.title("🤖 Free RAG Chatbot")
st.markdown("Upload a PDF and chat with its content for *100% free* using local CPU embeddings and Google's Gemini Free Tier API.")

# Try to retrieve Gemini API Key from Streamlit Secrets automatically
gemini_key = st.secrets.get("GEMINI_API_KEY", "")

# Sidebar for Setup
with st.sidebar:
    if not gemini_key:
        st.header("1. Setup Credentials")
        # Manual input fallback if no secret is configured
        gemini_key = st.text_input("Enter Gemini API Key", type="password", help="Get a free key from Google AI Studio: https://aistudio.google.com/")
    else:
        st.success("🔑 API Key loaded securely from Streamlit Cloud Secrets!")
    
    st.header("2. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Utility Functions
@st.cache_resource
def load_embedding_model():
    # Load lightweight local embedding model (runs completely free on your CPU)
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# Main Application Logic
if not gemini_key:
    st.warning("Please enter your Google Gemini API Key in the sidebar to get started.")
else:
    # Initialize Gemini API
    genai.configure(api_key=gemini_key)
    
    if uploaded_file is not None:
        # Step 1: Load local embedding model
        with st.spinner("Loading local AI embedding model... (This takes a few seconds on first run)"):
            model = load_embedding_model()
        
        # Step 2: Process the uploaded file
        if "processed_data" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner("Extracting text and chunking PDF..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                chunks = chunk_text(raw_text)
                st.session_state["chunks"] = chunks
                st.session_state["file_name"] = uploaded_file.name
                
            with st.spinner("Generating local text embeddings..."):
                # Encode text chunks locally
                embeddings = model.encode(chunks, show_progress_bar=False)
                st.session_state["embeddings"] = embeddings
                st.session_state["processed_data"] = True
                st.success(f"Successfully processed {len(chunks)} text chunks!")
        
        # Step 3: Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I have analyzed your document. Ask me anything about it!"}]
            
        # Display chat messages
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # Chat Input
        if user_query := st.chat_input("Ask a question about the document..."):
            # Add user message
            st.session_state["messages"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
                
            # Perform RAG retrieval
            with st.spinner("Searching document for relevant sections..."):
                query_embedding = model.encode(user_query)
                
                # Calculate similarities
                similarities = [cosine_similarity(query_embedding, chunk_emb) for chunk_emb in st.session_state["embeddings"]]
                
                # Get top 3 chunks
                top_indices = np.argsort(similarities)[-3:][::-1]
                relevant_context = "\n\n---\n\n".join([st.session_state["chunks"][idx] for idx in top_indices])
                
            # Call Gemini
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        # Construct a clean prompt combining context and query
                        prompt = f"""You are a helpful AI assistant. Answer the user's question based strictly on the provided document context.
If the answer is not in the context, politely state that you cannot find the answer in the document.

CONTEXT:
{relevant_context}

QUESTION:
{user_query}

ANSWER:"""
                        gemini_model = genai.GenerativeModel("gemini-3.5-flash")
                        response = gemini_model.generate_content(prompt)
                        assistant_response = response.text
                        
                        message_placeholder.write(assistant_response)
                        # Add assistant response to history
                        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})
                    except Exception as e:
                        st.error(f"Error calling Gemini API: {e}")
    else:
        st.info("Please upload a PDF document in the sidebar to begin chatting.")