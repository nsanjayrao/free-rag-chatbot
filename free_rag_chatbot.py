import streamlit as st
import pypdf
import docx
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Page Configuration
st.set_page_config(page_title="Free RAG Chatbot", page_icon="🤖", layout="wide")

# App Header
st.title("🤖 Free Multi-File RAG Chatbot")
st.markdown("Upload multiple files of different formats (PDF, DOCX, TXT, CSV, XLSX) and chat with all of them for **100% free** using local CPU embeddings and Gemini.")

# Try to retrieve Gemini API Key from Streamlit Secrets automatically
try:
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    gemini_key = ""

# Sidebar for Setup
with st.sidebar:
    if not gemini_key:
        st.header("1. Setup Credentials")
        # Manual input fallback if no secret is configured
        gemini_key = st.text_input("Enter Gemini API Key", type="password", help="Get a free key from Google AI Studio: https://aistudio.google.com/")
    else:
        st.success("🔑 API Key loaded securely from Streamlit Cloud Secrets!")
    
    st.header("2. Upload Documents")
    # Enabled multiple files upload and added support for PDF, DOCX, TXT, CSV, XLSX
    uploaded_files = st.file_uploader(
        "Upload document files", 
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"], 
        accept_multiple_files=True
    )

# Utility Functions
@st.cache_resource
def load_embedding_model():
    # Load lightweight local embedding model (runs completely free on your CPU)
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_file(file):
    name = file.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        elif name.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif name.endswith(".txt"):
            try:
                return file.read().decode("utf-8")
            except Exception:
                return file.read().decode("latin-1")
        elif name.endswith(".csv"):
            df = pd.read_csv(file)
            return df.to_string(index=False)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
            return df.to_string(index=False)
    except Exception as e:
        st.error(f"Error reading file {file.name}: {e}")
        return ""
    return ""

def chunk_text_with_source(text, filename, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "text": chunk_text,
            "source": filename
        })
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
    
    if uploaded_files:
        # Step 1: Load local embedding model
        with st.spinner("Loading local AI embedding model... (This takes a few seconds on first run)"):
            model = load_embedding_model()
        
        # Create a unique key representing the current set of files to check for changes
        current_file_signature = "|".join([f"{f.name}_{f.size}" for f in uploaded_files])
        
        # Step 2: Process the uploaded files if changed
        if "file_signature" not in st.session_state or st.session_state["file_signature"] != current_file_signature:
            all_chunks = []
            
            with st.spinner("Extracting text and chunking files..."):
                for file in uploaded_files:
                    raw_text = extract_text_from_file(file)
                    if raw_text.strip():
                        chunks = chunk_text_with_source(raw_text, file.name)
                        all_chunks.extend(chunks)
                
                if not all_chunks:
                    st.error("No extractable text was found in any of the uploaded files.")
                    st.stop()
                
                st.session_state["all_chunks"] = all_chunks
                st.session_state["file_signature"] = current_file_signature
                
            with st.spinner(f"Generating local text embeddings for {len(all_chunks)} chunks..."):
                # Encode text chunks locally
                texts = [c["text"] for c in all_chunks]
                embeddings = model.encode(texts, show_progress_bar=False)
                st.session_state["embeddings"] = embeddings
                st.session_state["processed_data"] = True
                st.success(f"Successfully indexed {len(uploaded_files)} files into {len(all_chunks)} text chunks!")
        
        # Step 3: Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I have analyzed your documents. Ask me anything about them!"}]
            
        # Display chat messages
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # Chat Input
        if user_query := st.chat_input("Ask a question across all documents..."):
            # Add user message
            st.session_state["messages"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
                
            # Perform RAG retrieval
            with st.spinner("Searching documents for relevant sections..."):
                query_embedding = model.encode(user_query)
                
                # Calculate similarities
                similarities = [cosine_similarity(query_embedding, chunk_emb) for chunk_emb in st.session_state["embeddings"]]
                
                # Get top 3 chunks
                top_indices = np.argsort(similarities)[-3:][::-1]
                
                # Format context with source metadata
                context_parts = []
                sources_used = set()
                for idx in top_indices:
                    chunk = st.session_state["all_chunks"][idx]
                    context_parts.append(f"[Source File: {chunk['source']}]\n{chunk['text']}")
                    sources_used.add(chunk['source'])
                
                relevant_context = "\n\n---\n\n".join(context_parts)
                sources_string = ", ".join(sources_used)
                
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
                        
                        # Display assistant response and append sources used
                        full_response_with_sources = f"{assistant_response}\n\n---\n*Sources used: {sources_string}*"
                        message_placeholder.write(full_response_with_sources)
                        
                        # Add assistant response to history
                        st.session_state["messages"].append({"role": "assistant", "content": full_response_with_sources})
                    except Exception as e:
                        st.error(f"Error calling Gemini API: {e}")
    else:
        st.info("Please upload one or more document files in the sidebar to begin chatting.")