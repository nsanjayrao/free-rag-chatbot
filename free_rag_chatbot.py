import streamlit as st
import pypdf
import docx
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Page Configuration
st.set_page_config(page_title="Free Multi-File RAG Chatbot", page_icon="🤖", layout="wide")

# Inject Custom CSS for Premium SaaS styling
st.markdown("""
    <style>
    /* Metric Cards Custom Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    /* Document Preview Cards */
    .doc-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 15px;
        margin-bottom: 10px;
    }
    
    /* Interactive Button Animations */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🤖 Free Multi-File RAG Analyzer")
st.markdown("Upload multiple enterprise documents (PDF, DOCX, TXT, CSV, XLSX) and run semantic intelligence queries over all of them concurrently.")

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
    uploaded_files = st.file_uploader(
        "Upload document files", 
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"], 
        accept_multiple_files=True,
        help="Upload up to 200MB of PDFs, Word docs, text files, or spreadsheets."
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

def recursive_split_text(text, filename, chunk_size=1000, overlap=200):
    separators = ["\n\n", "\n", " ", ""]
    
    def split_recursive(text_to_split, separators_list):
        if len(text_to_split) <= chunk_size:
            return [text_to_split]
        
        separator = separators_list[0]
        next_separators = separators_list[1:]
        
        if not next_separators:
            # Hard boundary fall-back
            return [text_to_split[i:i+chunk_size] for i in range(0, len(text_to_split), chunk_size - overlap)]
        
        splits = text_to_split.split(separator)
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If adding the next split exceeds chunk_size
            if len(current_chunk) + (len(separator) if current_chunk else 0) + len(split) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(split) > chunk_size:
                    # Recurse with finer separators for oversized split
                    chunks.extend(split_recursive(split, next_separators))
                    current_chunk = ""
                else:
                    # Construct smart overlap from last appended chunk if possible
                    if chunks:
                        last_chunk = chunks[-1]
                        overlap_start = max(0, len(last_chunk) - overlap)
                        potential_chunk = last_chunk[overlap_start:] + separator + split
                        if len(potential_chunk) <= chunk_size:
                            current_chunk = potential_chunk
                        else:
                            current_chunk = split
                    else:
                        current_chunk = split
            else:
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
        
    raw_chunks = split_recursive(text, separators)
    
    # Structure chunks with source filename metadata
    return [{"text": c, "source": filename} for c in raw_chunks]

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
            file_stats = []
            
            with st.spinner("Extracting text and chunking files recursively..."):
                for file in uploaded_files:
                    raw_text = extract_text_from_file(file)
                    if raw_text.strip():
                        # Using new robust Recursive Character Text Splitter logic
                        chunks = recursive_split_text(raw_text, file.name)
                        all_chunks.extend(chunks)
                        file_stats.append({
                            "name": file.name,
                            "size": f"{file.size / 1024:.1f} KB",
                            "chunks": len(chunks)
                        })
                
                if not all_chunks:
                    st.error("No extractable text was found in any of the uploaded files.")
                    st.stop()
                
                st.session_state["all_chunks"] = all_chunks
                st.session_state["file_stats"] = file_stats
                st.session_state["file_signature"] = current_file_signature
                
            with st.spinner(f"Generating local text embeddings for {len(all_chunks)} chunks..."):
                # Encode text chunks locally
                texts = [c["text"] for c in all_chunks]
                embeddings = model.encode(texts, show_progress_bar=False)
                st.session_state["embeddings"] = embeddings
                st.session_state["processed_data"] = True
                
        # Premium UI: Display system statistics metrics cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Processed", len(uploaded_files))
        with col2:
            st.metric("Vector Segments (Chunks)", len(st.session_state["all_chunks"]))
        with col3:
            st.metric("Vector Extraction Engine", "HuggingFace CPU")
            
        # Display document details in an Expandable Section
        with st.expander("📁 View Processed Files and Statistics", expanded=False):
            for stat in st.session_state.get("file_stats", []):
                st.markdown(f"""
                <div class="doc-card">
                    📄 <strong>{stat['name']}</strong> &nbsp;|&nbsp; ⚖️ Size: <code>{stat['size']}</code> &nbsp;|&nbsp; 🧩 Segments: <code>{stat['chunks']}</code>
                </div>
                """, unsafe_allow_html=True)
        
        # Step 3: Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I have analyzed your documents using recursive semantic indexing. Ask me anything about them!"}]
            
        # Display chat messages with integrated styled references
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("sources"):
                    with st.expander("📚 Sources & References used for this response", expanded=False):
                        st.markdown(message["sources"])
                
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
                source_details = []
                for idx in top_indices:
                    chunk = st.session_state["all_chunks"][idx]
                    context_parts.append(f"[Source File: {chunk['source']}]\n{chunk['text']}")
                    sources_used.add(chunk['source'])
                    # Save exact snippet details for UI references
                    source_details.append(f"**From `{chunk['source']}`:**\n> {chunk['text'].strip()}...")
                
                relevant_context = "\n\n---\n\n".join(context_parts)
                sources_string = "\n\n---\n\n".join(source_details)
                
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
                        
                        # Render clean assistant response
                        message_placeholder.write(assistant_response)
                        
                        # Display sources in expandable card underneath response
                        with st.expander("📚 Sources & References used for this response", expanded=False):
                            st.markdown(sources_string)
                        
                        # Add assistant response to history
                        st.session_state["messages"].append({
                            "role": "assistant", 
                            "content": assistant_response,
                            "sources": sources_string
                        })
                    except Exception as e:
                        st.error(f"Error calling Gemini API: {e}")
    else:
        st.info("Please upload one or more document files in the sidebar to begin chatting.")