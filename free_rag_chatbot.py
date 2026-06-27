import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import docx
import faiss
import google.generativeai as genai
import numpy as np
import pandas as pd
import pypdf
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / ".rag_cache"
CHAT_DIR = APP_DIR / ".chat_history"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL_NAME = "gemini-3.5-flash"
TOP_K_RETRIEVAL = 10
TOP_K_CONTEXT = 4


st.set_page_config(page_title="Professional RAG Chatbot", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 18px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05);
    }
    .doc-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 15px;
        margin-bottom: 10px;
    }
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Professional Multi-File RAG Analyzer")
st.markdown(
    "Upload PDF, DOCX, TXT, CSV, or Excel files and ask grounded questions with FAISS retrieval, hybrid search, reranking, memory, streaming, and citations."
)

try:
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    gemini_key = ""

with st.sidebar:
    if not gemini_key:
        st.header("1. Credentials")
        gemini_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            help="Create a free key in Google AI Studio.",
        )
    else:
        st.success("API key loaded from Streamlit secrets.")

    st.header("2. Documents")
    uploaded_files = st.file_uploader(
        "Upload document files",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload PDFs, Word docs, text files, CSVs, or spreadsheets.",
    )

    st.header("3. Retrieval")
    use_hybrid_search = st.toggle("Hybrid BM25 + FAISS", value=True)
    use_reranker = st.toggle("Cross-encoder reranking", value=True)
    use_query_expansion = st.toggle("Query expansion", value=False, help="Uses one extra Gemini request per question. Keep this off on the free tier unless you need broader retrieval.")
    top_k_context = st.slider("Cited chunks", 2, 8, TOP_K_CONTEXT)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


def ensure_storage_dirs():
    CACHE_DIR.mkdir(exist_ok=True)
    CHAT_DIR.mkdir(exist_ok=True)


def safe_filename(value):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def file_signature(files):
    digest = hashlib.sha256()
    for file in files:
        data = file.getvalue()
        digest.update(file.name.encode("utf-8", errors="ignore"))
        digest.update(str(len(data)).encode("utf-8"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def chat_history_path(signature):
    return CHAT_DIR / f"{safe_filename(signature)}.json"


def default_chat_message():
    return [
        {
            "role": "assistant",
            "content": "I indexed your documents with FAISS. Ask me anything about them.",
        }
    ]


def load_chat_history(signature):
    path = chat_history_path(signature)
    if not path.exists():
        return default_chat_message()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_chat_message()


def save_chat_history(signature, messages):
    ensure_storage_dirs()
    chat_history_path(signature).write_text(
        json.dumps(messages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def extract_text_from_file(file):
    name = file.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = pypdf.PdfReader(file)
            page_texts = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    page_texts.append(f"[Page {page_number}]\n{text}")
            return "\n\n".join(page_texts)
        if name.endswith(".docx"):
            document = docx.Document(file)
            return "\n".join(p.text for p in document.paragraphs if p.text.strip())
        if name.endswith(".txt"):
            data = file.read()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1")
        if name.endswith(".csv"):
            dataframe = pd.read_csv(file)
            return dataframe.to_string(index=False)
        if name.endswith((".xlsx", ".xls")):
            dataframe = pd.read_excel(file)
            return dataframe.to_string(index=False)
    except Exception as exc:
        st.error(f"Error reading {file.name}: {exc}")
    return ""


def recursive_split_text(text, filename, chunk_size=1000, overlap=200):
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_recursive(text_to_split, separators_list):
        if len(text_to_split) <= chunk_size:
            return [text_to_split.strip()]

        separator = separators_list[0]
        next_separators = separators_list[1:]

        if not next_separators:
            step = max(1, chunk_size - overlap)
            return [
                text_to_split[i : i + chunk_size].strip()
                for i in range(0, len(text_to_split), step)
            ]

        pieces = text_to_split.split(separator)
        chunks = []
        current = ""

        for piece in pieces:
            candidate = f"{current}{separator}{piece}" if current else piece
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())

            if len(piece) > chunk_size:
                chunks.extend(split_recursive(piece, next_separators))
                current = ""
            else:
                previous = chunks[-1] if chunks else ""
                prefix = previous[-overlap:] if previous else ""
                candidate = f"{prefix}{separator}{piece}" if prefix else piece
                current = candidate if len(candidate) <= chunk_size else piece

        if current:
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk]

    chunks = split_recursive(text, separators)
    return [
        {
            "id": f"{filename}::chunk-{index + 1}",
            "text": chunk,
            "source": filename,
            "chunk_number": index + 1,
        }
        for index, chunk in enumerate(chunks)
    ]


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_faiss_index(embeddings):
    normalized = embeddings.astype("float32")
    faiss.normalize_L2(normalized)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index


def cache_paths(signature):
    base = CACHE_DIR / signature
    return {
        "index": base.with_suffix(".faiss"),
        "metadata": base.with_suffix(".pkl"),
    }


def save_index_cache(signature, index, chunks, embeddings, file_stats):
    ensure_storage_dirs()
    paths = cache_paths(signature)
    faiss.write_index(index, str(paths["index"]))
    with paths["metadata"].open("wb") as handle:
        pickle.dump(
            {
                "chunks": chunks,
                "embeddings": embeddings,
                "file_stats": file_stats,
                "created_at": time.time(),
            },
            handle,
        )


def load_index_cache(signature):
    paths = cache_paths(signature)
    if not paths["index"].exists() or not paths["metadata"].exists():
        return None
    try:
        index = faiss.read_index(str(paths["index"]))
        with paths["metadata"].open("rb") as handle:
            metadata = pickle.load(handle)
        return {
            "faiss_index": index,
            "chunks": metadata["chunks"],
            "embeddings": metadata["embeddings"],
            "file_stats": metadata["file_stats"],
            "loaded_from_cache": True,
        }
    except Exception:
        return None


def build_document_index(files, model, signature):
    cached = load_index_cache(signature)
    if cached:
        return cached

    chunks = []
    file_stats = []
    for file in files:
        raw_text = extract_text_from_file(file)
        if raw_text.strip():
            file_chunks = recursive_split_text(raw_text, file.name)
            chunks.extend(file_chunks)
            file_stats.append(
                {
                    "name": file.name,
                    "size": f"{file.size / 1024:.1f} KB",
                    "chunks": len(file_chunks),
                }
            )

    if not chunks:
        return None

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")
    index = build_faiss_index(embeddings)
    save_index_cache(signature, index, chunks, embeddings, file_stats)

    return {
        "faiss_index": index,
        "chunks": chunks,
        "embeddings": embeddings,
        "file_stats": file_stats,
        "loaded_from_cache": False,
    }


def expand_query(query):
    prompt = f"""Rewrite this search query into three short retrieval queries.
Keep the original meaning and include synonyms or related terms only when useful.
Return one query per line.

Query: {query}"""
    try:
        response = genai.GenerativeModel(GEMINI_MODEL_NAME).generate_content(prompt)
        expansions = [
            line.strip("-* 0123456789.").strip()
            for line in response.text.splitlines()
            if line.strip()
        ]
        return [query] + expansions[:3]
    except Exception:
        return [query]


def retrieve_candidates(query, model, document_index, use_hybrid, use_expansion):
    chunks = document_index["chunks"]
    queries = expand_query(query) if use_expansion else [query]
    candidate_scores = {}

    for expanded_query in queries:
        query_embedding = np.asarray([model.encode(expanded_query)], dtype="float32")
        faiss.normalize_L2(query_embedding)
        distances, indices = document_index["faiss_index"].search(
            query_embedding,
            min(TOP_K_RETRIEVAL, len(chunks)),
        )

        for score, index in zip(distances[0], indices[0]):
            if index == -1:
                continue
            candidate_scores[index] = max(candidate_scores.get(index, 0), float(score))

    if use_hybrid:
        tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(tokenize(query))
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
        top_bm25_indices = np.argsort(bm25_scores)[-TOP_K_RETRIEVAL:][::-1]
        for index in top_bm25_indices:
            hybrid_score = 0.35 * float(bm25_scores[index])
            candidate_scores[index] = candidate_scores.get(index, 0) + hybrid_score

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    return [index for index, _score in ranked[:TOP_K_RETRIEVAL]]


def rerank_candidates(query, candidate_indices, chunks):
    if not candidate_indices:
        return []
    reranker = load_reranker()
    pairs = [(query, chunks[index]["text"]) for index in candidate_indices]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(candidate_indices, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [index for index, _score in ranked]


def build_context(selected_indices, chunks):
    context_parts = []
    source_parts = []
    for rank, index in enumerate(selected_indices, start=1):
        chunk = chunks[index]
        citation_id = f"{chunk['source']} - chunk {chunk['chunk_number']}"
        context_parts.append(f"[{rank}] Source: {citation_id}\n{chunk['text']}")
        snippet = chunk["text"].strip()
        if len(snippet) > 900:
            snippet = f"{snippet[:900]}..."
        source_parts.append(f"**[{rank}] {citation_id}**\n\n> {snippet}")
    return "\n\n---\n\n".join(context_parts), "\n\n---\n\n".join(source_parts)


def recent_conversation(messages, max_messages=6):
    recent = messages[-max_messages:]
    return "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in recent
    )


def stream_gemini_response(prompt):
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response_stream = model.generate_content(prompt, stream=True)
    for chunk in response_stream:
        text = getattr(chunk, "text", "")
        if text:
            yield text


def friendly_gemini_error(error):
    message = str(error)
    lower_message = message.lower()
    if "429" in message or "quota" in lower_message or "rate" in lower_message:
        retry_match = re.search(r"retry[_ ]delay\s*\{\s*seconds:\s*(\d+)", message, re.IGNORECASE)
        retry_seconds = retry_match.group(1) if retry_match else "30"
        return (
            f"Gemini is rate-limiting the free tier right now. "
            f"Please wait about {retry_seconds} seconds and ask again. "
            "Tip: keep Query expansion turned off to use only one Gemini request per answer."
        )
    return f"Error calling Gemini API: {message}"


if not gemini_key:
    st.warning("Please enter your Google Gemini API key in the sidebar to get started.")
else:
    genai.configure(api_key=gemini_key)

    if uploaded_files:
        ensure_storage_dirs()
        signature = file_signature(uploaded_files)

        with st.spinner("Loading local embedding model..."):
            embedding_model = load_embedding_model()

        if st.session_state.get("file_signature") != signature:
            with st.spinner("Indexing documents with FAISS..."):
                document_index = build_document_index(uploaded_files, embedding_model, signature)
                if not document_index:
                    st.error("No extractable text was found in the uploaded files.")
                    st.stop()
                st.session_state["document_index"] = document_index
                st.session_state["file_signature"] = signature
                st.session_state["messages"] = load_chat_history(signature)

        document_index = st.session_state["document_index"]
        chunks = document_index["chunks"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents", len(uploaded_files))
        col2.metric("Chunks", len(chunks))
        col3.metric("Vector Search", "FAISS")
        col4.metric("Index Status", "Cached" if document_index["loaded_from_cache"] else "Fresh")

        with st.expander("Processed files and statistics", expanded=False):
            for stat in document_index.get("file_stats", []):
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <strong>{stat['name']}</strong> &nbsp;|&nbsp; Size:
                        <code>{stat['size']}</code> &nbsp;|&nbsp; Chunks:
                        <code>{stat['chunks']}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if "messages" not in st.session_state:
            st.session_state["messages"] = load_chat_history(signature)

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("sources"):
                    with st.expander("Sources and citations", expanded=False):
                        st.markdown(message["sources"])

        if user_query := st.chat_input("Ask a question across all documents..."):
            st.session_state["messages"].append({"role": "user", "content": user_query})
            save_chat_history(signature, st.session_state["messages"])

            with st.chat_message("user"):
                st.write(user_query)

            with st.spinner("Retrieving and ranking evidence..."):
                candidate_indices = retrieve_candidates(
                    user_query,
                    embedding_model,
                    document_index,
                    use_hybrid_search,
                    use_query_expansion,
                )
                if use_reranker:
                    candidate_indices = rerank_candidates(user_query, candidate_indices, chunks)
                selected_indices = candidate_indices[:top_k_context]
                relevant_context, sources_string = build_context(selected_indices, chunks)

            prompt = f"""You are a careful RAG assistant.
Answer the user's question using only the provided document context.
Use concise, professional language.
Add inline citations like [1] or [2] when you use evidence.
If the answer is not supported by the context, say that the documents do not contain enough information.

Recent conversation:
{recent_conversation(st.session_state["messages"])}

Document context:
{relevant_context}

Question:
{user_query}

Answer:"""

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                streamed_text = ""
                try:
                    for text_chunk in stream_gemini_response(prompt):
                        streamed_text += text_chunk
                        message_placeholder.markdown(streamed_text)
                except Exception as exc:
                    streamed_text = friendly_gemini_error(exc)
                    message_placeholder.error(streamed_text)

                if sources_string:
                    with st.expander("Sources and citations", expanded=False):
                        st.markdown(sources_string)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": streamed_text,
                    "sources": sources_string,
                }
            )
            save_chat_history(signature, st.session_state["messages"])
    else:
        st.info("Upload one or more document files in the sidebar to begin chatting.")
