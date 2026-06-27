import hashlib
import json
import pickle
import re
import time
import urllib.error
import urllib.request
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
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
PARSER_VERSION = "v2-page-sheet-citations"
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
    "Upload PDF, DOCX, TXT, CSV, or Excel files and ask grounded questions with FAISS retrieval, hybrid search, reranking, memory, streaming, export, and page-aware citations."
)

try:
    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    gemini_key = ""

with st.sidebar:
    st.header("1. Answer Model")
    llm_provider = st.selectbox(
        "Provider",
        ["Gemini", "Ollama (local)"],
        help="Gemini works on Streamlit Cloud. Ollama is for local unlimited testing.",
    )

    if llm_provider == "Gemini":
        if not gemini_key:
            gemini_key = st.text_input(
                "Enter Gemini API Key",
                type="password",
                help="Create a free key in Google AI Studio.",
            )
        else:
            st.success("API key loaded from Streamlit secrets.")
    else:
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)
        st.caption("Run Ollama locally first, then use this option for quota-free testing.")

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
    use_query_expansion = st.toggle(
        "Query expansion",
        value=False,
        help="Uses one extra Gemini request per question. Keep this off on the free tier unless you need broader retrieval.",
    )
    top_k_context = st.slider("Cited chunks", 2, 8, TOP_K_CONTEXT)
    show_diagnostics = st.toggle("Show retrieval diagnostics", value=False)


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
    digest.update(PARSER_VERSION.encode("utf-8"))
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


def export_chat_markdown(messages):
    lines = ["# RAG Chat Export", ""]
    for message in messages:
        role = message.get("role", "message").title()
        lines.extend([f"## {role}", "", message.get("content", ""), ""])
        if message.get("sources"):
            lines.extend(["### Sources", "", message["sources"], ""])
    return "\n".join(lines)


def make_source_label(metadata):
    label = metadata.get("source", "Unknown source")
    details = []
    if metadata.get("page"):
        details.append(f"page {metadata['page']}")
    if metadata.get("sheet"):
        details.append(f"sheet {metadata['sheet']}")
    if metadata.get("row_range"):
        details.append(f"rows {metadata['row_range']}")
    if metadata.get("paragraph_range"):
        details.append(f"paragraphs {metadata['paragraph_range']}")
    if details:
        label += " (" + ", ".join(details) + ")"
    return label


def dataframe_to_sections(dataframe, filename, sheet_name=None, rows_per_section=40):
    sections = []
    total_rows = len(dataframe)
    if total_rows == 0:
        metadata = {"source": filename, "type": "spreadsheet", "sheet": sheet_name, "row_range": "empty"}
        sections.append({"text": dataframe.to_string(index=False), "metadata": metadata})
        return sections

    for start in range(0, total_rows, rows_per_section):
        end = min(start + rows_per_section, total_rows)
        subset = dataframe.iloc[start:end]
        metadata = {
            "source": filename,
            "type": "spreadsheet",
            "sheet": sheet_name,
            "row_range": f"{start + 1}-{end}",
        }
        sections.append({"text": subset.to_string(index=False), "metadata": metadata})
    return sections


def extract_sections_from_file(file):
    name = file.name
    lower_name = name.lower()
    sections = []
    try:
        if lower_name.endswith(".pdf"):
            reader = pypdf.PdfReader(file)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    sections.append(
                        {
                            "text": text,
                            "metadata": {"source": name, "type": "pdf", "page": page_number},
                        }
                    )
            return sections

        if lower_name.endswith(".docx"):
            document = docx.Document(file)
            paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
            for start in range(0, len(paragraphs), 20):
                end = min(start + 20, len(paragraphs))
                sections.append(
                    {
                        "text": "\n".join(paragraphs[start:end]),
                        "metadata": {
                            "source": name,
                            "type": "docx",
                            "paragraph_range": f"{start + 1}-{end}",
                        },
                    }
                )
            return sections

        if lower_name.endswith(".txt"):
            data = file.read()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
            return [{"text": text, "metadata": {"source": name, "type": "text"}}]

        if lower_name.endswith(".csv"):
            dataframe = pd.read_csv(file)
            return dataframe_to_sections(dataframe, name)

        if lower_name.endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(file, sheet_name=None)
            for sheet_name, dataframe in sheets.items():
                sections.extend(dataframe_to_sections(dataframe, name, sheet_name=sheet_name))
            return sections
    except Exception as exc:
        st.error(f"Error reading {file.name}: {exc}")
    return sections


def recursive_split_text(text, metadata, chunk_size=1000, overlap=200):
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

    raw_chunks = split_recursive(text, separators)
    source = metadata.get("source", "document")
    return [
        {
            "id": f"{source}::chunk-{index + 1}::{hashlib.sha1(chunk.encode('utf-8', errors='ignore')).hexdigest()[:8]}",
            "text": chunk,
            "source": source,
            "source_label": make_source_label(metadata),
            "chunk_number": index + 1,
            "metadata": metadata.copy(),
        }
        for index, chunk in enumerate(raw_chunks)
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


def remove_index_cache(signature):
    for path in cache_paths(signature).values():
        if path.exists():
            path.unlink()


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
                "parser_version": PARSER_VERSION,
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
        if metadata.get("parser_version") != PARSER_VERSION:
            return None
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
        sections = extract_sections_from_file(file)
        file_chunk_count = 0
        for section in sections:
            if section["text"].strip():
                section_chunks = recursive_split_text(section["text"], section["metadata"])
                chunks.extend(section_chunks)
                file_chunk_count += len(section_chunks)
        if file_chunk_count:
            file_stats.append(
                {
                    "name": file.name,
                    "size": f"{file.size / 1024:.1f} KB",
                    "chunks": file_chunk_count,
                    "sections": len(sections),
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
            details = candidate_scores.setdefault(
                index,
                {"index": index, "semantic": 0.0, "bm25": 0.0, "combined": 0.0},
            )
            details["semantic"] = max(details["semantic"], float(score))

    if use_hybrid:
        tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(tokenize(query))
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
        top_bm25_indices = np.argsort(bm25_scores)[-TOP_K_RETRIEVAL:][::-1]
        for index in top_bm25_indices:
            details = candidate_scores.setdefault(
                int(index),
                {"index": int(index), "semantic": 0.0, "bm25": 0.0, "combined": 0.0},
            )
            details["bm25"] = max(details["bm25"], float(bm25_scores[index]))

    for details in candidate_scores.values():
        details["combined"] = details["semantic"] + (0.35 * details["bm25"])

    ranked = sorted(candidate_scores.values(), key=lambda item: item["combined"], reverse=True)
    return ranked[:TOP_K_RETRIEVAL]


def rerank_candidates(query, candidates, chunks):
    if not candidates:
        return []
    reranker = load_reranker()
    pairs = [(query, chunks[candidate["index"]]["text"]) for candidate in candidates]
    scores = reranker.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["rerank"] = float(score)
    return sorted(candidates, key=lambda item: item.get("rerank", item["combined"]), reverse=True)


def build_context(selected_candidates, chunks):
    context_parts = []
    source_parts = []
    diagnostics = []
    for rank, candidate in enumerate(selected_candidates, start=1):
        chunk = chunks[candidate["index"]]
        citation_id = f"{chunk['source_label']} - chunk {chunk['chunk_number']}"
        context_parts.append(f"[{rank}] Source: {citation_id}\n{chunk['text']}")
        snippet = chunk["text"].strip()
        if len(snippet) > 900:
            snippet = f"{snippet[:900]}..."
        score_text = f"semantic={candidate.get('semantic', 0):.3f}, bm25={candidate.get('bm25', 0):.3f}, combined={candidate.get('combined', 0):.3f}"
        if "rerank" in candidate:
            score_text += f", rerank={candidate['rerank']:.3f}"
        source_parts.append(f"**[{rank}] {citation_id}**\n\nScore: {score_text}\n\n> {snippet}")
        diagnostics.append(
            {
                "rank": rank,
                "source": citation_id,
                "semantic": round(candidate.get("semantic", 0), 4),
                "bm25": round(candidate.get("bm25", 0), 4),
                "combined": round(candidate.get("combined", 0), 4),
                "rerank": round(candidate["rerank"], 4) if "rerank" in candidate else None,
            }
        )
    return "\n\n---\n\n".join(context_parts), "\n\n---\n\n".join(source_parts), diagnostics


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


def stream_ollama_response(prompt, base_url, model_name):
    endpoint = base_url.rstrip("/") + "/api/generate"
    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": True}).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        for line in response:
            if not line.strip():
                continue
            data = json.loads(line.decode("utf-8"))
            if data.get("response"):
                yield data["response"]
            if data.get("done"):
                break


def stream_llm_response(prompt, provider, **kwargs):
    if provider == "Gemini":
        yield from stream_gemini_response(prompt)
    else:
        yield from stream_ollama_response(prompt, kwargs["ollama_url"], kwargs["ollama_model"])


def friendly_llm_error(error, provider):
    message = str(error)
    lower_message = message.lower()
    if provider == "Gemini" and ("429" in message or "quota" in lower_message or "rate" in lower_message):
        retry_match = re.search(r"retry[_ ]delay\s*\{\s*seconds:\s*(\d+)", message, re.IGNORECASE)
        retry_seconds = retry_match.group(1) if retry_match else "30"
        return (
            f"Gemini is rate-limiting the free tier right now. "
            f"Please wait about {retry_seconds} seconds and ask again. "
            "Tip: keep Query expansion turned off to use only one Gemini request per answer."
        )
    if provider != "Gemini":
        return (
            "Could not reach Ollama. Make sure Ollama is running locally, "
            f"the model is pulled, and the URL is correct. Details: {message}"
        )
    return f"Error calling Gemini API: {message}"


def build_prompt(messages, context, query):
    return f"""You are a careful RAG assistant.
Answer the user's question using only the provided document context.
Use concise, professional language.
Add inline citations like [1] or [2] when you use evidence.
If the answer is not supported by the context, say that the documents do not contain enough information.

Recent conversation:
{recent_conversation(messages)}

Document context:
{context}

Question:
{query}

Answer:"""


if llm_provider == "Gemini" and not gemini_key:
    st.warning("Please enter your Google Gemini API key in the sidebar to get started.")
elif llm_provider == "Gemini":
    genai.configure(api_key=gemini_key)

if (llm_provider == "Gemini" and gemini_key) or llm_provider != "Gemini":
    if uploaded_files:
        ensure_storage_dirs()
        signature = file_signature(uploaded_files)

        if st.sidebar.button("Clear chat history", use_container_width=True):
            st.session_state["messages"] = default_chat_message()
            save_chat_history(signature, st.session_state["messages"])
            st.rerun()

        if st.sidebar.button("Rebuild index cache", use_container_width=True):
            remove_index_cache(signature)
            st.session_state.pop("file_signature", None)
            st.session_state.pop("document_index", None)
            st.rerun()

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

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Documents", len(uploaded_files))
        col2.metric("Chunks", len(chunks))
        col3.metric("Vector Search", "FAISS")
        col4.metric("Index Status", "Cached" if document_index["loaded_from_cache"] else "Fresh")
        col5.metric("Answer Model", llm_provider.split()[0])

        with st.expander("Processed files and statistics", expanded=False):
            for stat in document_index.get("file_stats", []):
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <strong>{stat['name']}</strong> &nbsp;|&nbsp; Size:
                        <code>{stat['size']}</code> &nbsp;|&nbsp; Sections:
                        <code>{stat['sections']}</code> &nbsp;|&nbsp; Chunks:
                        <code>{stat['chunks']}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if "messages" not in st.session_state:
            st.session_state["messages"] = load_chat_history(signature)

        st.download_button(
            "Download chat transcript",
            data=export_chat_markdown(st.session_state["messages"]),
            file_name="rag_chat_transcript.md",
            mime="text/markdown",
        )

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("sources"):
                    with st.expander("Sources and citations", expanded=False):
                        st.markdown(message["sources"])
                if show_diagnostics and message.get("diagnostics"):
                    with st.expander("Retrieval diagnostics", expanded=False):
                        st.dataframe(pd.DataFrame(message["diagnostics"]), use_container_width=True)

        if user_query := st.chat_input("Ask a question across all documents..."):
            st.session_state["messages"].append({"role": "user", "content": user_query})
            save_chat_history(signature, st.session_state["messages"])

            with st.chat_message("user"):
                st.write(user_query)

            with st.spinner("Retrieving and ranking evidence..."):
                candidates = retrieve_candidates(
                    user_query,
                    embedding_model,
                    document_index,
                    use_hybrid_search,
                    use_query_expansion,
                )
                if use_reranker:
                    candidates = rerank_candidates(user_query, candidates, chunks)
                selected_candidates = candidates[:top_k_context]
                relevant_context, sources_string, diagnostics = build_context(selected_candidates, chunks)

            prompt = build_prompt(st.session_state["messages"], relevant_context, user_query)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                streamed_text = ""
                try:
                    llm_kwargs = {}
                    if llm_provider != "Gemini":
                        llm_kwargs = {"ollama_url": ollama_url, "ollama_model": ollama_model}
                    for text_chunk in stream_llm_response(prompt, llm_provider, **llm_kwargs):
                        streamed_text += text_chunk
                        message_placeholder.markdown(streamed_text)
                except Exception as exc:
                    streamed_text = friendly_llm_error(exc, llm_provider)
                    message_placeholder.error(streamed_text)

                if sources_string:
                    with st.expander("Sources and citations", expanded=False):
                        st.markdown(sources_string)
                if show_diagnostics and diagnostics:
                    with st.expander("Retrieval diagnostics", expanded=False):
                        st.dataframe(pd.DataFrame(diagnostics), use_container_width=True)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": streamed_text,
                    "sources": sources_string,
                    "diagnostics": diagnostics,
                }
            )
            save_chat_history(signature, st.session_state["messages"])
    else:
        st.info("Upload one or more document files in the sidebar to begin chatting.")
