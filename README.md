# Professional Multi-File RAG Analyzer

A Streamlit RAG application for asking grounded questions across PDF, DOCX, TXT, CSV, XLSX, and XLS files. The app now uses a more production-style retrieval pipeline: recursive chunking, local SentenceTransformer embeddings, a cached FAISS vector index, optional BM25 hybrid search, optional cross-encoder reranking, conversation memory, query expansion, streaming Gemini responses, and source citations.

## RAG Pipeline

PDF / DOCX / TXT / CSV / XLSX
-> recursive chunking
-> SentenceTransformer embeddings (all-MiniLM-L6-v2)
-> cached FAISS vector index
-> top-k retrieval
-> optional BM25 hybrid search
-> optional cross-encoder reranking
-> Gemini 3.5 Flash
-> answer with citations

## Key Features

- Multi-format parsing for PDF, Word, text, CSV, and Excel files.
- Recursive chunking that preserves paragraphs, sentence boundaries, and overlap.
- Local CPU embeddings with all-MiniLM-L6-v2, so document chunks are not sent out for embedding.
- FAISS vector search using normalized inner product similarity.
- Content-hash based FAISS index cache in .rag_cache, so unchanged uploads do not rebuild every run.
- Optional hybrid retrieval with BM25 plus FAISS for better lexical and semantic matching.
- Optional cross-encoder reranking with cross-encoder/ms-marco-MiniLM-L-6-v2.
- Optional Gemini-powered query expansion before retrieval.
- Conversation memory using recent chat turns in the answer prompt.
- Persistent per-document chat history saved in .chat_history.
- Streaming model responses and expandable citations for retrieved chunks.

## Technology Stack

- Frontend: Streamlit
- Document parsing: pypdf, python-docx, pandas, openpyxl, xlrd
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector database: FAISS
- Keyword retrieval: rank-bm25
- Reranking: sentence-transformers CrossEncoder
- LLM API: Google Generative AI Gemini 3.5 Flash

## Quick Start

1. Install Python 3.9 or newer.
2. Install dependencies:

       pip install -r requirements.txt

3. Create a free Gemini API key in Google AI Studio.
4. Run the app:

       streamlit run free_rag_chatbot.py

5. Add your Gemini key in the sidebar, or configure Streamlit secrets:

       GEMINI_API_KEY = "your_actual_api_key_here"

## Deployment Notes

The app is compatible with Streamlit Community Cloud. The first run may take longer because SentenceTransformer and cross-encoder models need to download. The generated .rag_cache and .chat_history folders are local runtime artifacts and are intentionally ignored by Git.

## Portfolio Highlights

This project now demonstrates the same core retrieval stages used in professional RAG systems: vector indexing, hybrid retrieval, reranking, prompt grounding, citations, streaming UX, and persistence. Recruiters can quickly recognize FAISS, BM25, and cross-encoder reranking as practical search and ranking components rather than a toy cosine-similarity demo.
