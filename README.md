# Professional Multi-File RAG Analyzer

A production-style RAG chatbot built with Streamlit, FAISS, SentenceTransformers, BM25, cross-encoder reranking, and Gemini. Upload PDF, DOCX, TXT, CSV, XLSX, or XLS files, then ask grounded questions with streaming answers and citations.

This project is designed as a portfolio-ready retrieval system rather than a basic document chatbot. It demonstrates document parsing, recursive chunking, local embeddings, vector indexing, hybrid retrieval, reranking, prompt grounding, citation UX, index caching, and persistent chat history.

## Live Demo

Add your deployed Streamlit link here after publishing:

~~~text
https://your-app-name.streamlit.app
~~~

## What It Does

- Upload multiple documents across common office formats.
- Extract and chunk text with overlap-aware recursive splitting.
- Embed chunks locally with `all-MiniLM-L6-v2`.
- Store and search vectors with FAISS.
- Blend semantic search with BM25 keyword search.
- Rerank retrieved chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Generate grounded answers with Gemini 3.5 Flash.
- Stream responses into the chat UI.
- Show citations with the exact retrieved source snippets.
- Cache FAISS indexes so repeated uploads are fast.
- Persist chat history per document set.

## Architecture

~~~mermaid
flowchart TD
    A[PDF / DOCX / TXT / CSV / XLSX] --> B[Text Extraction]
    B --> C[Recursive Chunking]
    C --> D[SentenceTransformer Embeddings<br/>all-MiniLM-L6-v2]
    D --> E[Cached FAISS Vector Index]
    C --> F[BM25 Keyword Index]
    E --> G[Top-k Semantic Retrieval]
    F --> H[Keyword Retrieval]
    G --> I[Hybrid Candidate Merge]
    H --> I
    I --> J[Cross-Encoder Reranking]
    J --> K[Grounded Context + Citations]
    K --> L[Gemini 3.5 Flash]
    L --> M[Streaming Answer]
    M --> N[Persistent Chat History]
~~~

## Demo Walkthrough

1. Start the app and enter a Gemini API key.
2. Upload one or more files, such as a PDF report, a DOCX policy, and a spreadsheet.
3. The app extracts text, builds chunks, creates local embeddings, and writes a FAISS index cache.
4. Ask a question like `What are the main risks mentioned across these documents?`
5. The assistant retrieves evidence, reranks it, streams an answer, and shows expandable citations.

## Example Questions

- `Summarize the uploaded documents in five bullet points.`
- `What are the key risks, deadlines, or obligations mentioned?`
- `Compare the financial figures in the spreadsheet.`
- `Which document discusses implementation details?`
- `What evidence supports this answer? Cite the source chunks.`
- `What information is missing from these documents?`

## Retrieval Pipeline

~~~text
Documents
  -> text extraction
  -> recursive chunking
  -> SentenceTransformer embeddings
  -> FAISS vector search
  -> optional BM25 hybrid retrieval
  -> optional query expansion
  -> optional cross-encoder reranking
  -> Gemini grounded generation
  -> answer with citations
~~~

## Key Features

- Multi-format parsing for PDF, Word, text, CSV, and Excel files.
- Recursive chunking that preserves paragraphs, sentence boundaries, and overlap.
- Local CPU embeddings so document chunks are not sent to an embedding API.
- FAISS vector search using normalized inner product similarity.
- Content-hash based FAISS index cache in `.rag_cache/`.
- Optional BM25 + FAISS hybrid retrieval for stronger lexical and semantic matching.
- Optional cross-encoder reranking for more relevant final context.
- Optional Gemini-powered query expansion for broader search coverage.
- Conversation memory using recent chat turns in the answer prompt.
- Persistent per-document chat history saved in `.chat_history/`.
- Streaming responses and expandable citation snippets.
- Free/open-source retrieval stack with only the final answer generation using Gemini.

## Technology Stack

| Layer | Tools |
| --- | --- |
| UI | Streamlit |
| Document parsing | pypdf, python-docx, pandas, openpyxl, xlrd |
| Chunking | Custom recursive splitter |
| Embeddings | sentence-transformers, all-MiniLM-L6-v2 |
| Vector search | FAISS |
| Keyword search | rank-bm25 |
| Reranking | SentenceTransformers CrossEncoder |
| LLM | Google Generative AI, Gemini 3.5 Flash |
| Persistence | Local FAISS cache and JSON chat history |

## Why This Is Portfolio-Ready

This project shows practical RAG engineering choices that appear in real systems:

- **FAISS indexing:** faster and more recognizable than manual cosine similarity loops.
- **Hybrid retrieval:** combines semantic similarity with exact keyword matching.
- **Reranking:** improves final context quality before generation.
- **Grounded prompting:** instructs the model to answer only from retrieved evidence.
- **Citations:** makes outputs easier to trust and inspect.
- **Caching:** avoids rebuilding embeddings and indexes for unchanged uploads.
- **Memory:** supports follow-up questions without losing conversation context.

## Quick Start

1. Install Python 3.9 or newer.
2. Install dependencies:

~~~bash
pip install -r requirements.txt
~~~

3. Create a free Gemini API key in Google AI Studio.
4. Run the app:

~~~bash
streamlit run free_rag_chatbot.py
~~~

5. Add your Gemini key in the sidebar, or configure Streamlit secrets:

~~~toml
GEMINI_API_KEY = "your_actual_api_key_here"
~~~

## Free-Tier Notes

Gemini free tier may rate-limit requests. Query expansion is off by default because it uses an extra Gemini request before answer generation. Keep it off for normal demos, then turn it on only when you want broader retrieval.

## Deployment

This app can be deployed on Streamlit Community Cloud.

1. Push this repository to GitHub.
2. Create a new Streamlit app from the repository.
3. Set the main file path to `free_rag_chatbot.py`.
4. Add `GEMINI_API_KEY` in Streamlit secrets.
5. Add the deployed URL to the Live Demo section above.

## Local Runtime Artifacts

The app creates these folders during use:

- `.rag_cache/` for FAISS indexes and chunk metadata.
- `.chat_history/` for persistent chat transcripts.

Both are ignored by Git because they are generated locally and can become large.
