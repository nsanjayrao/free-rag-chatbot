# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run free_rag_chatbot.py
```

Requires API keys in `.streamlit/secrets.toml` (gitignored — copy from `.streamlit/secrets.toml.example`):
```toml
GEMINI_API_KEY = "..."
HF_API_TOKEN = "..."
```

There are no tests, linter config, or build steps. The entire app is a single file.

## Architecture

Everything lives in `free_rag_chatbot.py`. The file executes top-to-bottom as a Streamlit script on every browser interaction. Sections in order:

1. **Constants** — model names, cache dirs, limits. To swap the HuggingFace model, change `HF_MODEL_NAME` (line ~29); `HF_ENDPOINT` is derived from it automatically.
2. **Page config & CSS** — Apple-style minimalist theme with ChatGPT-style chat bubbles. All styling is inline `st.markdown` CSS.
3. **Secrets** — `GEMINI_API_KEY` and `HF_API_TOKEN` loaded from `st.secrets`.
4. **Sidebar** — provider selector (Gemini / HuggingFace), file uploader, retrieval toggles.
5. **Functions** — pure Python, no side effects on import. Defined before use.
6. **Main area** — three rendering states based on `api_ready` and `uploaded_files`:
   - No key → welcome/feature screen
   - Key set, no files → upload prompt screen
   - Files uploaded → metrics row + chat UI

## RAG Pipeline (data flow)

```
uploaded files
  → extract_sections_from_file()     # per-format: PDF pages, DOCX paragraphs, CSV row chunks
  → recursive_split_text()           # overlap-aware recursive splitter, 1000 char chunks / 200 overlap
  → SentenceTransformer.encode()     # local CPU embeddings, all-MiniLM-L6-v2
  → build_faiss_index()              # IndexFlatIP with L2 normalisation (cosine similarity)
  → BM25Okapi()                      # built once, cached in document_index dict
  → [saved to .rag_cache/ by signature hash]

on each query:
  → retrieve_candidates()            # FAISS search + optional BM25 merge (weight 0.35)
  → optional expand_query()          # 1 extra LLM call → 3 variant queries, all searched
  → rerank_candidates()              # CrossEncoder scores, replaces combined score
  → build_context()                  # top-k chunks → numbered context string + citation metadata
  → build_prompt()                   # context + last 6 turns of history → LLM
  → stream_gemini_response()         # or stream_hf_response() — SSE streaming
```

## Key Design Decisions

**Index caching:** `file_signature()` hashes file names + sizes + SHA256 of content + `PARSER_VERSION`. If the signature matches `.rag_cache/<sig>.faiss` + `.rag_cache/<sig>.pkl`, indexing is skipped entirely. Bump `PARSER_VERSION` to invalidate all caches after parser changes.

**BM25 is not persisted to disk** — only FAISS and chunk metadata are serialised. BM25 is rebuilt from cached chunks on load (fast) and stored in the in-memory `document_index` dict to avoid rebuilding on every query.

**LLM providers are interchangeable** — both use SSE streaming via `urllib.request` (no extra HTTP library). `stream_llm_response()` dispatches by provider string. HuggingFace uses the OpenAI-compatible router endpoint at `router.huggingface.co/hf-inference/v1/chat/completions` (the old `api-inference.huggingface.co` endpoint is deprecated and DNS-dead).

**Session state keys:** `file_signature`, `document_index`, `messages`. When `file_signature` changes (new upload), the index and chat history are both reloaded.

**Chat history** is persisted to `.chat_history/<sig>.json` keyed by the same file signature. Switching document sets automatically loads the matching history.

## LLM Provider Details

| Provider | Model constant | Free tier |
|---|---|---|
| Gemini | `GEMINI_MODEL_NAME = "gemini-2.5-flash"` | Google AI Studio free tier, rate-limited |
| HuggingFace | `HF_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"` | HF Serverless Inference, free Read token, ~few hundred req/hr |

To change the HuggingFace model, update only `HF_MODEL_NAME` at the top of the file — `HF_ENDPOINT` is constructed from it.

## File Structure

```
free_rag_chatbot.py          # entire app
requirements.txt             # pinned deps
sample_docs/                 # demo files for testing (txt, csv)
.streamlit/
    secrets.toml             # gitignored — real keys go here
    secrets.toml.example     # committed template
.rag_cache/                  # gitignored — FAISS indexes + pickle metadata
.chat_history/               # gitignored — JSON chat history per document set
```
