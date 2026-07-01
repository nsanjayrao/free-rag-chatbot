# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run free_rag_chatbot.py
```

Requires API keys in `.streamlit/secrets.toml` (gitignored — copy from `.streamlit/secrets.toml.example`):
```toml
GEMINI_API_KEY = "..."
GROQ_API_KEY = "..."
```

There are no tests, linter config, or build steps. The entire app is a single file.

## Architecture

Everything lives in `free_rag_chatbot.py`. The file executes top-to-bottom as a Streamlit script on every browser interaction. Sections in order:

1. **Constants** — model names, cache dirs, limits. To swap the Groq model, change `GROQ_MODEL_NAME` (line ~29).
2. **Page config & CSS** — Apple-style theme with a black hero banner, iMessage-style chat bubbles, and a `@media (max-width: 768px)` block for phones. All styling is one inline `st.markdown` CSS blob. Default chat avatars are hidden via `display:none` (the `:has([data-testid="chatAvatarIcon-user"])` selector still works because the element stays in the DOM). A `.mobile-hint` div is `display:none` on desktop and shown on mobile to point users at the collapsed sidebar.
3. **Secrets** — `GEMINI_API_KEY` and `GROQ_API_KEY` loaded from `st.secrets`.
4. **Sidebar** — provider selector (Gemini / Groq), file uploader, retrieval toggles.
5. **Functions** — pure Python, no side effects on import. Defined before use.
6. **Main area** — an always-visible hero banner (also the first thing seen on phones), then three rendering states based on `api_ready` and `uploaded_files`:
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
  → stream_gemini_response()         # or stream_groq_response() — SSE streaming
```

## Key Design Decisions

**Index caching:** `file_signature()` hashes file names + sizes + SHA256 of content + `PARSER_VERSION`. If the signature matches `.rag_cache/<sig>.faiss` + `.rag_cache/<sig>.pkl`, indexing is skipped entirely. Bump `PARSER_VERSION` to invalidate all caches after parser changes.

**BM25 is not persisted to disk** — only FAISS and chunk metadata are serialised. BM25 is rebuilt from cached chunks on load (fast) and stored in the in-memory `document_index` dict to avoid rebuilding on every query.

**LLM providers are interchangeable** — both use SSE streaming via `urllib.request` (no extra HTTP library). `stream_llm_response()` dispatches by provider string. Groq uses the OpenAI-compatible endpoint `api.groq.com/openai/v1/chat/completions` with a plain bearer key (no permission scopes to configure, unlike HuggingFace). Both `stream_groq_response()` and `_groq_json_request()` (query expansion) surface the actual HTTP error body on failure.

**Session state keys:** `file_signature`, `document_index`, `messages`. When `file_signature` changes (new upload), the index and chat history are both reloaded.

**Chat history** is persisted to `.chat_history/<sig>.json` keyed by the same file signature. Switching document sets automatically loads the matching history.

**Transcript export** is a styled PDF via `export_chat_pdf()` (fpdf2, black header banner). fpdf core fonts are latin-1 only, so text is run through `_strip_markdown()` + `.encode("latin-1", "replace")` before writing — non-latin chars/emoji become `?`. The download button falls back to `export_chat_markdown()` if PDF generation raises.

## LLM Provider Details

| Provider | Model constant | Free tier |
|---|---|---|
| Gemini | `GEMINI_MODEL_NAME = "gemini-2.5-flash"` | Google AI Studio free tier, rate-limited |
| Groq | `GROQ_MODEL_NAME = "llama-3.3-70b-versatile"` | Free API key, no credit card, ~14.4k req/day, 30 req/min |

To change the Groq model, update only `GROQ_MODEL_NAME` at the top of the file. Available free models are listed at console.groq.com/docs/models.

## File Structure

```
free_rag_chatbot.py          # entire app
requirements.txt             # pinned deps
sample_docs/                 # demo files for testing (txt, csv)
.streamlit/
    secrets.toml             # gitignored — real keys go here (GEMINI_API_KEY, GROQ_API_KEY)
    secrets.toml.example     # committed template
.rag_cache/                  # gitignored — FAISS indexes + pickle metadata
.chat_history/               # gitignored — JSON chat history per document set
```
