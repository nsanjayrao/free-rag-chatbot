# 🤖 Free Multi-File & Multi-Format RAG Analyzer

An interactive, high-performance web application built with **Streamlit** that allows users to upload multiple documents of various formats (PDF, DOCX, TXT, CSV, XLSX, XLS) and run semantic intelligence queries over all of them concurrently for **100% free**.

This project features a hybrid RAG architecture designed to optimize resources: it runs an open-source, local **Recursive Character Text Splitter** and **CPU embeddings** (no API costs or chunk leakage) and connects with the **Google Gemini 3.5 Flash Free Tier** for natural language reasoning.

---

## 🌟 Key Features
* **Zero Cost Architecture:** Uses open-source local CPU embeddings and Google's free-tier API.
* **Recursive Character Chunking:** Uses a custom recursive splitting algorithm that respects paragraph, sentence, and word boundaries to preserve contextual cohesion and avoid cutting sentences in half.
* **Multi-File Uploads:** Upload multiple files at the same time and run cross-document searches.
* **Multi-Format Parsing Support:** Seamlessly parses PDF, DOCX, TXT, CSV, and Excel (XLSX, XLS) files.
* **Premium SaaS Dashboard UI:** Integrated statistics cards displaying document processing metrics, along with an interactive collapsible document previewer.
* **Premium Citation UX:** Keeps the conversational interface clean by tucking the exact retrieved source snippets and filenames inside styled, collapsible reference cards under each response.
* **Local Embeddings (Privacy-Friendly):** Text chunks are embedded locally using the `all-MiniLM-L6-v2` model from HuggingFace—no document chunks are sent to external services for vectorization.

---

## 🛠️ Technology Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Document Parsers:** [pypdf](https://pypdf.readthedocs.io/en/stable/), [python-docx](https://python-docx.readthedocs.io/en/latest/), and [Pandas](https://pandas.pydata.org/) (for CSV/Excel)
* **Local Embeddings:** [Sentence-Transformers](https://sbert.net/) (`all-MiniLM-L6-v2`)
* **Vector Math:** [NumPy](https://numpy.org/) (for fast cosine similarity calculations)
* **LLM API:** [Google Generative AI](https://aistudio.google.com/) (Gemini 3.5 Flash)

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your computer.

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/free-rag-chatbot.git
cd free-rag-chatbot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a Free Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Click **Get API Key** and copy your free-tier key.

### 5. Run the Application
```bash
streamlit run free_rag_chatbot.py
```
This will launch the app in your default browser at `http://localhost:8501`.

---

## 🌐 Free Cloud Deployment

This app is pre-configured to be deployed for free on **Streamlit Community Cloud**:
1. Push this repository to your GitHub account.
2. Go to [Streamlit Share](https://share.streamlit.io/) and log in with GitHub.
3. Click **New App**, select your repository, branch (`main`), and set the Main file path to `free_rag_chatbot.py`.
4. Click **Advanced Settings** > **Secrets**, and add your API key in TOML format:
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```
5. Click **Deploy**—your RAG Analyzer will be live on a public URL in minutes!

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.