# 🤖 Free Retrieval-Augmented Generation (RAG) Chatbot

An interactive web application built with *Streamlit* that allows users to upload a PDF document and have intelligent, context-aware conversations with its contents for *100% free*.

This project is designed as a hybrid RAG architecture to optimize resources: it runs *local, open-source embeddings* on CPU (requiring zero API costs) and connects with the *Google Gemini 1.5 Flash Free Tier* for natural language reasoning.

---

## 🌟 Key Features
* *Zero Cost Architecture:* Uses open-source local CPU embeddings and Google's free-tier API.
* *Local Embeddings (Privacy-Friendly):* Text chunks are embedded locally using the all-MiniLM-L6-v2 model from HuggingFace—no document chunks are sent to external services for vectorization.
* *Smart Semantic Retrieval:* Calculates cosine similarity to find and retrieve the top-3 most relevant sections of your PDF.
* *Gemini 1.5 Flash Integration:* Leverages Google's high-speed, large-context model to formulate answers based strictly on retrieved document context.
* *Conversational Streamlit UI:* Features an intuitive, interactive chat interface with chat history persistence.

---

## 🛠️ Technology Stack
* *Frontend:* [Streamlit](https://streamlit.io/)
* *PDF Parser:* [pypdf](https://pypdf.readthedocs.io/en/stable/)
* *Local Embeddings:* [Sentence-Transformers](https://sbert.net/) (all-MiniLM-L6-v2)
* *Vector Math:* [NumPy](https://numpy.org/) (for fast cosine similarity calculations)
* *LLM API:* [Google Generative AI](https://aistudio.google.com/) (Gemini 1.5 Flash)

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your computer.

### 2. Clone the Repository
bash
git clone https://github.com/YOUR_USERNAME/free-rag-chatbot.git
cd free-rag-chatbot


### 3. Install Dependencies
bash
pip install -r requirements.txt


### 4. Get a Free Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Click *Get API Key* and copy your free-tier key.

### 5. Run the Application
bash
streamlit run free_rag_chatbot.py

This will launch the app in your default browser at http://localhost:8501.

---

## 🌐 Free Cloud Deployment

This app is pre-configured to be deployed for free on *Streamlit Community Cloud*:
1. Push this repository to your GitHub account.
2. Go to [Streamlit Share](https://share.streamlit.io/) and log in with GitHub.
3. Click *New App*, select your repository, branch (main), and set the Main file path to free_rag_chatbot.py.
4. Click *Deploy*—your RAG Chatbot will be live on a public URL in minutes!

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.