# 🔍 Agentic RAG Document Search

# Live Demo : https://docragsearch-xyz.streamlit.app/ 

A **Retrieval-Augmented Generation (RAG)** system powered by LangGraph, Groq LLM, and FAISS vector search — with a clean Streamlit UI. Ask natural language questions over your documents and get AI-generated answers with source citations.

---

## ✨ Features

- 📄 **Multi-source document ingestion** — URLs, PDFs, and TXT files
- 🧠 **Agentic RAG pipeline** built with [LangGraph](https://github.com/langchain-ai/langgraph)
- ⚡ **Groq LLM** (`qwen3-32b`) for ultra-fast inference
- 🔎 **FAISS vector store** with HuggingFace sentence embeddings
- 🤖 **ReAct Agent** with Wikipedia fallback tool
- 🖥️ **Streamlit UI** with search history and source document viewer
- ☁️ **One-click deploy** to Streamlit Cloud

---

## 🏗️ Project Structure

```
RAG PROJECT/
│
├── streamlit_app.py          # Main Streamlit UI
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── .env                      # Local API keys (never commit!)
├── .streamlit/
│   └── secrets.toml          # Streamlit secrets (never commit!)
│
├── data/
│   ├── attention.pdf         # Sample document
│   └── url.txt               # Sample URLs
│
└── src/
    ├── config/
    │   └── config.py         # App configuration & LLM setup
    ├── document_ingestion/
    │   └── document_processor.py  # Load & split documents
    ├── vectorstore/
    │   └── vectorstore.py    # FAISS embedding & retrieval
    ├── graph_builder/
    │   └── graph_builder.py  # LangGraph workflow builder
    ├── nodes/
    │   ├── nodes.py          # Basic RAG nodes
    │   └── reactnode.py      # ReAct agent nodes
    └── state/
        └── rag_state.py      # LangGraph state schema
```

---

## ⚙️ How It Works

```
User Question
     │
     ▼
┌─────────────┐
│  Retriever  │  ──► FAISS vector search over documents
└─────────────┘
     │
     ▼
┌─────────────┐
│  Responder  │  ──► Groq LLM generates answer from context
└─────────────┘
     │
     ▼
  Answer + Source Docs
```

The LangGraph workflow:
1. **Retrieve** — embeds the question and fetches top-k relevant chunks from FAISS
2. **Generate** — Groq LLM answers the question grounded in retrieved context
3. **(Optional ReAct)** — agent can also query Wikipedia for general knowledge

---

## 🚀 Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/HarshitKamriya/DOC_RAG_Search.git
cd DOC_RAG_Search
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file in the root:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```
> Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **"New app"** and select your repo
4. Set **Main file path** to `streamlit_app.py`
5. Set **Python version** to `3.12`
6. Click **"Advanced settings"** and paste your secrets:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   OPENAI_API_KEY = "your_openai_api_key"
   GOOGLE_API_KEY = "your_google_api_key"
   ```
7. Click **"Deploy!"** 🎉

---

## 🔧 Configuration

Edit `src/config/config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `groq:qwen/qwen3-32b` | LLM provider and model |
| `CHUNK_SIZE` | `500` | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `DEFAULT_URLS` | Lilian Weng's blog | URLs loaded at startup |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | [Groq](https://groq.com) — `qwen3-32b` |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| Document Loading | LangChain Community Loaders |
| UI | [Streamlit](https://streamlit.io) |
| Python | 3.12 |

---

## 📋 Requirements

See [`requirements.txt`](requirements.txt) for full list. Key packages:

```
langchain
langgraph
langchain-groq
langchain-huggingface
faiss-cpu
sentence-transformers
streamlit
pypdf
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push and open a Pull Request

---

## 📄 License

MIT License — feel free to use and modify for your own projects.

---

> Built with ❤️ using LangGraph + Groq + Streamlit
