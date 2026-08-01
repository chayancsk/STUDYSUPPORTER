# 📚 Multi-Document Research Assistant

A local, CPU-friendly Retrieval-Augmented Generation (RAG) app built with Streamlit and LangChain. Upload multiple PDF or text files, ask questions across all of them at once, and get answers grounded in your documents — complete with source citations.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![LangChain](https://img.shields.io/badge/langchain-RAG-green)

---

## ✨ Features

- **Multi-document upload** — ingest several PDFs and/or `.txt` files in one session
- **Cross-document Q&A** — ask a single question and get answers synthesized across all uploaded sources
- **Source citations** — every answer links back to the specific document(s) it was drawn from
- **Fully local & CPU-friendly** — no external LLM API required; runs entirely on your machine
- **Fast semantic search** — powered by FAISS vector indexing

---

## 🏗️ How It Works

This app follows a standard RAG (Retrieval-Augmented Generation) pipeline:

```
Upload Files → Load & Parse → Chunk Text → Embed Chunks → Index in FAISS
                                                                  ↓
                                          User Question → Retrieve Top-K Chunks
                                                                  ↓
                                                    LLM generates grounded answer
                                                                  ↓
                                                  Answer + Source Citations
```

**Pipeline breakdown:**

1. **Document Loading** — `PyPDFLoader` handles PDFs, `TextLoader` handles `.txt` files. Each document is tagged with its original filename as metadata.
2. **Chunking** — Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter` to preserve context across boundaries.
3. **Embeddings** — Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`.
4. **Vector Store** — Embeddings are indexed with **FAISS** for fast similarity search.
5. **Retrieval** — On each query, the top-k most relevant chunks are retrieved.
6. **Generation** — [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) (~250M params) generates the answer, run locally via a HuggingFace `pipeline`.
7. **Citations** — Source documents used for each answer are returned and displayed alongside the response.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI / App Framework | [Streamlit](https://streamlit.io/) |
| Orchestration | [LangChain](https://www.langchain.com/) (`0.3.x`) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | `google/flan-t5-base` (via 🤗 Transformers) |
| Document Loaders | `PyPDFLoader`, `TextLoader` |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/chayancsk/studysupporter.git
cd studysupporter
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** This project pins `langchain==0.3.27`, `langchain-community==0.3.27`, and `transformers==4.44.2` for compatibility. Newer `langchain` (1.x) removed the classic `langchain.chains` module used here, and mismatched `transformers` versions can break pipeline task recognition.

---

## ▶️ Usage

Run the app with the Streamlit CLI — **not** your IDE's run/play button, since that won't start the actual Streamlit server:

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

**Steps in the app:**
1. Upload one or more PDF/`.txt` files using the file uploader
2. Click **Process Documents** to index them (embeddings + FAISS indexing happen here)
3. Ask a question in the chat input
4. View the generated answer along with the source document(s) it was drawn from

---

## 📁 Project Structure

```
studysupporter/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration Notes

- **Model choice:** `flan-t5-base` was selected specifically because it runs reasonably fast on CPU-only hardware — no GPU required.
- **Chunking:** `chunk_size` and `chunk_overlap` can be tuned in `app.py` to balance retrieval granularity vs. context preservation.
- **Retrieval count (`top_k`):** Controls how many chunks are retrieved per query; higher values give more context but slower inference.

---

## 🐞 Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` for any package | Run `pip install -r requirements.txt` inside the activated `.venv` |
| `ModuleNotFoundError: langchain.text_splitter` | Install `langchain-text-splitters` and import from it directly |
| `ModuleNotFoundError: langchain.chains` | Ensure `langchain==0.3.27` is installed, not a 1.x release |
| `KeyError: Unknown task text2text-generation` | Pin `transformers==4.44.2`, then **fully restart** the Streamlit process (it won't hot-reload changed packages mid-session) |
| `IndexError: list index out of range` on `FAISS.from_documents` | Usually means no text was extracted — check for scanned/image-only PDFs, which need OCR instead of `PyPDFLoader` |
| App runs but `st.title`/`st.set_page_config` seem to do nothing when run via `python app.py` | Use `streamlit run app.py` instead — Streamlit commands require the Streamlit server context |

---

## 🚀 Roadmap / Future Improvements

- [ ] OCR support for scanned/image-based PDFs
- [ ] Swap in a larger or hosted LLM for higher-quality answers
- [ ] Persistent vector store (currently rebuilt each session)
- [ ] Chat history export
- [ ] Support for `.docx` and `.md` uploads

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

Built by [**chayancsk**](https://github.com/chayancsk) as part of a Data Science portfolio project.
